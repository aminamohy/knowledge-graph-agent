"""
graph.py — LangGraph StateGraph definition for the Knowledge Graph Agent.

Flow:
  classifier → (chitchat/out_of_scope/agent_info) → response → END
  classifier → (add/edit/delete/inquire)           → cypher → executor
                                                    ↑           ↓ (error & retry < 3)
                                                    └───────────┘
                                                                ↓ (success or retry limit)
                                                             response → END
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    intent_classifier_node,
    generate_cypher_node,
    execute_cypher_node,
    generate_response_node,
)

MAX_RETRIES = 3

# ─────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────
workflow = StateGraph(AgentState)

# ── Nodes ────────────────────────────────────────────────────
workflow.add_node("classifier", intent_classifier_node)
workflow.add_node("cypher",     generate_cypher_node)
workflow.add_node("executor",   execute_cypher_node)
workflow.add_node("response",   generate_response_node)

# ── Entry point ──────────────────────────────────────────────
workflow.set_entry_point("classifier")


# ── Routing: after classifier ─────────────────────────────────
def route_after_classification(state: AgentState) -> str:
    """
    Non-graph intents skip Cypher entirely and go straight to response.
    All data-related intents need a Cypher query first.
    """
    if state["intent"] in ("chitchat", "out_of_scope", "agent_info"):
        return "response"
    return "cypher"


workflow.add_conditional_edges(
    "classifier",
    route_after_classification,
    {"response": "response", "cypher": "cypher"},
)

# classifier → cypher is already handled above via conditional edges
workflow.add_edge("cypher", "executor")


# ── Routing: after executor ───────────────────────────────────
def route_after_execution(state: AgentState) -> str:
    """
    Retry Cypher generation if there was an execution error AND we
    haven't exceeded MAX_RETRIES. Otherwise proceed to response.
    """
    has_error = bool(state.get("error_message"))
    under_limit = state.get("retry_count", 0) < MAX_RETRIES

    if has_error and under_limit:
        return "cypher"  # retry

    return "response"


workflow.add_conditional_edges(
    "executor",
    route_after_execution,
    {"cypher": "cypher", "response": "response"},
)

# ── Terminal edge ─────────────────────────────────────────────
workflow.add_edge("response", END)

# ─────────────────────────────────────────────────────────────
# Memory & Compilation
# ─────────────────────────────────────────────────────────────
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)