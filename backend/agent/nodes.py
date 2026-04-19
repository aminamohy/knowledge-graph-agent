"""
nodes.py — LangGraph node functions for the Knowledge Graph Agent.

Each function receives the current AgentState and returns a partial
state dict that LangGraph merges back using the declared reducers.
"""
from dotenv import load_dotenv
load_dotenv()
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages

from .state import AgentState
from .tools.database import db_manager

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# LLM + Message Trimmer
# ─────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

trimmer = trim_messages(
    max_tokens=2000,
    strategy="last",
    token_counter=llm,
    start_on="human",
    include_system=True,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
PRONOUNS = {"she", "he", "they", "her", "him", "them", "it"}

VALID_INTENTS = {
    "add", "delete", "edit", "inquire",
    "chitchat", "out_of_scope", "agent_info",
}

MUTABLE_RELATIONSHIPS = {"WORKS_AT", "LIVES_IN", "STUDIES_AT", "HAS_AGE"}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def safe(value, default: str = "unknown") -> str:
    """Convert any value to a safe string."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return str(value)
    return str(value)


def normalize_neo4j_result(result: list) -> list:
    """
    Flatten Neo4j record dicts so each value is a plain scalar.

    Neo4j returns node objects as dicts like {'name': 'sama'}.
    This extracts the `name` field so downstream code sees plain strings.

    Input : [{'s': {'name': 'sama'}, 'r': 'WORKS_AT', 'c': {'name': 'google'}}]
    Output: [{'s': 'sama', 'r': 'WORKS_AT', 'c': 'google'}]
    """
    if not result:
        return []

    normalized = []
    for record in result:
        clean = {}
        for key, value in record.items():
            if isinstance(value, dict):
                clean[key] = value.get("name", str(value))
            else:
                clean[key] = value
        normalized.append(clean)
    return normalized


def validate_cypher(query: str) -> dict:
    """
    Basic structural validation before sending a query to Neo4j.

    Returns: {"is_valid": bool, "error": str | None}
    """
    if not query or not query.strip():
        return {"is_valid": False, "error": "Empty query"}

    q = query.lower()

    forbidden = ["drop", "detach delete *", "create constraint", "create index"]
    for token in forbidden:
        if token in q:
            return {"is_valid": False, "error": f"Forbidden clause: '{token}'"}

    if "return" not in q:
        return {"is_valid": False, "error": "Query is missing a RETURN clause"}

    if not any(kw in q for kw in ("match", "merge")):
        return {"is_valid": False, "error": "Query must contain MATCH or MERGE"}

    return {"is_valid": True, "error": None}


def resolve_pronoun(current_input: str, messages: list) -> str:
    """
    If the current message contains a pronoun, scan recent messages
    for the most recent non-pronoun noun to use as the subject.

    Returns the resolved subject string, or "unknown" if none found.
    """
    words = current_input.lower().split()

    if not any(w in PRONOUNS for w in words):
        return "unknown"

    for msg in reversed(messages[:-1]):
        if hasattr(msg, "content"):
            for word in msg.content.lower().split():
                if word not in PRONOUNS and word.isalpha() and len(word) > 2:
                    return word

    return "unknown"


# ─────────────────────────────────────────────────────────────
# NODE 1 — Intent Classifier
# ─────────────────────────────────────────────────────────────
def intent_classifier_node(state: AgentState) -> dict:
    """
    Classify the user's latest message into one of VALID_INTENTS.
    Falls back to 'chitchat' if the model returns an unrecognised label.
    """
    from .prompts import classify_intent

    user_message = state["messages"][-1].content
    logger.info("[Classifier] Input: %s", user_message)

    messages = trimmer.invoke(state["messages"])
    messages = messages + [("system", classify_intent)]

    response = llm.invoke(messages)
    raw = response.content.strip().lower().split()[0]
    intent = raw if raw in VALID_INTENTS else "chitchat"

    logger.info("[Classifier] Intent: %s", intent)
    return {"intent": intent}


# ─────────────────────────────────────────────────────────────
# NODE 2 — Cypher Generator
# ─────────────────────────────────────────────────────────────
def generate_cypher_node(state: AgentState) -> dict:
    """
    Generate a Cypher query for add / edit / delete / inquire intents.

    Uses string .replace() for placeholder substitution (not .format())
    to avoid conflicts with Cypher's own curly-brace syntax.
    """
    from .prompts import generate_cypher

    intent = state["intent"]

    # Non-graph intents don't need Cypher
    if intent in ("chitchat", "out_of_scope", "agent_info"):
        return {"generated_query": ""}

    user_input = state["messages"][-1].content.lower()
    error = safe(state.get("error_message"), "none")
    resolved_subject = resolve_pronoun(user_input, state["messages"])

    prompt = (
        generate_cypher
        .replace("{resolved_subject}", resolved_subject)
        .replace("{user_input}", user_input)
        .replace("{intent}", intent)
        .replace("{error_message}", error)
    )

    trimmed = trimmer.invoke(state["messages"])
    response = llm.invoke(trimmed + [("system", prompt)])

    # Strip Markdown code fences if the model added them
    query = (
        response.content
        .replace("```cypher", "")
        .replace("```", "")
        .strip()
    )

    logger.info("[CypherGen] Generated:\n%s", query)

    validation = validate_cypher(query)
    if not validation["is_valid"]:
        logger.warning("[CypherGen] Invalid query: %s", validation["error"])
        return {
            "generated_query": "",
            "error_message": validation["error"],
        }

    return {
        "generated_query": query,
        "error_message": None,
    }


# ─────────────────────────────────────────────────────────────
# NODE 3 — Cypher Executor
# ─────────────────────────────────────────────────────────────
def execute_cypher_node(state: AgentState) -> dict:
    """
    Execute the generated Cypher query against Neo4j.

    For 'inquire' intents with empty results, falls back to LlamaIndex.
    Tracks retry_count for the graph's retry routing logic.
    """
    intent = state.get("intent", "unknown")

    # Pass-through for intents that never touch the database
    if intent in ("chitchat", "out_of_scope", "agent_info"):
        return {
            "database_result": "N/A",
            "error_message": None,
            "fallback_used": "N/A",
            "retry_count": 0,
        }

    try:
        cypher_query = state.get("generated_query", "")

        # ── Inquire flow ──────────────────────────────────────
        if intent == "inquire":
            result = []

            if cypher_query:
                result = db_manager.execute_query(cypher_query)
                logger.info("[Executor] Cypher result: %s", result)
                result = normalize_neo4j_result(result)

            if not result:
                logger.info("[Executor] Empty result — trying LlamaIndex fallback")
                return _llama_fallback(state)

            return {
                "database_result": str(result),
                "error_message": None,
                "fallback_used": "cypher",
                "retry_count": 0,
            }

        # ── Write flow (add / edit / delete) ─────────────────
        raw_result = db_manager.execute_query(cypher_query)
        logger.info("[Executor] Write result: %s", raw_result)

        raw_result = normalize_neo4j_result(raw_result)

        if intent == "delete" and raw_result in ([], [{}]):
            result_str = "DELETE_SUCCESS"
        else:
            result_str = str(raw_result)

        return {
            "database_result": result_str,
            "error_message": None,
            "fallback_used": "cypher",
            "retry_count": 0,
        }

    except Exception as exc:
        logger.error("[Executor] Error: %s", exc)
        return {
            "error_message": str(exc),
            "retry_count": state.get("retry_count", 0) + 1,
        }


def _llama_fallback(state: AgentState) -> dict:
    """Run LlamaIndex fallback query. Returns a state-compatible dict."""
    try:
        from .tools.database import build_llama_engine

        engine = build_llama_engine()
        if engine is None:
            raise RuntimeError("LlamaIndex engine could not be built")

        user_query = state["messages"][-1].content
        llama_response = engine.query(user_query)
        result_str = str(llama_response).strip() or "No data found"

        logger.info("[LlamaFallback] Response: %s", result_str)
        return {
            "database_result": result_str,
            "error_message": None,
            "fallback_used": "llama_index",
            "retry_count": 0,
        }

    except Exception as exc:
        logger.warning("[LlamaFallback] Failed: %s", exc)
        return {
            "database_result": "No data found",
            "error_message": None,
            "fallback_used": "none",
            "retry_count": 0,
        }


# ─────────────────────────────────────────────────────────────
# NODE 4 — Response Generator
# ─────────────────────────────────────────────────────────────
def generate_response_node(state: AgentState) -> dict:
    """
    Generate the final natural-language response for the user.
    Uses database_result and intent to fill the response prompt.
    """
    from .prompts import generate_response

    intent = safe(state.get("intent"))
    db_res = safe(state.get("database_result"))

    logger.info("[Response] intent=%s db_result=%s", intent, db_res)

    prompt = generate_response.format(
        intent=intent,
        database_result=db_res,
    )

    trimmed = trimmer.invoke(state["messages"])
    response = llm.invoke(trimmed + [("system", prompt)])

    return {"messages": [("assistant", response.content)]}