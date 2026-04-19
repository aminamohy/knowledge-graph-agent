from typing import Annotated, Optional, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Core conversation history — managed by LangGraph's add_messages reducer
    messages: Annotated[list, add_messages]

    # Classified intent: add | inquire | edit | delete | chitchat | out_of_scope | agent_info
    intent: str

    # Cypher query produced by the generator node
    generated_query: str

    # Raw result returned from Neo4j (or LlamaIndex fallback)
    database_result: Union[dict, list, str]

    # Which retrieval path was used: "cypher" | "llama_index" | "none" | "N/A"
    fallback_used: Optional[str]

    # How many times Cypher generation has been retried after execution failure
    retry_count: int

    # Last execution error, cleared when execution succeeds
    error_message: Optional[str]