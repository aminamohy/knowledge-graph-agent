"""
main_api.py — FastAPI backend for the Knowledge Graph Agent.

Endpoints:
  POST /chat          — main conversation endpoint
  GET  /health        — liveness + Neo4j connectivity check
  DELETE /session/{id} — clear a session's conversation memory

Session handling:
  - Each request carries a session_id (UUID string).
  - LangGraph's MemorySaver maps thread_id → conversation history,
    so passing the same session_id restores the prior context.
  - No server-side session store needed; memory lives inside `app`.

Run:
  uvicorn backend.main_api:api --reload --host 0.0.0.0 --port 8000
"""
from dotenv import load_dotenv
load_dotenv()

import os
print("KEY:", os.getenv("OPENAI_API_KEY"))
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agent.graph import app as agent_app          # compiled LangGraph app
from backend.agent.tools.database import db_manager
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown hooks)
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Knowledge Graph Agent API...")
    yield
    logger.info("Shutting down — closing Neo4j driver.")
    db_manager.close()


# ─────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────
api = FastAPI(
    title="Knowledge Graph Agent API",
    description="LangGraph + Neo4j knowledge graph agent with session memory.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the React dev server and any local origin ───
api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React dev server (CRA / Vite default)
        "http://localhost:5173",   # Vite alternative port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation session ID. Pass the same ID to maintain context.",
    )

    model_config = {"json_schema_extra": {"example": {"message": "Sama works at Google", "session_id": "my-session-001"}}}


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID (echo back for client tracking)")
    intent: str = Field(..., description="Classified intent for this message")
    fallback_used: str = Field(..., description="Which retrieval path was used")


class HealthResponse(BaseModel):
    status: str
    neo4j: str


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@api.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the Knowledge Graph Agent",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversation endpoint.

    - Pass the same `session_id` across requests to maintain multi-turn memory.
    - Omit `session_id` to start a fresh conversation (a new UUID is generated).
    """
    logger.info("[/chat] session=%s message=%r", request.session_id, request.message)

    config = {"configurable": {"thread_id": request.session_id}}
    initial_state = {"messages": [("user", request.message)]}

    try:
        final_state = agent_app.invoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("[/chat] Agent error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {exc}",
        )

    # Extract the last assistant message
    last_message = final_state["messages"][-1]
    reply = last_message.content if hasattr(last_message, "content") else str(last_message)

    return ChatResponse(
        reply=reply,
        session_id=request.session_id,
        intent=final_state.get("intent", "unknown"),
        fallback_used=final_state.get("fallback_used", "N/A"),
    )


@api.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and Neo4j connectivity check",
)
async def health() -> HealthResponse:
    """Returns 200 if the API is up; indicates Neo4j status separately."""
    neo4j_status = "connected" if db_manager.is_healthy() else "unreachable"
    return HealthResponse(status="ok", neo4j=neo4j_status)


@api.delete(
    "/session/{session_id}",
    summary="Clear session memory",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_session(session_id: str):
    """
    Instructs the client to start a new session.

    LangGraph's MemorySaver keeps conversation history in memory keyed by
    thread_id. To clear, start using a new session_id on the next request —
    the old thread will eventually be garbage-collected.

    Note: If you need persistent cross-restart memory, swap MemorySaver for
    SqliteSaver or PostgresSaver and expose a proper delete here.
    """
    logger.info("[/session] Clear requested for session=%s", session_id)
    # With in-memory checkpointer, we cannot delete a specific thread.
    # Returning 204 signals to the client to generate a new session_id.
    return