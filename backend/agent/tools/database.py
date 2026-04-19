"""
database.py — Neo4j connection manager + LlamaIndex fallback engine.

Two responsibilities:
  1. Neo4jManager  — raw Cypher execution (CRUD) via the official driver.
  2. build_llama_engine() — builds a fresh LlamaIndex query engine each call,
     ensuring it always sees the latest Neo4j schema.
"""

import os
import logging

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


# ─────────────────────────────────────────────────────────────
# LlamaIndex Fallback Engine
# Rebuilt fresh on every call so it always picks up the latest
# Neo4j schema (avoids stale schema from a cached instance).
# ─────────────────────────────────────────────────────────────
def build_llama_engine():
    """
    Build and return a LlamaIndex query engine backed by Neo4j.

    Called only when Cypher execution returns an empty result for
    an 'inquire' intent (fallback path).

    Returns:
        A LlamaIndex query engine, or None if dependencies are missing.
    """
    try:
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
        from llama_index.core import PropertyGraphIndex
        from llama_index.llms.openai import OpenAI as LlamaOpenAI

        graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
            database="neo4j",
            # refresh_schema=True ensures the store re-reads node labels
            # and property keys on every query — critical after writes.
        )

        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=LlamaOpenAI(model="gpt-4o-mini"),
        )

        return index.as_query_engine(
            include_text=True,
            # tree_summarize synthesises multiple matching nodes into one
            # coherent answer instead of returning only the first result.
            response_mode="tree_summarize",
        )

    except ImportError as e:
        logger.warning("LlamaIndex not installed — fallback disabled. %s", e)
        return None
    except Exception as e:
        logger.error("Failed to build LlamaIndex engine: %s", e)
        return None


# ─────────────────────────────────────────────────────────────
# Neo4j Manager
# ─────────────────────────────────────────────────────────────
class Neo4jManager:
    """Handles all direct Cypher operations (CRUD) against Neo4j."""

    def __init__(self):
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self._driver = None

    # ── Connection ────────────────────────────────────────────

    def connect(self):
        """Lazy-initialise the driver (called before every query)."""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                self._driver.verify_connectivity()
                logger.info("Connected to Neo4j at %s", self.uri)
            except AuthError as e:
                raise RuntimeError(
                    f"Neo4j authentication failed. Check NEO4J_USER / NEO4J_PASSWORD. ({e})"
                ) from e
            except ServiceUnavailable as e:
                raise RuntimeError(
                    f"Neo4j is unreachable at {self.uri}. Is the database running? ({e})"
                ) from e

    def close(self):
        """Close the driver (call on application shutdown)."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    # ── Query Execution ───────────────────────────────────────

    def execute_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        """
        Run a Cypher query and return results as a list of dicts.

        Args:
            query:      Valid Cypher query string.
            parameters: Optional parameter map for parameterised queries.

        Returns:
            List of record dicts. Empty list if query produces no rows.

        Raises:
            RuntimeError: On connection or query-execution failure.
        """
        self.connect()
        parameters = parameters or {}

        try:
            with self._driver.session() as session:
                result = session.run(query, parameters)
                records = [record.data() for record in result]
                logger.debug("Query returned %d record(s).", len(records))
                return records

        except Exception as e:
            logger.error("Cypher execution error: %s\nQuery: %s", e, query)
            raise

    # ── Health Check ──────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Return True if Neo4j is reachable, False otherwise."""
        try:
            self.connect()
            return True
        except Exception:
            return False


# Singleton — imported and used throughout the agent
db_manager = Neo4jManager()