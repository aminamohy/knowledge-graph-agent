# 🕸️ Knowledge Graph Agent

A conversational AI agent that stores, retrieves, and manages knowledge as a **graph of relationships** — powered by Neo4j, LangGraph, GPT-4o-mini, and FastAPI.

---

##  Features

| Feature | Description |
|--------|-------------|
|  **Intent Classification** | Automatically detects user intent: add, inquire, edit, delete, chitchat, out_of_scope, agent_info |
| 🔗 **Graph Storage** | Stores facts as entity relationships in Neo4j (e.g. `Sama -[WORKS_AT]-> Google`) |
|  **Multi-turn Memory** | Remembers conversation context within a session — resolves pronouns like "she", "he", "they" |
|  **Persistent Long-term Memory** | Facts survive server restarts — stored permanently in Neo4j |
|  **Auto-retry on Error** | If Cypher execution fails, the agent regenerates the query automatically (up to 3 retries) |
|  **Cypher Validation** | All generated queries are validated before execution — blocks DROP, DETACH DELETE *, etc. |
|  **LangSmith Evaluation** | Intent accuracy and Cypher validity tracked via LangSmith cloud |
|  **REST API** | FastAPI backend with session support, health check, and CORS |
|  **React Frontend** | Clean dark-themed chat UI with intent badges, session management, and Neo4j health indicator |

---

##  System Architecture


```
![System Architecture](https://github.com/aminamohy/knowledge-graph-agent/blob/main/arch.png)
---

##  AI System Design

### Agent Nodes

| Node | Responsibility |
|------|---------------|
| `classifier` | Classifies user message into one of 7 intents using GPT-4o-mini |
| `cypher` | Generates a Neo4j Cypher query from the user message and intent |
| `executor` | Executes the Cypher query against Neo4j; falls back to LlamaIndex if empty |
| `response` | Generates a natural language response from the database result |

### Routing Logic

```
classifier
    ├── chitchat / out_of_scope / agent_info  ──→  response → END
    └── add / edit / delete / inquire         ──→  cypher → executor
                                                        │
                                              error & retry < 3?
                                                    yes → cypher (retry)
                                                    no  → response → END
```

### State Schema (`AgentState`)

```python
class AgentState(TypedDict):
    messages:        list          # full conversation history
    intent:          str           # classified intent
    generated_query: str           # Cypher query
    database_result: str           # Neo4j result
    fallback_used:   str           # "cypher" | "llama_index" | "none"
    retry_count:     int           # number of Cypher retries
    error_message:   str           # last execution error
```

---

##  Memory Architecture

### 1. Short-Term Memory (In-Context)
- **What:** The current conversation's message history
- **Where:** LangGraph `MemorySaver` — stored in RAM
- **Scope:** Single session only — lost on server restart
- **Used for:** Pronoun resolution ("she" → "Sama"), conversation continuity
- **Trimmed at:** 2000 tokens using `trim_messages` to avoid context overflow

### 2. Long-Term Memory (External Graph)
- **What:** All stored facts as entity relationships
- **Where:** Neo4j graph database — persisted to disk
- **Scope:** Permanent — survives server restarts, available across all sessions
- **Used for:** Answering queries like "Where does Sama work?" days later
- **Example:** `(sama:Entity)-[:WORKS_AT]->(google:Entity)`

### Memory Comparison

| | Short-Term | Long-Term |
|--|-----------|-----------|
| Storage | RAM (MemorySaver) | Neo4j (disk) |
| Lifetime | Session only | Permanent |
| Survives restart? | ❌ | ✅ |
| Scope | One conversation | All conversations |
| Purpose | Context & pronouns | Knowledge retrieval |

---

## 📁 Project Structure

```
KG_agent/
├── backend/
│   ├── main_api.py          # FastAPI app — endpoints, CORS, session handling
│   ├── .env                 # API keys and DB credentials
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py         # LangGraph StateGraph definition and routing
│   │   ├── nodes.py         # Node functions: classifier, cypher, executor, response
│   │   ├── prompts.py       # All LLM prompt templates
│   │   ├── state.py         # AgentState TypedDict definition
│   │   └── tools/
│   │       ├── __init__.py
│   │       └── database.py  # Neo4j manager + LlamaIndex fallback engine
│   └── eval/
│       ├── __init__.py
│       └── evaluate.py      # LangSmith evaluation suite
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── api/
        │   └── agentApi.js      # HTTP client for the FastAPI backend
        └── components/
            ├── ChatWindow.jsx   # Main chat interface + session management
            ├── MessageBubble.jsx # Single message with intent badge
            └── InputBar.jsx     # Auto-growing textarea + send button
```

---

##  API Reference

### `POST /chat`
Main conversation endpoint.

**Request:**
```json
{
  "message": "Sama works at Google",
  "session_id": "my-session-001"
}
```

**Response:**
```json
{
  "reply": "Got it! I've stored that Sama works at Google.",
  "session_id": "my-session-001",
  "intent": "add",
  "fallback_used": "cypher"
}
```

### `GET /health`
```json
{
  "status": "ok",
  "neo4j": "connected"
}
```

### `DELETE /session/{session_id}`
Signals session clear. Client should generate a new session_id.

---

##  Knowledge Graph Model

```
Nodes:    (:Entity {name: "lowercase_string"})
Relationships: UPPER_SNAKE_CASE

Examples:
  (sama)-[:WORKS_AT]->(google)
  (amina)-[:LIVES_IN]->(cairo)
  (sara)-[:STUDIES_AT]->(cairo university)
  (john)-[:KNOWS]->(amina)
  (sara)-[:HAS_AGE]->(28)
```

**Rules:**
- All node names are **lowercase**
- Nodes have **only one property**: `name`
- All facts are stored as **relationships**, never as node properties
- MERGE is used for add (idempotent), MATCH+DELETE for delete, MATCH+DELETE+MERGE for edit

---

##  Evaluation (LangSmith)

### Evaluators

| Evaluator | Method | Score |
|-----------|--------|-------|
| `intent_accuracy` | Exact match — predicted vs expected intent | 0 or 1 |
| `cypher_valid` | Structural validation — RETURN, MATCH/MERGE, forbidden clauses | 0 or 1 |

### Run Evaluation
```bash
cd backend
python -m eval.evaluate
```

Results are pushed to [smith.langchain.com](https://smith.langchain.com) under **Projects → knowledge-graph-agent**.

---

##  Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- Neo4j (Desktop or AuraDB)
- OpenAI API Key
- LangSmith API Key (optional)

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**`.env` file:**
```env
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# Optional — for LangSmith
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=knowledge-graph-agent
```

```bash
uvicorn main_api:api --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`

---

## 🧪 Manual Test Checklist

| Test | Input | Expected |
|------|-------|----------|
| Add fact | "Sama works at Google" | Stored confirmation |
| Inquire | "Where does Sama work?" | Google |
| Pronoun | "Where does she work?" (after Sama) | Resolves to Sama |
| Edit | "Sama now works at Meta instead of Google" | Updated |
| Confirm edit | "Where does Sama work?" | Meta (not Google) |
| Delete | "Forget that Sama works at Meta" | Removed |
| Confirm delete | "Where does Sama work?" | No data found |
| Out of scope | "What's the weather?" | Polite decline |
| Chitchat | "Hello!" | Greeting |
| Agent info | "What can you do?" | Capability description |
| Persistence | Store → restart server → inquire | Still returns data |
| Session isolation | Store in session 1 → new session → inquire | No data found |

## 🔄 Example Flow

User Input:
"Sama works at Google"

Generated Cypher:
MERGE (a:Entity {name:"sama"})
MERGE (b:Entity {name:"google"})
MERGE (a)-[:WORKS_AT]->(b)
RETURN a, b

Response:
"Got it! I've stored that Sama works at Google."
Pipeline:
User Input → Intent Classification → Cypher Generation → Neo4j Execution → Response Generation
---
## 📊 Evaluation Results

- **Intent Classification Accuracy:** ~90%
- **Cypher Generation Validity:** 100%
- Evaluated using LangSmith on multi-intent queries

**Insights:**
- Most errors occurred in paragraph classification
- Improved via prompt tuning and stricter intent rules
----------------------------
## 🚧 Challenges & Learnings

- Handling hallucinated Cypher queries from LLMs
- Ensuring consistent relationship schema
- Stabilizing intent classification for multi-sentence inputs
- Designing short-term vs long-term memory interaction
- Preventing fallback from generating ungrounded answers


## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o-mini (OpenAI) |
| Agent Framework | LangGraph |
| Graph Database | Neo4j |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite |
| Evaluation | LangSmith |
| Memory (short-term) | LangGraph MemorySaver |
| Memory (long-term) | Neo4j |
| Fallback Retrieval | LlamaIndex (optional) |
