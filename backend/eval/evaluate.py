import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    from langsmith import Client
    from langsmith.evaluation import evaluate as langsmith_evaluate
    LANGSMITH_AVAILABLE = True
    logger.info("LangSmith loaded successfully")
except Exception as e:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith could not be loaded: %s", e)

from agent.graph import app as agent_app
from agent.nodes import validate_cypher

EVAL_DATASET = [
    {"input": "Sama works at Google",                     "expected_intent": "add",          "category": "intent"},
    {"input": "Sama now works at Meta instead of Google", "expected_intent": "edit",         "category": "intent"},
    {"input": "Who works at Google?",                     "expected_intent": "inquire",      "category": "intent"},
    {"input": "Forget that Sama works at Google",         "expected_intent": "delete",       "category": "intent"},
    {"input": "Hello!",                                   "expected_intent": "chitchat",     "category": "intent"},
    {"input": "What's the weather in Cairo?",             "expected_intent": "out_of_scope", "category": "intent"},
    {"input": "What can you do?",                         "expected_intent": "agent_info",   "category": "intent"},
    {"input": "Amina is 28 and lives in Giza",            "expected_intent": "add",          "category": "e2e"},
    {"input": "Where does Amina live?",                   "expected_intent": "inquire",      "category": "e2e"},
    {"input": "Tell me about machine learning",           "expected_intent": "inquire",      "category": "e2e"},
]


def run_agent(inputs: dict) -> dict:
    message    = inputs.get("input", "")
    session_id = inputs.get("session_id", f"eval-{os.urandom(4).hex()}")
    config        = {"configurable": {"thread_id": session_id}}
    initial_state = {"messages": [("user", message)]}
    try:
        final_state = agent_app.invoke(initial_state, config=config)
        last_msg    = final_state["messages"][-1]
        reply       = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        return {
            "reply":           reply,
            "intent":          final_state.get("intent", "unknown"),
            "generated_query": final_state.get("generated_query", ""),
            "fallback_used":   final_state.get("fallback_used", "N/A"),
        }
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        return {"reply": f"ERROR: {exc}", "intent": "error", "generated_query": ""}


def intent_exact_match(run, example) -> dict:
    predicted = (run.outputs or {}).get("intent", "")
    expected  = (example.outputs or {}).get("expected_intent", "")
    score     = 1 if predicted == expected else 0
    return {"key": "intent_accuracy", "score": score, "comment": f"predicted={predicted!r} expected={expected!r}"}


def cypher_validity_check(run, example) -> dict:
    query = (run.outputs or {}).get("generated_query", "")
    if not query:
        return {"key": "cypher_valid", "score": 0, "comment": "No query generated"}
    result = validate_cypher(query)
    return {"key": "cypher_valid", "score": 1 if result["is_valid"] else 0, "comment": result.get("error") or "valid"}


def run_local_evaluation():
    print("\n" + "=" * 60)
    print("  Knowledge Graph Agent — Local Evaluation")
    print("=" * 60)
    results = []
    for i, sample in enumerate(EVAL_DATASET, 1):
        user_input = sample["input"]
        expected   = sample.get("expected_intent", "")
        category   = sample["category"]
        print(f"\n[{i}/{len(EVAL_DATASET)}] {category.upper()} | {user_input!r}")
        output    = run_agent({"input": user_input, "session_id": f"eval-{i}"})
        intent_ok = output["intent"] == expected
        has_query = bool(output.get("generated_query"))
        cypher_ok = validate_cypher(output["generated_query"])["is_valid"] if has_query else None
        results.append({"input": user_input, "expected_intent": expected, "predicted_intent": output["intent"], "intent_correct": intent_ok, "cypher_valid": cypher_ok, "reply": output["reply"]})
        print(f"  Intent : {output['intent']!r} (expected {expected!r}) {'OK' if intent_ok else 'FAIL'}")
        if has_query:
            print(f"  Cypher : {'valid' if cypher_ok else 'invalid'}")
        print(f"  Reply  : {output['reply'][:120]}")
    total     = len(results)
    intent_ok = sum(1 for r in results if r["intent_correct"])
    cypher_all = [r for r in results if r["cypher_valid"] is not None]
    cypher_ok  = sum(1 for r in cypher_all if r["cypher_valid"])
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Intent accuracy : {intent_ok}/{total} ({100 * intent_ok // total}%)")
    if cypher_all:
        print(f"  Cypher valid    : {cypher_ok}/{len(cypher_all)} ({100 * cypher_ok // len(cypher_all)}%)")
    print("=" * 60 + "\n")
    return results


def run_langsmith_evaluation(dataset_name: str = "kg-agent-eval"):
    if not LANGSMITH_AVAILABLE:
        logger.error("LangSmith not available.")
        return
    client   = Client()
    existing = [d.name for d in client.list_datasets()]
    if dataset_name not in existing:
        logger.info("Creating LangSmith dataset: %s", dataset_name)
        dataset = client.create_dataset(dataset_name, description="KG Agent eval set")
        client.create_examples(
            inputs  = [{"input": s["input"]} for s in EVAL_DATASET],
            outputs = [{"expected_intent": s.get("expected_intent", "")} for s in EVAL_DATASET],
            dataset_id=dataset.id,
        )
    else:
        logger.info("Reusing dataset: %s", dataset_name)
    logger.info("Running LangSmith evaluation...")
    results = langsmith_evaluate(
        run_agent,
        data       = dataset_name,
        evaluators = [intent_exact_match, cypher_validity_check],
        experiment_prefix="kg-agent",
        metadata={"model": "gpt-4o-mini", "version": "1.0"},
    )
    print("\nLangSmith evaluation complete.")
    print("   View at: https://smith.langchain.com/")
    return results


if __name__ == "__main__":
    use_langsmith = bool(os.getenv("LANGCHAIN_API_KEY"))
    if use_langsmith:
        print("LangSmith API key found — running cloud evaluation.")
        run_langsmith_evaluation()
    else:
        print("No LANGCHAIN_API_KEY — running local evaluation.")
        run_local_evaluation()