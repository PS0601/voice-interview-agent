"""Run the interview agent evals and write evals/results.md + evals/results.json.

    uv run python evals/run_evals.py                 # retrieval + behavior evals, 3 runs per case
    uv run python evals/run_evals.py --runs 5        # more runs = better stability estimate
    uv run python evals/run_evals.py --retrieval-only  # no LLM calls, no API key needed

Behavior evals call the real LLM (DEEPSEEK_API_KEY from .env.local) through the same LangGraph
workflow the voice agent uses; only the speech-to-text / text-to-speech layer is skipped.
"""
import argparse
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import graph  # noqa: E402
from cases import CASES, RETRIEVAL_CASES  # noqa: E402


def to_messages(history):
    return [AIMessage(content=t) if role == "ai" else HumanMessage(content=t) for role, t in history]


def grade(case, new_messages):
    """Apply a case's checks to the messages the agent produced. Returns {check_name: (passed, detail)}."""
    called = [tc["name"] for m in new_messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    reply = next((m.content for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content), "")
    low = reply.lower()
    c = case["checks"]
    results = {}
    for tool in c.get("calls", []):
        results[f"calls {tool}"] = (tool in called, f"called={called}")
    for tool in c.get("not_calls", []):
        results[f"does not call {tool}"] = (tool not in called, f"called={called}")
    if "reply_contains_any" in c:
        hit = [k for k in c["reply_contains_any"] if k.lower() in low]
        results["reply has expected content"] = (bool(hit), f"matched={hit}")
    if "reply_not_matches" in c:
        m = re.search(c["reply_not_matches"], reply, flags=re.I)
        results["no invented facts"] = (m is None, f"matched={m.group(0)!r}" if m else "")
    if "max_words" in c:
        n = len(reply.split())
        results[f"concise (<= {c['max_words']} words)"] = (n <= c["max_words"], f"{n} words")
    results["non-empty reply"] = (bool(reply.strip()), "")
    return results, reply, called


def run_retrieval_eval(retriever):
    rows = []
    for question, expected in RETRIEVAL_CASES:
        docs = retriever.invoke(question)
        text = docs[0].page_content.lower() if docs else ""
        rows.append({"question": question, "expected_any": expected,
                     "hit": any(e.lower() in text for e in expected)})
    return rows


def run_behavior_eval(retriever, runs):
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        wf = graph.create_workflow(retriever=retriever, answers_path=Path(tmp) / "answers.txt")
        for case in CASES:
            for i in range(runs):
                inputs = to_messages(case["history"])
                start = time.perf_counter()
                try:
                    out = wf.invoke({"messages": inputs})["messages"]
                    checks, reply, called = grade(case, out[len(inputs):])
                    error = None
                except Exception as e:  # an API error counts as a failed run, not a crash
                    checks, reply, called, error = {"ran without error": (False, str(e))}, "", [], str(e)
                latency = time.perf_counter() - start
                passed = all(ok for ok, _ in checks.values())
                records.append({"id": case["id"], "category": case["category"], "run": i + 1,
                                "passed": passed, "latency_s": round(latency, 2), "tools_called": called,
                                "checks": {k: {"passed": ok, "detail": d} for k, (ok, d) in checks.items()},
                                "reply": reply, "error": error})
                mark = "PASS" if passed else "FAIL"
                print(f"  [{mark}] {case['id']} run {i + 1} ({latency:.1f}s)")
    return records


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def write_report(retrieval, behavior, runs, out_dir):
    lines = [f"# Eval results", "",
             f"Run on {datetime.now():%Y-%m-%d %H:%M} · model `{graph.os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}` "
             f"· embeddings all-MiniLM-L6-v2 (local) · {runs} run(s) per behavior case", ""]

    hits = sum(r["hit"] for r in retrieval)
    lines += ["## Retrieval (top-1 chunk contains the needed fact)", "",
              f"**Hit rate @1: {hits}/{len(retrieval)} ({pct(hits, len(retrieval))})**", "",
              "| Question | Hit |", "|---|---|"]
    lines += [f"| {r['question']} | {'✅' if r['hit'] else '❌'} |" for r in retrieval]

    if behavior:
        by_cat, by_case = defaultdict(list), defaultdict(list)
        for r in behavior:
            by_cat[r["category"]].append(r)
            by_case[r["id"]].append(r)
        total = sum(r["passed"] for r in behavior)
        lat = sorted(r["latency_s"] for r in behavior)
        p50, p95 = lat[len(lat) // 2], lat[min(len(lat) - 1, int(len(lat) * 0.95))]
        stable = sum(len({r["passed"] for r in rs}) == 1 for rs in by_case.values())
        lines += ["", "## Agent behavior (LangGraph + LLM, text only)", "",
                  f"**Overall pass rate: {total}/{len(behavior)} ({pct(total, len(behavior))})** · "
                  f"latency p50 {p50:.1f}s / p95 {p95:.1f}s · "
                  f"consistent across runs: {stable}/{len(by_case)} cases", "",
                  "| Category | Pass rate |", "|---|---|"]
        lines += [f"| {cat} | {sum(r['passed'] for r in rs)}/{len(rs)} ({pct(sum(r['passed'] for r in rs), len(rs))}) |"
                  for cat, rs in by_cat.items()]
        lines += ["", "| Case | Passed runs | Failed checks |", "|---|---|---|"]
        for cid, rs in by_case.items():
            failed = sorted({k for r in rs for k, v in r["checks"].items() if not v["passed"]})
            lines.append(f"| {cid} | {sum(r['passed'] for r in rs)}/{len(rs)} | {', '.join(failed) or '-'} |")

    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps({"retrieval": retrieval, "behavior": behavior}, indent=2),
                                          encoding="utf-8")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="runs per behavior case (default 3)")
    ap.add_argument("--retrieval-only", action="store_true", help="skip LLM calls")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as db:
        retriever = graph.build_retriever(persist_directory=db)  # fresh index, real local embeddings
        print("Retrieval eval...")
        retrieval = run_retrieval_eval(retriever)
        behavior = []
        if not args.retrieval_only:
            print(f"Behavior eval ({len(CASES)} cases x {args.runs} runs)...")
            behavior = run_behavior_eval(retriever, args.runs)
    write_report(retrieval, behavior, args.runs, out_dir)


if __name__ == "__main__":
    main()
