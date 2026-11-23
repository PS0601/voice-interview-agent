"""Run the interview agent evals and write evals/results.md + evals/results.json.

    uv run python evals/run_evals.py                 # retrieval + behavior evals, 3 runs per case
    uv run python evals/run_evals.py --runs 5        # more runs = better stability estimate
    uv run python evals/run_evals.py --retrieval-only  # no LLM calls, no API key needed
    uv run python evals/run_evals.py --sweep           # compare chunk sizes / top-k (no LLM calls)

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
from cases import CASES, RETRIEVAL_CASES, RETRIEVAL_HOLDOUT  # noqa: E402


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


def run_retrieval_eval(retriever, cases=RETRIEVAL_CASES, split="tuning"):
    """A question is a hit if the chunks the tool would return (all top-k) contain the needed fact."""
    rows = []
    for question, expected in cases:
        docs = retriever.invoke(question)
        text = " ".join(d.page_content for d in docs).lower()
        rows.append({"question": question, "expected_any": expected, "split": split,
                     "hit": any(e.lower() in text for e in expected), "context_chars": len(text)})
    return rows


SWEEP = [(1000, 200, 1), (1000, 200, 2), (500, 100, 1), (500, 100, 2), (500, 100, 3),
         (300, 60, 1), (300, 60, 2), (300, 60, 3), (200, 40, 2), (200, 40, 3)]


def run_sweep(out_dir):
    """Try several chunk sizes and top-k values. Choose using the tuning set only, then report the
    held-out score for the chosen setting so the result isn't overfitted to the tuning questions."""
    rows = []
    for size, overlap, k in SWEEP:
        with tempfile.TemporaryDirectory() as db:
            r = graph.build_retriever(persist_directory=db, chunk_size=size, chunk_overlap=overlap, k=k)
            n_chunks = r.vectorstore._collection.count()
            tune = run_retrieval_eval(r, RETRIEVAL_CASES, "tuning")
            hold = run_retrieval_eval(r, RETRIEVAL_HOLDOUT, "holdout")
        ctx = sum(x["context_chars"] for x in tune + hold) / len(tune + hold)
        rows.append({"chunk_size": size, "overlap": overlap, "k": k, "chunks": n_chunks,
                     "tuning_hits": sum(x["hit"] for x in tune), "holdout_hits": sum(x["hit"] for x in hold),
                     "avg_context_chars": round(ctx)})
        print(f"  size={size:<5} overlap={overlap:<4} k={k}  tuning {rows[-1]['tuning_hits']}/{len(tune)}  "
              f"holdout {rows[-1]['holdout_hits']}/{len(hold)}  ~{rows[-1]['avg_context_chars']} chars")
    # pick: best tuning score, then the least context (cheaper, more focused prompts)
    best = max(rows, key=lambda r: (r["tuning_hits"], -r["avg_context_chars"]))
    nt, nh = len(RETRIEVAL_CASES), len(RETRIEVAL_HOLDOUT)
    lines = ["# Retrieval sweep", "",
             "Chosen on the tuning set only (best hit rate, then smallest context). "
             "The held-out column shows whether that choice generalises.", "",
             "| Chunk size | Overlap | Top-k | Chunks | Tuning hits | Held-out hits | Avg context (chars) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " ⭐" if r is best else ""
        lines.append(f"| {r['chunk_size']}{mark} | {r['overlap']} | {r['k']} | {r['chunks']} | "
                     f"{r['tuning_hits']}/{nt} | {r['holdout_hits']}/{nh} | {r['avg_context_chars']} |")
    lines += ["", f"**Chosen:** chunk size {best['chunk_size']}, overlap {best['overlap']}, top-k {best['k']}. "
              f"To use it, add to `.env.local`:", "", "```",
              f"RAG_CHUNK_SIZE={best['chunk_size']}", f"RAG_CHUNK_OVERLAP={best['overlap']}", f"RAG_TOP_K={best['k']}", "```"]
    (out_dir / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


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

    tune = [r for r in retrieval if r["split"] == "tuning"]
    hold = [r for r in retrieval if r["split"] == "holdout"]
    th, hh = sum(r["hit"] for r in tune), sum(r["hit"] for r in hold)
    lines += [f"## Retrieval (returned chunks contain the needed fact)", "",
              f"Settings: chunk size {graph.DEFAULT_CHUNK_SIZE}, overlap {graph.DEFAULT_CHUNK_OVERLAP}, "
              f"top-k {graph.DEFAULT_TOP_K}", "",
              f"**Tuning set: {th}/{len(tune)} ({pct(th, len(tune))}) · "
              f"Held-out set: {hh}/{len(hold)} ({pct(hh, len(hold))})**", "",
              "| Question | Set | Hit |", "|---|---|---|"]
    lines += [f"| {r['question']} | {r['split']} | {'✅' if r['hit'] else '❌'} |" for r in retrieval]

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
    ap.add_argument("--sweep", action="store_true", help="compare chunk sizes / top-k (no LLM calls)")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    if args.sweep:
        print("Retrieval sweep...")
        run_sweep(out_dir)
        return
    with tempfile.TemporaryDirectory() as db:
        retriever = graph.build_retriever(persist_directory=db)  # fresh index, real local embeddings
        print("Retrieval eval...")
        retrieval = run_retrieval_eval(retriever) + run_retrieval_eval(retriever, RETRIEVAL_HOLDOUT, "holdout")
        behavior = []
        if not args.retrieval_only:
            print(f"Behavior eval ({len(CASES)} cases x {args.runs} runs)...")
            behavior = run_behavior_eval(retriever, args.runs)
    write_report(retrieval, behavior, args.runs, out_dir)


if __name__ == "__main__":
    main()
