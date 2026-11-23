# Eval results

Run on 2026-11-22 23:52 · model `deepseek-chat` · embeddings all-MiniLM-L6-v2 (local) · 3 run(s) per behavior case

## Retrieval (top-1 chunk contains the needed fact)

**Hit rate @1: 12/15 (80%)**

| Question | Hit |
|---|---|
| What benefits do you offer? | ✅ |
| How much paid time off do employees get? | ✅ |
| What programming languages and frameworks does the backend use? | ✅ |
| Where is the company headquartered? | ✅ |
| When was NovaLoop founded? | ✅ |
| What does the company's main product do? | ❌ |
| Who are your customers? | ✅ |
| How much revenue does the company make? | ✅ |
| What is the interview process like? | ✅ |
| Can I work remotely? | ✅ |
| Is there a learning or education budget? | ✅ |
| What does an AI Engineer do at NovaLoop? | ✅ |
| What is the company's mission? | ❌ |
| What are the company values? | ✅ |
| How many employees work there? | ❌ |

## Agent behavior (LangGraph + LLM, text only)

**Overall pass rate: 42/42 (100%)** · latency p50 2.4s / p95 3.2s · consistent across runs: 14/14 cases

| Category | Pass rate |
|---|---|
| Interview flow | 9/9 (100%) |
| Company Q&A (RAG) | 24/24 (100%) |
| No hallucination | 9/9 (100%) |

| Case | Passed runs | Failed checks |
|---|---|---|
| flow_q1_to_q2 | 3/3 | - |
| flow_q2_to_q3 | 3/3 | - |
| flow_q3_to_q4 | 3/3 | - |
| rag_benefits | 3/3 | - |
| rag_tech_stack | 3/3 | - |
| rag_location | 3/3 | - |
| rag_interview_process | 3/3 | - |
| rag_remote | 3/3 | - |
| rag_learning_budget | 3/3 | - |
| rag_role | 3/3 | - |
| rag_funding | 3/3 | - |
| honest_ceo_salary | 3/3 | - |
| honest_visa | 3/3 | - |
| honest_ceo_name | 3/3 | - |
