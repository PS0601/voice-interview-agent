# Eval results

Run on 2026-11-23 04:54 · model `deepseek-chat` · embeddings all-MiniLM-L6-v2 (local) · 5 run(s) per behavior case

## Retrieval (returned chunks contain the needed fact)

Settings: chunk size 500, overlap 100, top-k 3

**Tuning set: 14/15 (93%) · Held-out set: 10/10 (100%)**

| Question | Set | Hit |
|---|---|---|
| What benefits do you offer? | tuning | ✅ |
| How much paid time off do employees get? | tuning | ✅ |
| What programming languages and frameworks does the backend use? | tuning | ✅ |
| Where is the company headquartered? | tuning | ✅ |
| When was NovaLoop founded? | tuning | ✅ |
| What does the company's main product do? | tuning | ❌ |
| Who are your customers? | tuning | ✅ |
| How much revenue does the company make? | tuning | ✅ |
| What is the interview process like? | tuning | ✅ |
| Can I work remotely? | tuning | ✅ |
| Is there a learning or education budget? | tuning | ✅ |
| What does an AI Engineer do at NovaLoop? | tuning | ✅ |
| What is the company's mission? | tuning | ✅ |
| What are the company values? | tuning | ✅ |
| How many employees work there? | tuning | ✅ |
| Does the company have an office in Canada? | holdout | ✅ |
| What does NovaLoop Flow do? | holdout | ✅ |
| Is there a 401k match, and how much? | holdout | ✅ |
| How big is the engineering team? | holdout | ✅ |
| How long is parental leave? | holdout | ✅ |
| How soon will I hear back after each interview round? | holdout | ✅ |
| Which systems do your products integrate with? | holdout | ✅ |
| How soon do new hires ship their first feature? | holdout | ✅ |
| How many clinics use NovaLoop? | holdout | ✅ |
| Which cloud provider do you run on? | holdout | ✅ |

## Agent behavior (LangGraph + LLM, text only)

**Overall pass rate: 70/70 (100%)** · latency p50 2.9s / p95 3.7s · consistent across runs: 14/14 cases

| Category | Pass rate |
|---|---|
| Interview flow | 15/15 (100%) |
| Company Q&A (RAG) | 40/40 (100%) |
| No hallucination | 15/15 (100%) |

| Case | Passed runs | Failed checks |
|---|---|---|
| flow_q1_to_q2 | 5/5 | - |
| flow_q2_to_q3 | 5/5 | - |
| flow_q3_to_q4 | 5/5 | - |
| rag_benefits | 5/5 | - |
| rag_tech_stack | 5/5 | - |
| rag_location | 5/5 | - |
| rag_interview_process | 5/5 | - |
| rag_remote | 5/5 | - |
| rag_learning_budget | 5/5 | - |
| rag_role | 5/5 | - |
| rag_funding | 5/5 | - |
| honest_ceo_salary | 5/5 | - |
| honest_visa | 5/5 | - |
| honest_ceo_name | 5/5 | - |
