# Retrieval sweep

Chosen on the tuning set only (best hit rate, then smallest context). The held-out column shows whether that choice generalises.

| Chunk size | Overlap | Top-k | Chunks | Tuning hits | Held-out hits | Avg context (chars) |
|---|---|---|---|---|---|---|
| 1000 | 200 | 1 | 4 | 12/15 | 9/10 | 744 |
| 1000 | 200 | 2 | 4 | 14/15 | 10/10 | 1488 |
| 500 | 100 | 1 | 6 | 12/15 | 9/10 | 450 |
| 500 | 100 | 2 | 6 | 13/15 | 10/10 | 895 |
| 500 ⭐ | 100 | 3 | 6 | 14/15 | 10/10 | 1347 |
| 300 | 60 | 1 | 10 | 13/15 | 9/10 | 253 |
| 300 | 60 | 2 | 10 | 13/15 | 9/10 | 501 |
| 300 | 60 | 3 | 10 | 13/15 | 9/10 | 746 |
| 200 | 40 | 2 | 16 | 13/15 | 9/10 | 331 |
| 200 | 40 | 3 | 16 | 13/15 | 10/10 | 496 |

**Chosen:** chunk size 500, overlap 100, top-k 3. To use it, add to `.env.local`:

```
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=100
RAG_TOP_K=3
```
