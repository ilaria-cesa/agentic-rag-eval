# LLM smoke test on 7 example questions (4-bit NF4, greedy, same retrieved context)

Run before the golden set existed, so it is only a rough check of the loaders and the answer format,
not a benchmark. The golden-set comparison is in notebook `02_baseline_rag.ipynb`.

| model | right answer-or-refuse | invalid citations | answers without citation* | thinking leaked | mean s/answer | tokens/s | peak VRAM (GB) |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | 0.86 | 0 | 1 | 0 | 13.9 | 6.3 | 9.4 |
| Qwen/Qwen3.5-9B | 1.00 | 0 | 2 | 0 | 13.4 | 6.0 | 9.6 |

\* this column also counted correct `NOT_FOUND` refusals; the metric was fixed afterwards.
One of the seven examples (a question about the next iPhone release date) turned out not to be
unanswerable, because the filing lists launches by quarter; it was replaced.
