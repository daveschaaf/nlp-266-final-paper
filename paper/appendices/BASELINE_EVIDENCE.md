# Baseline appendix: evidence and author notes

This is supporting material and a writing scaffold, not drafted paper prose.

## Sources checked

- `data/FinRank/baselines/results/results.json`: saved numerical results;
  1,185 records, corpus_size 5,230.
- `data/FinRank/baselines/run_baselines.py`: model identifiers and methodology.
- `src/dearborn/finrank.py`: project integration with the baseline data/helpers.

User reports reproducing the baseline in this project. The saved results are in
the FinRank baseline directory; their execution provenance has not independently
been established. Do not label these as independently reproduced measurements
until the corresponding execution output or log is identified.

## Configuration

- Retriever: `sentence-transformers/all-mpnet-base-v2`.
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Dense reranking depth: 20.
- Random seed in baseline source: 42.
- Global retrieval corpus: deduplicated gold and hard-negative passage texts.
- Recall: per-record fraction of gold passages retrieved, macro-averaged.
- Table values are global retrieval results, not in-record reranking metrics.

## Saved results

| Method | Recall@10 (%) | nDCG@10 (×100) | MRR (×100) |
| --- | ---: | ---: | ---: |
| TF-IDF | 32.8242 | 21.9267 | 23.8798 |
| BM25 | 32.0999 | 21.7668 | 23.7843 |
| MPNet | 31.7651 | 20.3749 | 22.2439 |
| MPNet + MS MARCO cross-encoder | 35.1927 | 24.5817 | 26.9191 |

Scope: the baseline designated by the author, MPNet/MS MARCO and lexical
comparisons. Additional retriever experiments are outside this appendix table.

## Checks before submission

- Resolve corpus-count discrepancy: baseline README describes 5,270 entries;
  saved results report 5,230. Use the exact evaluated corpus and explain differences.
- Identify the reproduction run/log and confirm its data revision/configuration.
- Verify FinRank bibliographic entry and the specific boilerplate discussion.
- The retrieval scores alone do not establish boilerplate as a causal explanation.
- Add your own procedure, interpretation and limitations to `baseline.tex`.

## LaTeX integration

After `\appendix`, include `\input{appendices/baseline}`.
Introduction references: `Appendix~\ref{app:finrank-baseline}` and
`Table~\ref{tab:finrank-baseline}`.

The fragment has not been compiled: the main manuscript does not exist yet.
