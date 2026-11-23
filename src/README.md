# Runners for the paper

Run from the checkout root with the project dependencies installed and `src` on Python's import path. For example, `PYTHONPATH=src python -m dearborn.experiments.finrank_quantitative_baseline --help`.

| Result | Module under `dearborn` |
|---|---|
| FinRank full benchmark and supplied-metadata BM25 | `upstream.run_baselines` |
| Original project lexical baseline | `initial_baseline` |
| Quantitative retrieval baseline | `experiments.finrank_quantitative_baseline` |
| Metadata enrichment and filtering | `experiments.finrank_metadata_enrichment`, `experiments.finrank_hyst_filter`, `experiments.finrank_hyst_enriched` |
| Selector and blinded repetition audit | `extract.train_run`, `extract.selector_quality`, `experiments.boilerplate_similarity` |
| Remark screening and usefulness evaluation | `labeling.corpus_run`, `labeling.usefulness_comparison` |
| Passage matching and synthetic generation | `experiments.remark_filing_trial`, `training.finrank_pilot`, `experiments.finrank_summary_roundtrip`, `experiments.remark_generation_scale` |
| Direct and synthetic training | `training.finrank_remark_training`, `training.finrank_metadata_training`, `training.finrank_hard_negatives` |
| Gold-data control | `training.finrank_gold_control` |

The original lexical entry point comes from Dearborn commit `bc4cb54`. FinRank's upstream benchmark runner and dataset attribution are included. The paper tables read the saved results under `data/`.

For corpus-dependent reruns, decompress the SQLite snapshots to their original filenames under `data/`. Neural runs also require the relevant model weights and platform runtimes. The original generation seed and 14 completed batch traces are included. From the checkout root, run `PYTHONPATH=src python -m dearborn.training.finrank_pilot plan` to rebuild the acceptance summary without model inference. Run `python src/export_generation_cases.py --output /tmp/generation-cases.csv` to regenerate the paper CSV. To start or resume unfinished generation, supply `--snapshot` and `--jev-package` to the pilot runner and use the decompressed databases. Some other historical artifacts retain paths to external models or inputs. Use a fresh output directory for reruns to protect the saved results.
