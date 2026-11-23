# JEV versus Qwen remark-selection experiments

Recorded November 11, 2026. The selected production candidate is local
4-bit Qwen3-8B using the generation-usefulness prompt.

## Purpose and hardware

Compare local models for selecting earnings-call remarks that can guide
financial-filing question generation. The experiments evolved from content
classification into binary generation usefulness after human review exposed a
gap between substantive information and useful question material.

Hardware was an Apple M4 Pro with 24 GB unified memory. Models ran separately.

- Qwen checkpoint was `mlx-community/Qwen3-8B-4bit`, using MLX LM.
- JEV checkpoint was `netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit`,
  revision `39bc135bf354aac7a12a8a8328d2d6f7d6039a11`, using the publisher's
  dedicated bidirectional-attention runner.
- Peak memory below is MLX allocation, in decimal GB unless labeled GiB.
  It excludes other applications and operating-system memory.
- Inference totals exclude model loading. These are observed configurations
  with differing prompt sizes and runtime libraries.

## 1. Initial 17-statement comparison

The existing EW 2025 Q1 holdout contained 12 substantive and 5 unwanted
statements. Qwen used 12 fixed demonstrations. JEV's first attempt with a
long few-shot prompt stalled before its first prediction and was stopped.
The subsequent JEV test used compact definitions without demonstrations.

| Metric | JEV compact | Qwen few-shot |
| --- | ---: | ---: |
| Correct keep/drop | 17/17 | 17/17 |
| Correct four-class labels | 12/17 | 12/17 |
| Macro-F1 | 0.790 | 0.778 |
| Inference total | 68.89 s | 113.57 s |
| Median per statement | 3.77 s | 6.62 s |
| Model loading | 26.90 s | 5.99 s |
| Peak MLX memory | 16.78 GB | 5.21 GiB |

JEV was faster in the compact configuration. The different prompting prevents
attributing the latency difference solely to model architecture.

## 2. Resumable 100-item JEV pilot and 96-item Qwen replay

JEV labeled 96 of 100 queued statements before the user paused it. Qwen then
labeled those same 96 frozen inputs using its established few-shot prompt.

| Measure | JEV | Qwen |
| --- | ---: | ---: |
| Completed | 96 | 96 |
| Kept as substantive | 77 | 63 |
| Dropped | 19 | 33 |
| Inference total | 753.86 s | 649.25 s |
| Peak MLX memory | 16.80 GB | 5.59 GB |

JEV initially ran near 3–4 seconds per statement, then took 41–110 seconds
on several short inputs. System monitoring showed active swap traffic and
about 9.6 GiB of swap in use. Input length did not explain the outliers.
This supports memory pressure as a likely contributor, although per-call
swap activity was not recorded.

The models disagreed on 18 keep/drop decisions. The user relabeled those
with a fifth category, `general_business_context`. The results were
12 general-context, 3 substantive, and 3 administrative statements.

Keep/drop agreement on this disagreement-only sample was Qwen 13/18 and
JEV 5/18. The user subsequently described the three substantive items as
borderline and favored stricter filtering. Preserve those comments separately
from the saved labels.

Source review also found Valero Q&A segments in the prepared-remark sample.
That is an upstream parser issue. It must be distinguished from classifier
errors.

## 3. Five-label development replay

Adding general business context and a stricter instruction to compact JEV
produced 14/18 exact matches on the audited disagreements, in 73.33 seconds
at a median 3.93 seconds. JEV dropped all 18. This included the three
borderline substantive items, so the development result did not establish
recall on clear useful facts.

## 4. Fresh 100-item five-label comparison

Both models received the same definitions and one human-labeled demonstration
per category. The new sample excluded prior labeling calls and contained
20 remarks each from AME, BIIB, BMY, NDAQ, and PHM.

| Measure | JEV | Qwen |
| --- | ---: | ---: |
| Completed / failed | 100 / 0 | 100 / 0 |
| Substantive | 57 | 49 |
| General business context | 31 | 41 |
| Administrative or boilerplate | 12 | 10 |
| Inference total | 896.99 s | 253.01 s |
| Median per remark | 8.89 s | 2.52 s |
| Peak MLX memory | 16.97 GB | 5.40 GB |

There were 76 exact-label agreements and 90 keep/drop agreements.
The user labeled the ten keep/drop disagreements as seven novel substantive
and three general-context remarks. JEV agreed on six keep/drop decisions
and Qwen on four. These remarks were nevertheless considered weak generation
candidates, motivating a direct usefulness judgment.

## 5. Human usefulness audit

The user judged 97 unique remarks previously hand-labeled substantive.

- 56 were useful for generation.
- 41 were not useful.
- Useful items included 43 novel and 13 repeated substantive remarks.
- Standalone forward-looking EPS guidance was explicitly designated not useful,
  despite precise numbers.
- Specific risks suitable for risk-factor or MD&A questions could remain useful.

This changed the operational target to whether a remark can guide a specific,
worthwhile question supportable by financial statements, MD&A, business
disclosures, or risk factors.

## 6. Binary usefulness comparison

The shared v2 prompt contained ten examples. Eight came from the 97-item
usefulness audit and were excluded from the replay. Two boilerplate/admin
examples came from the earlier content audit. The remaining 89 cases contained
52 Yes and 37 No reference labels.

| Measure | JEV, partial | Qwen, complete |
| --- | ---: | ---: |
| Completed | 46/89 | 89/89 |
| Correct | 28/46 | 64/89 |
| Useful retained | 14/25 | 32/52 |
| Not-useful retained | 7/21 | 5/37 |
| Missed useful | 11 | 20 |
| Retained-candidate precision | 66.7% | 86.5% |
| Useful recall | 56.0% | 61.5% |
| Inference total | 861.05 s | 325.26 s |
| Peak MLX memory | 17.10 GB | 5.61 GB |

The denominators differ because JEV was paused. On the 45 cases both models
had completed at the earlier comparison, Qwen got 30 correct and JEV 28.
Qwen retained two non-useful remarks versus JEV's seven, while missing
13 useful remarks versus JEV's ten.

Qwen encountered one incomplete JSON response. The output limit was raised
from 64 to 256 tokens before retrying the failed row and finishing the
queue. Completed rows were preserved. This configuration change should
accompany reporting of aggregate runtime.

The user reviewed Qwen's five retained false positives and considered them
reasonable candidates to send to filing verification. Their weakness was
likely evidence availability or testability, rather than absence of business
content. Keep the original human labels for reproducible metrics.

## Decision and limits

Select Qwen3-8B for the full-corpus usefulness runner. Its measured runtime,
memory footprint, and retained-candidate precision support the next operational
pilot. Subsequent filing matching must establish actual evidence support.

These are development experiments on modest samples, including previously
inspected cases. Disagreement audits are selected samples. Their agreement
rates cannot be treated as full-corpus accuracy. The binary results also
reflect evolving usefulness policy. No retriever-training or FinRank benchmark
improvement has been demonstrated by these labeling experiments.

## Artifacts

- [Shared usefulness prompt](../labeling/prompts/GENERATION_USEFULNESS_PROMPT.md)
- [Human usefulness audit](../../data/experiments/labeling_evaluation/human-substantive-usefulness/)
- [Original disagreements](../../data/experiments/labeling_evaluation/jev-qwen-disagreement-18/)
- [Five-label disagreements](../../data/experiments/labeling_evaluation/five-label-disagreement-10/)
- [Labeling database](../../data/remark_labels.sqlite)
- [Five-label database](../../data/remark_labels_five.sqlite)
- [Usefulness database](../../data/remark_usefulness_eval.sqlite)
- [Overall working plan](../../TRANSCRIPT_GUIDED_FINRANK_SYNTHESIS_STRATEGY.md)

SQLite files are local artifacts and were not included in the preceding code
commit. Early benchmark JSONL files remain under `/private/tmp`, including
`dearborn-jev-ew-qwen8b.jsonl`, `dearborn-jev-ew-jev27b-compact.jsonl`,
and `dearborn-jev-five-label-18.jsonl`.
