# Remark–filing matching instructions v2

Purpose: identify specific information requests suggested by prepared remarks and the matched quarterly filing passages that supply their evidence. Produce retrieval supervision, not numerical answers. Use only our own corpus examples. Do not use FinRank questions, answers or gold passages.

## Inputs and separation

The request stage sees the original remark and supplied issuer/call/fiscal-period metadata only. It does not see retrieved passages, human judgments or reference answers. The evidence stage sees that unchanged remark, the proposed request, metadata and candidate passages from the single matched filing. Neither stage may silently change the issuer, period, business scope or transaction.

The caller supplies and verifies the exact company-quarter filing link. Fiscal quarters need not match calendar quarters. Q4 uses the corresponding annual filing under the existing corpus convention. Do not retrieve another period or filing to rescue a match.

## Stage 1: identify a retrievable request

Identify a concrete financial or business fact, comparison, policy, risk mechanism, transaction or explanation communicated by the remark. Name the metric/fact, subject, business scope and period. Spell out needed evidence or calculation inputs. A reported change can be looked up directly; do not require arithmetic if the filing already reports it.

- Filter a sentence whose central purpose is personal opinion or support. Do not turn a speaker’s endorsement into an appointment fact and call it the same task.
- Filter a broad business-model description without a sufficiently specific factual target. Do not invent a more specific question from unrelated filing details.
- Do not automatically reject qualitative content: a specific cash-flow reclassification is a valid target.
- Remove promotional adjectives only when factual meaning is preserved. Keep material qualifiers such as after-tax, adjusted, worldwide, segment, completed, forecast, and open-market.
- Compound remarks may produce a focused request for an explicitly stated factual component. Record which component it covers and which assertions remain outside the request. Do not imply that the entire original remark is verified.
- Resolve “this quarter” using supplied metadata. Do not assume a calendar-quarter end, infer missing transaction identity, or borrow surrounding transcript discussion.

Return a compact record: eligibility (eligible, personal_opinion, too_general, unclear), information_request, required_evidence (list), and short reason. For ineligible remarks leave information_request and required_evidence empty. The request should seek information, not embed the answer values from the remark.

## Stage 2: establish evidence coverage

Evaluate the request against the supplied filing candidates. For each relevant passage record which required facts/inputs it supplies and whether support is direct, calculational, explanatory or period/context only. Use supplied passage identifiers; never invent identifiers. Provide a brief evidence summary for each claimed support. Paraphrasing is allowed. Do not copy exact quotations or produce character offsets. Keep supplied passage IDs so the summary can be checked against the full passage.

Accept only if one passage or an explicit set of passages supplies all material requirements of the request. A passage supporting one required component is relevant partial evidence, not an automatic negative. Multiple passages may be complementary, redundant or alternative sufficient sources. Do not require every candidate to support the request.

Check these distinctions:

1. Same issuer and business scope; identify a subsidiary relationship only when established by supplied evidence/metadata.
2. Same measurement period and duration: fiscal versus calendar year, quarter versus year-to-date, current versus prior year. A document-year label is not a measurement period.
3. Same metric and definition: adjusted versus GAAP, total versus long-term debt, worldwide versus regional sales, after-tax versus pre-tax, payments versus declarations, cash outflow versus the transaction amount in a note, revenue share versus revenue growth.
4. Comparable units and currencies. Standard magnitude/sign conventions and rounding to the remark’s stated precision are permitted; a genuinely different measure is not rounding.
5. Explanations must be stated by the evidence. Topic overlap and plausible causal reasoning are insufficient.
6. Table headers, row labels, periods and units must be available. A neighboring header passage may be included explicitly; do not guess missing context.
7. For calculations, identify the operation and sourced inputs. Do not demand an already calculated answer, and do not invent inputs or definitions. This matching step does not perform numerical answer generation.

Do not rewrite the request after reading candidates merely to make them pass. A request change requires a separately recorded proposal/review. In particular, replacing a business-model claim with “which utilities does the company own?” changes the target.

## Decisions and output

Return: decision (accept, review, reject), reason, request_coverage (complete, partial, none), and evidence (supplied passage_id, supplies, summary). Also return original_remark_coverage (complete, partial, not_established) and a short list of uncovered original assertions. Eligibility and unsupported evidence are separate failure types.

- accept: the specific request has complete evidence coverage. Approval of a narrower request does not approve uncovered parts of the remark.
- review: weak/ambiguous support or uncertain metric/period/identity. Do not export as a positive pair.
- reject: insufficient candidate evidence, personal opinion, overly general statement, or unresolved meaning. Do not export a positive pair.

No support is not a contradiction. Candidate failure does not prove the full/raw filing lacks evidence. Keep these reasons separate: remark unsuitable; candidate retrieval miss; parsed-filing evidence not found; coverage uncertain. Declare a contradiction only when evidence gives an incompatible value/assertion for the same fact.

Do not require a supported request to preserve every incidental phrase of a compound remark. Do require complete support for every material part of the chosen request. When exporting retrieval labels, keep the request–passage labels separate from original remark–passage labels.

## Reviewed boundaries

- Agilent: cash-flow statement supports both operating cash flow and capex; MD&A supports only cash flow. “Payments to acquire property, plant and equipment” supplies capex.
- Applied Materials: a request for Semiconductor Systems sales and reported growth is supported. It does not verify the complete remark’s 20% foundry/logic growth and China explanation. Market share is not growth rate.
- Labcorp: a different laboratory acquisition does not support the named North Mississippi acquisition. Insufficient parsed evidence; no positive pair.
- AES tariffs: sourcing measures support mitigation, not the remark’s stronger zero-exposure/7-GW assertion. Human judgment: weak support; hold, do not accept.
- Global Payments: a specific cash-flow move from operating to financing activities is directly supported qualitative content. General discussion of customer balances is not evidence of the change.
- PACCAR: after-tax litigation charge and adjusted earnings are directly reported; accept permissible rounding. Do not substitute the pre-tax charge.
- Viatris: impairment total and explanation for discount-rate increases are supported. Non-cash wording was not separately established in the reviewed excerpts.
- Cboe: speaker’s support is personal opinion. Mention of a CEO role elsewhere is not support for that sentiment.
- Apple: dividend payments and repurchased-share transaction values have complementary evidence; retain header context. Cash-flow, equity and note amounts must not be treated as interchangeable. Open-market-only wording is not established by the excerpts.
- AES business model: too general for this exercise; do not salvage by inventing a narrower utilities question.

## v2 change

Exact-string quotation matching is removed. Nonempty evidence summaries and valid supplied passage IDs are required. Semantic support must be reviewed against the full passage; schema validity is not proof of factual support.
