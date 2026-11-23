Generate up to THREE distinct, standalone financial retrieval questions from the CURRENT INPUT. Return {"questions": [{"question": "...", "answer": "...", "evidence_indices": [0]}]}.

The bounded remark selects the topic; only the supplied filing passages establish facts. Do not use the original unbounded remark or demonstration passages as current evidence. The diagnostic answer is a brief explanation for checking answerability, not a separately graded prose task. evidence_indices are zero-based positions in CURRENT INPUT passages; include every passage needed, no invented IDs or exact quotations.

Each question must name the company, relevant fiscal year/quarter or explicit event date, metric and business scope. Preserve adjusted versus GAAP, actual versus guidance, signs, currencies and period duration. Do not infer causes from numbers alone. Simple arithmetic is allowed when the supplied passages establish every input. A reporting quarter is not automatically the date of an event disclosed after quarter-end.

Vary the question form only when useful: amount, comparison, calculation, transaction purpose or stated explanation. Do not create paraphrase duplicates or insist on three questions when only one useful question exists. Stay within the bounded remark topic. Return an empty questions list if no supported question can be written. Never ask about omitted prior-quarter trends, unrelated transactions, motives or unsupported explanations.

Examples are demonstrations only. Their questions were drafted using reviewed own-corpus evidence; they are not FinRank gold examples. Use their specificity and evidence boundaries, not their company names or values, for the new input.
