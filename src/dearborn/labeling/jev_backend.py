"""Cached JEV 27B backend for the compact four-choice labeling prompt."""

from importlib.metadata import version
import math
from pathlib import Path
from time import perf_counter

MODEL_ID = "netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit"
PROMPT_VERSION = "jev-prepared-remark-compact-v1"
LABEL_DESCRIPTIONS = {
    "boilerplate": "Generic reusable legal or disclosure language. It states no useful business fact.",
    "administrative": "Introductions, transitions, instructions, or other call logistics without a business fact.",
    "repeated_substantive": "A meaningful business fact in familiar wording or continuing a prior claim, including updated figures.",
    "novel_substantive": "A meaningful business fact materially different from the prior-call match.",
}
INSTRUCTION = (
    "Classify only the current earnings-call statement. First decide if it conveys a useful "
    "business fact, then compare it with the previous call. Similar wording alone is not boilerplate."
)


class JevLabeler:
    def __init__(self, snapshot: Path, *, general_context=False):
        import jev27b_mlx

        self.engine_version = version("jev27b-mlx")
        self.model = jev27b_mlx.load(str(snapshot))
        descriptions = dict(LABEL_DESCRIPTIONS)
        instruction = INSTRUCTION
        if general_context:
            descriptions["general_business_context"] = (
                "Recurring explanations, broad commentary, generic optimism, or vague outlook "
                "without a specific result, event, change, or interpretable business claim."
            )
            instruction += (
                " Be selective. Substantive requires a specific, interpretable business claim. "
                "Business vocabulary alone is insufficient. Updated numbers and concrete qualitative "
                "developments are substantive; generic explanations, aspirations and context-dependent "
                "fragments belong to general_business_context. Speaker introductions, announcements "
                "of what will be discussed and call mechanics are administrative."
            )
        self.labels = frozenset(descriptions)
        self.question = self.model.question(
            "choice", instruction, list(descriptions), list(descriptions.values())
        )

    def __call__(self, row):
        state = (
            f"Classify this statement.\n\nCURRENT STATEMENT:\n{row['remark_text']}"
            f"\n\nPREVIOUS-CALL MATCH:\n{row['prior_text']}"
            f"\n\nSAME-COMPANY SIMILARITY:\n{row['similarity']:.4f}"
        )
        start = perf_counter()
        result = self.model.decide(state, self.question)
        seconds = perf_counter() - start
        scores = dict(zip(result["options"], result["probabilities"], strict=True))
        if (
            scores.keys() != self.labels
            or result["choice"] not in scores
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values())
            or not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5)
            or scores[result["choice"]] != max(scores.values())
        ):
            raise ValueError("JEV returned invalid four-choice scores.")
        return result["choice"], scores, seconds, self.model.last_peak_gb
