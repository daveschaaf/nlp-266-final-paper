"""Qwen 8B backend using the established prepared-remark evaluation prompt."""

from importlib.metadata import version
import json
from time import perf_counter

from .contracts import ALLOWED_LABELS, LabelingInput
from .models import MODEL_IDS

MODEL_ID = MODEL_IDS["qwen"]
PROMPT_VERSION = "prepared-remark-labeling-v1"


class QwenLabeler:
    def __init__(self):
        import mlx.core as mx

        from .dataset import LabelingEvaluationDataLoader
        from .evaluation_run import generate_response
        from .models import LabelingModelLoader
        from .prompt import build_prompt_messages, PROMPT_VERSION as ACTUAL_PROMPT_VERSION

        if PROMPT_VERSION != ACTUAL_PROMPT_VERSION:
            raise ValueError("Qwen backend prompt version needs updating.")
        self.mx = mx
        self.engine_version = "mlx-lm=" + version("mlx-lm") + ";mlx=" + version("mlx")
        self.demonstrations = LabelingEvaluationDataLoader().load().demonstrations
        self.model = LabelingModelLoader(local_files_only=True).load("qwen")
        self.build_messages = build_prompt_messages
        self.generate = generate_response

    def __call__(self, row):
        target = LabelingInput(
            row["remark_chunk_id"], row["ticker"], row["remark_text"],
            row["prior_text"], row["similarity"],
        )
        self.mx.reset_peak_memory()
        started = perf_counter()
        raw = self.generate(self.model, self.build_messages(self.demonstrations, target), 128)
        seconds = perf_counter() - started
        response = json.loads(raw)
        label = response.get("label") if isinstance(response, dict) else None
        if not isinstance(label, str) or label not in ALLOWED_LABELS:
            raise ValueError("Qwen returned an invalid label.")
        return label, None, seconds, self.mx.get_peak_memory() / 1e9
