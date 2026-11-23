"""Native Apple-Silicon model loading for prepared-remark labeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from huggingface_hub import snapshot_download


LabelingModelName: TypeAlias = Literal["qwen", "qwen14b", "qwen14b6bit", "nemotron", "nemotron6bit"]

MODEL_IDS: dict[LabelingModelName, str] = {
    "qwen": "mlx-community/Qwen3-8B-4bit",
    "qwen14b": "mlx-community/Qwen3-14B-4bit",
    "qwen14b6bit": "mlx-community/Qwen3-14B-6bit",
    "nemotron": "mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits",
    "nemotron6bit": "mlx-community/NVIDIA-Nemotron-Nano-9B-v2-6bit",
}


@dataclass(frozen=True)
class LoadedLabelingModel:
    """An MLX model and its matching tokenizer."""

    name: LabelingModelName
    model_id: str
    model: Any
    input_processor: Any
    backend: Literal["mlx"] = "mlx"


class LabelingModelLoader:
    """Load the selected pre-quantized model through MLX/Metal."""

    def __init__(
        self,
        local_files_only: bool = False,
    ) -> None:
        self.local_files_only = local_files_only

    def load(self, name: LabelingModelName) -> LoadedLabelingModel:
        """Load one candidate without introducing model-specific code elsewhere."""

        if name not in MODEL_IDS:
            raise ValueError(
                f"Unknown labeling model {name!r}; expected one of {tuple(MODEL_IDS)}."
            )

        # Import lazily so prompt inspection does not initialize Metal.
        from mlx_lm import load as mlx_load

        model_id = MODEL_IDS[name]
        model_path = model_id
        if self.local_files_only:
            from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

            # Match MLX's own download scope so repo docs aren't required offline.
            model_path = snapshot_download(
                model_id, local_files_only=True, allow_patterns=DEFAULT_ALLOW_PATTERNS
            )
        model, tokenizer = mlx_load(model_path)

        return LoadedLabelingModel(
            name=name,
            model_id=model_id,
            model=model,
            input_processor=tokenizer,
        )
