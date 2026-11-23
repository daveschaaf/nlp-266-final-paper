"""Run one candidate model over the locked prepared-remark holdout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from dearborn.constants import DATA_ROOT

from .dataset import LabelingEvaluationDataLoader
from .models import MODEL_IDS, LabelingModelLoader, LabelingModelName
from .prompt import PROMPT_VERSION, build_prompt_messages


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print populated messages without loading model weights.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N locked test examples.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL destination; defaults to a timestamped evaluation file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file instead of refusing to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output and skip completed audit IDs.",
    )
    return parser


def generate_responses(
    loaded_model,
    messages_batch: list[list[dict[str, str]]],
    max_new_tokens: int,
) -> list[str]:
    """Generate one response per conversation in a padded batch."""

    from mlx_lm import batch_generate

    tokenizer = loaded_model.input_processor
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for messages in messages_batch
    ]
    result = batch_generate(
        loaded_model.model,
        tokenizer,
        prompts,
        max_tokens=max_new_tokens,
        prefill_batch_size=len(prompts),
        completion_batch_size=len(prompts),
        verbose=False,
    )
    responses = [response.strip() for response in result.texts]
    for index, response in enumerate(responses):
        if response:
            continue
        raise RuntimeError(f"Model generated an empty response for batch row {index}.")
    return responses


def generate_response(loaded_model, messages, max_new_tokens: int) -> str:
    """Compatibility wrapper for one-conversation inference."""

    return generate_responses(loaded_model, [messages], max_new_tokens)[0]


def run(arguments: argparse.Namespace) -> None:
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if arguments.resume and arguments.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")

    data = LabelingEvaluationDataLoader().load()
    test_examples = data.test_examples[: arguments.limit]

    prompts = [
        build_prompt_messages(data.demonstrations, test_example)
        for test_example in test_examples
    ]

    if arguments.prompt_only:
        print(
            json.dumps(
                {
                    "prompt_version": PROMPT_VERSION,
                    "model": arguments.model,
                    "demonstration_audit_ids": [
                        example.audit_id for example in data.demonstrations
                    ],
                    "target_audit_id": test_examples[0].audit_id,
                    "messages": prompts[0],
                },
                indent=2,
            )
        )
        return

    output_path = arguments.output or _default_output_path(arguments.model)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not (arguments.overwrite or arguments.resume):
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --resume or --overwrite."
        )

    completed_ids = (
        _load_completed_ids(output_path, arguments.model)
        if arguments.resume and output_path.exists()
        else set()
    )
    pending = [
        (example, prompt)
        for example, prompt in zip(test_examples, prompts)
        if example.audit_id not in completed_ids
    ]
    if not pending:
        print(f"No pending predictions; {len(completed_ids)} already saved in {output_path}")
        return

    print(
        f"Saving predictions to {output_path}; completed={len(completed_ids)} "
        f"pending={len(pending)} batch_size={arguments.batch_size}"
    )
    loaded_model = LabelingModelLoader(
        local_files_only=arguments.local_files_only
    ).load(arguments.model)

    mode = "a" if arguments.resume else "w"
    saved_count = 0
    with output_path.open(mode, encoding="utf-8") as output_file:
        for batch_index, start in enumerate(
            range(0, len(pending), arguments.batch_size),
            start=1,
        ):
            batch = pending[start : start + arguments.batch_size]
            started = perf_counter()
            responses = generate_responses(
                loaded_model,
                [prompt for _, prompt in batch],
                max_new_tokens=arguments.max_new_tokens,
            )
            batch_seconds = perf_counter() - started
            for (example, _), response in zip(batch, responses):
                record = {
                    "prompt_version": PROMPT_VERSION,
                    "model_id": loaded_model.model_id,
                    "audit_id": example.audit_id,
                    "sentence_id": example.input.sentence_id,
                    "response": response,
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "batch_inference_seconds": batch_seconds,
                }
                serialized = json.dumps(record)
                output_file.write(serialized + "\n")
                print(serialized)
                saved_count += 1
            output_file.flush()

    print(
        f"Saved {saved_count} new predictions "
        f"({len(completed_ids) + saved_count} total) to {output_path}"
    )


def _load_completed_ids(path: Path, model_name: LabelingModelName) -> set[int]:
    """Validate an append-only checkpoint and return completed audit IDs."""

    completed_ids = set()
    expected_model_id = MODEL_IDS[model_name]
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if record.get("prompt_version") != PROMPT_VERSION:
                raise ValueError(
                    f"Prompt-version mismatch at {path}:{line_number}."
                )
            if record.get("model_id") != expected_model_id:
                raise ValueError(f"Model mismatch at {path}:{line_number}.")
            audit_id = int(record["audit_id"])
            if audit_id in completed_ids:
                raise ValueError(f"Duplicate audit_id {audit_id} in {path}.")
            completed_ids.add(audit_id)
    return completed_ids


def _default_output_path(model_name: LabelingModelName) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        DATA_ROOT
        / "experiments"
        / "labeling_evaluation"
        / f"{model_name}_{timestamp}.jsonl"
    )


def main() -> None:
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
