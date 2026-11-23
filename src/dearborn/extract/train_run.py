"""Run or evaluate the Extract model.

Examples:
    uv run python -m dearborn.extract.train_run --epochs 20
    uv run python -m dearborn.extract.train_run --resume checkpoints/<run>/last.pt --epochs 30
    uv run python -m dearborn.extract.train_run --evaluate checkpoints/<run>/best.pt
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import random

from dearborn.company_data import sqlite_connection
from dearborn.constants import PROJECT_ROOT
from dearborn.corpus.extract_repository import fetch_valid_extract_scopes
from dearborn.logging_config import setup_logging

from .collate import ExtractCollator
from .dataset import build_extract_datasets
from .embedding_store import ExtractEmbeddingStore, QDRANT_COLLECTION
from .model import ExtractModel
from .selector_quality import selector_quality_report, write_selector_quality_report
from .trainer import ExtractTrainer, create_data_loader, get_device


SPLIT_LENGTHS = (0.8, 0.1, 0.1)
MANIFEST_FILENAME = "run_manifest.json"
REVIEW_FILENAME = "selected_remarks_test.jsonl"
QUALITY_REPORT_FILENAME = "selector_quality_test.json"


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int,
        help="Total epoch target. A resumed run continues through this epoch.",
    )
    parser.add_argument(
        "--resume", type=Path,
        help="Path to last.pt or best.pt. Its parent remains the checkpoint directory.",
    )
    parser.add_argument(
        "--evaluate", type=Path,
        help="Evaluate a checkpoint against its manifest-defined test split and write a review artifact.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help="Directory for a new run. Defaults to checkpoints/<timestamp>.",
    )
    parser.add_argument(
        "--scope-id", action="append", dest="scope_ids",
        help="Company-quarter ID to include. Repeat to bound a run. Defaults to every valid scope.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--compression-rate", type=float, default=0.15)
    parser.add_argument("--top-k-lower-bound", type=int, default=5)
    parser.add_argument(
        "--qa-alignment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Q&A alignment. Use --no-qa-alignment for filing-only training.",
    )
    return parser


def valid_scope_ids() -> tuple[str, ...]:
    connection = sqlite_connection()
    try:
        return tuple(scope.company_quarter_id for scope in fetch_valid_extract_scopes(connection))
    finally:
        connection.close()


def resolve_qa_alignment(
    arguments: argparse.Namespace,
    manifest: dict | None = None,
) -> bool:
    """Resolve the objective flag and reject resume-time objective changes."""
    requested = arguments.qa_alignment
    if manifest is None:
        return True if requested is None else requested

    saved = manifest["model"].get("use_qa_alignment", True)
    if requested is not None and requested != saved:
        raise ValueError(
            "Q&A alignment setting does not match the saved run manifest: "
            f"requested={requested}, saved={saved}."
        )
    return saved


def build_manifest(arguments: argparse.Namespace, scope_ids: tuple[str, ...]) -> dict:
    """Record every input needed to reconstruct a run's exact data split."""
    return {
        "schema_version": 1,
        "scope_ids": list(scope_ids),
        "split": {"seed": arguments.seed, "lengths": list(SPLIT_LENGTHS)},
        "data": {"batch_size": arguments.batch_size},
        "model": {
            "dimension": 768,
            "slots": arguments.slots,
            "temperature": arguments.temperature,
            "tau": arguments.tau,
            "beta": arguments.beta,
            "compression_rate": arguments.compression_rate,
            "top_k_lower_bound": arguments.top_k_lower_bound,
            "use_qa_alignment": resolve_qa_alignment(arguments),
        },
        "optimizer": {"name": "Adam", "learning_rate": arguments.learning_rate},
    }


def write_manifest(checkpoint_dir: Path, manifest: dict) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / MANIFEST_FILENAME
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run manifest: {path}")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(checkpoint_path: Path) -> dict:
    path = checkpoint_path.parent / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint evaluation requires {path}. "
            "This checkpoint predates run manifests and cannot be reproduced exactly."
        )
    return json.loads(path.read_text())


def write_selector_review(result, path: Path) -> None:
    """Write selected remarks ranked by their final per-remark weight."""
    budgets = [
        budget
        for selector_output in result.selector_outputs
        for budget in selector_output.selection_budget.detach().cpu().tolist()
    ]
    if len(budgets) != len(result.company_quarters):
        raise ValueError("Selector outputs do not align with evaluated company quarters.")

    with path.open("w") as review_file:
        for company_quarter, selection_budget in zip(result.company_quarters, budgets):
            texts_by_id = {record.unit_id: record.text for record in company_quarter.embeddings}
            payload = {
                "company_quarter_id": company_quarter.company_quarter_id,
                "call_id": company_quarter.call_id,
                "accession_number": company_quarter.accession_number,
                "selection_budget": selection_budget,
                "selected_remarks": [
                    {
                        "rank": rank,
                        "remark_id": remark_id,
                        "weight": weight,
                        "text": texts_by_id[remark_id],
                    }
                    for rank, (remark_id, weight) in enumerate(
                        zip(company_quarter.selected_ids, company_quarter.selected_weights),
                        start=1,
                    )
                ],
            }
            review_file.write(json.dumps(payload) + "\n")


def evaluate_checkpoint(arguments: argparse.Namespace, logger: logging.Logger) -> None:
    checkpoint_path = arguments.evaluate
    manifest = load_manifest(checkpoint_path)
    split = manifest["split"]
    generator = torch.Generator().manual_seed(split["seed"])
    train, validation, test = build_extract_datasets(
        scope_ids=tuple(manifest["scope_ids"]),
        lengths=tuple(split["lengths"]),
        generator=generator,
    )
    del train, validation
    if not test:
        raise ValueError("Manifest-defined test split is empty.")

    model = ExtractModel(**manifest["model"])
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=manifest["optimizer"]["learning_rate"],
    )
    trainer = ExtractTrainer(model, optimizer, get_device(), logger)
    trainer.load_checkpoint(checkpoint_path)
    test_loader = create_data_loader(
        test,
        manifest["data"]["batch_size"],
        shuffle=False,
        collate_fn=ExtractCollator(),
    )
    embedding_store = ExtractEmbeddingStore(QDRANT_COLLECTION)
    result = trainer.evaluate(
        test_loader,
        embedding_store=embedding_store,
        inspect=True,
    )
    loss = result.losses[-1]
    review_path = checkpoint_path.parent / REVIEW_FILENAME
    quality_report_path = checkpoint_path.parent / QUALITY_REPORT_FILENAME
    write_selector_review(result, review_path)
    write_selector_quality_report(
        selector_quality_report(result, embedding_store), quality_report_path
    )
    logger.info(
        "Test total=%.4f qa=%.4f filing=%.4f kl=%.4f; selector review=%s quality report=%s",
        loss.total_loss,
        loss.qa_loss,
        loss.filing_loss,
        loss.kl_loss,
        review_path,
        quality_report_path,
    )


def run(arguments: argparse.Namespace) -> None:
    if arguments.evaluate is not None and (
        arguments.resume is not None or arguments.checkpoint_dir is not None or arguments.epochs is not None
    ):
        raise ValueError("--evaluate cannot be combined with --resume, --checkpoint-dir, or --epochs.")
    if arguments.evaluate is None and (arguments.epochs is None or arguments.epochs < 1):
        raise ValueError("--epochs must be at least 1.")
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if arguments.resume is not None and arguments.checkpoint_dir is not None:
        raise ValueError("Use either --resume or --checkpoint-dir, not both.")

    resume_manifest = (
        load_manifest(arguments.resume)
        if arguments.resume is not None
        else None
    )
    use_qa_alignment = resolve_qa_alignment(arguments, resume_manifest)

    logger = logging.getLogger("dearborn.extract.train_run")
    setup_logging()
    if arguments.evaluate is not None:
        evaluate_checkpoint(arguments, logger)
        return

    scope_ids = tuple(arguments.scope_ids) if arguments.scope_ids else valid_scope_ids()
    # seed all random generators to make the runs deterministic
    generator = torch.Generator().manual_seed(arguments.seed)
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    train, validation, test = build_extract_datasets(
        scope_ids=scope_ids,
        lengths=SPLIT_LENGTHS,
        generator=generator,
    )
    logger.info(
        "Dataset scopes=%d train=%d validation=%d test=%d seed=%d batch_size=%d",
        len(scope_ids), len(train), len(validation), len(test), arguments.seed, arguments.batch_size,
    )
    if len(train) == 0 or len(validation) == 0:
        raise ValueError("Training requires non-empty train and validation splits.")

    collator = ExtractCollator()
    train_loader = create_data_loader(train, arguments.batch_size, shuffle=True, collate_fn=collator)
    validation_loader = create_data_loader(
        validation, arguments.batch_size, shuffle=False, collate_fn=collator
    )

    model = ExtractModel(
        dimension=768,
        slots=arguments.slots,
        temperature=arguments.temperature,
        tau=arguments.tau,
        beta=arguments.beta,
        compression_rate=arguments.compression_rate,
        top_k_lower_bound=arguments.top_k_lower_bound,
        use_qa_alignment=use_qa_alignment,
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=arguments.learning_rate,
    )
    trainer = ExtractTrainer(model, optimizer, get_device(), logger)
    embedding_store = ExtractEmbeddingStore(QDRANT_COLLECTION)
    logger.info(
        "Objective: filing_alignment=true qa_alignment=%s beta=%s",
        str(use_qa_alignment).lower(),
        arguments.beta,
    )

    if arguments.resume is not None:
        start_epoch, history, best_loss = trainer.load_checkpoint(arguments.resume)
        checkpoint_dir = arguments.resume.parent
    else:
        start_epoch, history, best_loss = 1, None, float("inf")
        checkpoint_dir = arguments.checkpoint_dir or (
            PROJECT_ROOT / "checkpoints" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        manifest_path = write_manifest(checkpoint_dir, build_manifest(arguments, scope_ids))
        logger.info("Starting new run checkpoint_dir=%s", checkpoint_dir)
        logger.info("Saved run manifest %s", manifest_path)

    if start_epoch > arguments.epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {start_epoch - 1}; "
            f"--epochs={arguments.epochs} would run no epochs."
        )

    trainer.fit(
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=arguments.epochs,
        checkpoint_dir=checkpoint_dir,
        embedding_store=embedding_store,
        start_epoch=start_epoch,
        history=history,
        best_loss=best_loss,
    )


def main() -> None:
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
