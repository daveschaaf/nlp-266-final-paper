# optimization, checkpoints, metric logging
from .dataset import AlignmentDataset
from .model import ExtractModel
from .contracts import ExtractTrainerLoss, SelectorOutput, ExtractTrainingResult, EmbeddingRecordLoader, EvaluatedCompanyQuarter, ExtractTrainerHistory, AlignmentBatch, ExtractModelLoss
from .embedding_store import ExtractEmbeddingStore, QDRANT_COLLECTION
from .alignment import ANSWER_ID
from torch.utils.data import DataLoader
import torch
import numpy as np
import logging
from pathlib import Path
import random
import os
import psutil

process = psutil.Process(os.getpid())

def memory_snapshot_gb(logger: logging.Logger | None) -> dict[str, float]:
    snapshot = {
        "rss_gb": process.memory_info().rss / 1024**3,
    }
    if logger is not None:
        logger.info(f"RSS = %.2f GB", snapshot["rss_gb"])
    if torch.backends.mps.is_available():
        snapshot["mps_allocated_gb"] = (
            torch.mps.current_allocated_memory() / 1024**3
        )
        snapshot["mps_driver_gb"] = (
            torch.mps.driver_allocated_memory() / 1024**3
        )
        if logger:
            logger.info(
                f"MPS Allocated = {snapshot['mps_allocated_gb']:.2f} GB, Driver = {snapshot['mps_driver_gb']:.2f} GB"
            )
    return snapshot

class ExtractTrainer:
    def __init__(
        self,
        model: ExtractModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.model = model
        self.optimizer=optimizer
        self.device=device
        self.logger=logger

    def train_epoch(self, data_loader: DataLoader):
        self.model.to(self.device)
        self.model.train()
        total_loss = 0.0
        qa_loss = 0.0
        filing_loss = 0.0
        kl_loss = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(data_loader, start=1):
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            model_output = self.model(batch)
            losses = model_output.model_loss

            self.assert_losses_are_finite(batch_index, batch, losses)

            losses.total_loss.backward()

            if not self.gradients_are_finite():
                self.logger.warning("Optimizer step skipped: non-finite gradient.")
                continue
            assert self.model.selector.container.grad is not None
            assert self.model.selector.importance_scorer.layers[0].weight.grad is not None
            assert self.model.alignment.remark_encoder.layers[0].weight.grad is not None
            assert self.model.alignment.filing_encoder.layers[0].weight.grad is not None
            if self.model.use_qa_alignment:
                assert self.model.alignment.qa_encoder.layers[0].weight.grad is not None

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.assert_parameters_are_finite()

            if str(batch_index)[-1] == 1:
                memory_snapshot_gb(self.logger)
            
            loss = losses.total_loss.item()
            total_loss += loss
            qa_loss += losses.qa_loss.item()
            filing_loss += losses.filing_loss.item()
            kl_loss += losses.kl_loss.item()
            batch_count += 1
            self.logger.info("Batch %d Loss: %.2f", batch_index, loss)

        if batch_count == 0:
            raise ValueError("Training loader yielded no batches.")
        return ExtractTrainerLoss(
            total_loss = total_loss / batch_count,
            qa_loss=qa_loss / batch_count,
            filing_loss=filing_loss / batch_count,
            kl_loss = kl_loss / batch_count,
        )


    def evaluate(
        self,
        data_loader: DataLoader,
        embedding_store: ExtractEmbeddingStore | None = None,
        inspect: bool = False,
    ) -> ExtractTrainingResult:
        self.model.to(self.device)
        self.model.eval()
        total_loss = 0.0
        qa_loss = 0.0
        filing_loss = 0.0
        kl_loss = 0.0
        batch_count = 0
        evaluation_losses = []
        selector_outputs = []
        company_quarters = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                model_output = self.model(batch)
                model_loss = model_output.model_loss
                loss = model_loss.total_loss.item()
                total_loss += loss
                qa_loss += model_loss.qa_loss.item()
                filing_loss += model_loss.filing_loss.item()
                kl_loss += model_loss.kl_loss.item()
                batch_count += 1
                self.logger.info(
                    f"Batch {batch_count} Loss: {loss:.2f}"
                )
                loss = ExtractTrainerLoss(
                    total_loss = total_loss / batch_count,
                    qa_loss=qa_loss / batch_count,
                    filing_loss=filing_loss / batch_count,
                    kl_loss=kl_loss / batch_count
                )
                evaluation_losses.append(loss)
                selector_outputs.append(model_output.selector_output)

                for index, metadata in enumerate(batch.metadata):
                    if inspect:
                        if embedding_store is None:
                            raise ValueError(f"Embedding store required for inspection (inspect=True).")
                        remark_weights = (
                            model_output.selector_output.remark_selection_weights[index]
                            .detach()
                            .cpu()
                        )
                        selected_indices = (
                            model_output.selector_output.remark_selected_mask[index]
                            .detach()
                            .cpu()
                            .nonzero(as_tuple=False)
                            .flatten()
                            .tolist()
                        )
                        selected_indices.sort(
                            key=lambda position: float(remark_weights[position]),
                            reverse=True,
                        )
                        selected_ids = [
                            metadata.remark_ids[position]
                            for position in selected_indices
                        ]
                        selected_weights = [
                            float(remark_weights[position])
                            for position in selected_indices
                        ]
                        embeddings = [
                            EmbeddingRecordLoader(
                                unit_id=record.input.unit_id,
                                text=record.input.text,
                                vector=record.vector
                            )
                            for record in embedding_store.fetch(selected_ids)
                        ]
                        for record in embeddings:
                            self.logger.info("%s\n%s", record.unit_id, record.text)
                    else:
                        embeddings = []
                        selected_ids = []
                        selected_weights = []
                    company_quarters.append(EvaluatedCompanyQuarter(
                        company_quarter_id=metadata.company_quarter_id,
                        call_id=metadata.call_id,
                        accession_number=metadata.accession_number,
                        metadata=metadata,
                        selected_ids=tuple(selected_ids),
                        selected_weights=tuple(selected_weights),
                        embeddings=tuple(embeddings)
                    ))

        return ExtractTrainingResult(
            losses = tuple(evaluation_losses),
            selector_outputs=tuple(selector_outputs),
            company_quarters=tuple(company_quarters),
        )

    def fit(self,
            train_loader:DataLoader,
            val_loader: DataLoader,
            epochs:int,
            checkpoint_dir: Path,
            embedding_store: ExtractEmbeddingStore | None = None,
            start_epoch: int = 1,
            history: ExtractTrainerHistory | None = None,
            best_loss: float = float('inf')
            ) -> ExtractTrainerHistory:
        self.logger.info("Starting training...")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_path = checkpoint_dir / "last.pt"
        best_path = checkpoint_dir / "best.pt"
        history = history or ExtractTrainerHistory(train=list(),val=list())
        qa_losses = []
        filing_losses = []
        total_losses = []
        for epoch in range(start_epoch, epochs+1):
            self.logger.info(f"Epoch {epoch:2}:")
            train_loss = self.train_epoch(train_loader)
            validation = self.evaluate(val_loader, embedding_store=embedding_store)
            history.train.append(train_loss)
            if not validation.losses:
                raise ValueError(f"Validation loss missing for epoch {epoch}.")
            history.val.append(validation.losses[-1])
            validation_loss = validation.losses[-1].total_loss
            if validation_loss < best_loss: 
                best_loss = validation_loss
                self.save_checkpoint(best_path, epoch, history, best_loss=best_loss)
            self.save_checkpoint(last_path, epoch, history, best_loss=best_loss)
            total_losses.append(train_loss.total_loss)
            qa_losses.append(train_loss.qa_loss)
            filing_losses.append(train_loss.filing_loss)
            self.logger.info(
                "total=%.4f qa=%.4f filing=%.4f",
                train_loss.total_loss,
                train_loss.qa_loss,
                train_loss.filing_loss,
            )
        self.logger.info(f"Average Loss: {np.mean(total_losses):.4f}")
        self.logger.info(f"Average QA Loss: {np.mean(qa_losses):.4f}")
        self.logger.info(f"Average Filing Loss: {np.mean(filing_losses):.4f}")
        self.logger.info(f"Training complete!")
        return history

    def assert_losses_are_finite(self, batch_index: int, batch: AlignmentBatch, losses: ExtractModelLoss) -> bool:
        if not torch.isfinite(losses.total_loss).all().item():
            answer_mask = batch.qa_mask & batch.qa_role_ids.eq(ANSWER_ID)

            self.logger.error("Non-finite loss at batch=%d", batch_index)
            self.logger.error(
                "Company quarters: %s",
                [metadata.company_quarter_id for metadata in batch.metadata],
            )
            self.logger.error(
                "Losses: total=%s qa=%s filing=%s",
                losses.total_loss.detach().cpu().item(),
                losses.qa_loss.detach().cpu().item(),
                losses.filing_loss.detach().cpu().item(),
            )
            self.logger.error(
                "Mask counts: remarks=%s answers=%s filings=%s",
                batch.remark_mask.sum(dim=1).detach().cpu().tolist(),
                answer_mask.sum(dim=1).detach().cpu().tolist(),
                batch.filing_mask.sum(dim=1).detach().cpu().tolist(),
            )
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}: "
                f"{losses.total_loss.detach().cpu().item()}"
            )
        else:
            return True
    def assert_parameters_are_finite(self) -> bool:
        non_finite = [
            name
            for name, parameter in self.model.named_parameters()
            if not torch.isfinite(parameter).all().item()
        ]
        if non_finite:
            self.logger.error(
                "Non-finite parameter values: %s",
                ", ".join(non_finite),
            )
            raise FloatingPointError(
                f"Non-finite parameter values: {', '.join(non_finite)}"
            )
        return True

    def gradients_are_finite(self) -> bool:
        result = {}

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            gradient = parameter.grad

            if gradient is None:
                result[f"{name}.has_gradient"] = False
                continue

            result[f"{name}.has_gradient"] = True
            result[f"{name}.is_finite"] = bool(torch.isfinite(gradient).all())
            result[f"{name}.norm"] = float(gradient.norm().item())

        if not all(
            value
            for name, value in result.items()
            if name.endswith(".is_finite")
        ) and len(result.items()) > 0:
            return False

        return True

    def save_checkpoint(self, path: Path, epoch: int, history: ExtractTrainerHistory, best_loss: float):
        self.logger.info("Saving checkpoint %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": history,
            "best_validation_loss": best_loss,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        tmp_path = path.with_suffix('.tmp')
        torch.save(payload, tmp_path)
        tmp_path.replace(path)

    def load_checkpoint(self, path: Path):
        if not path.exists():
            raise ValueError(f"No checkpoint at '{path}'.")
        self.logger.info("Loading checkpoint %s", path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        
        self.model.to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])

        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        next_epoch = payload['epoch'] + 1
        history = payload['history']
        best_loss = payload['best_validation_loss']

        self.logger.info(
            "Resumed checkpoint=%s next_epoch=%d train_epochs=%d "
            "validation_epochs=%d best_validation_loss=%.4f",
            path,
            next_epoch,
            len(history.train),
            len(history.val),
            best_loss,
        )
        return (next_epoch, history, best_loss)


def create_data_loader(dataset: AlignmentDataset, batch_size, shuffle=False, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
