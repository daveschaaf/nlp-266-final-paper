"""Prepared-remark labeling and label-model evaluation.

This package owns the workflow from similarity-based audit preparation through
human review, LLM comparison, and corpus-scale weak labeling. It does not own
transcript parsing or synthetic-question generation.
"""

from .contracts import ALLOWED_LABELS, PreparedRemarkLabel

__all__ = ["ALLOWED_LABELS", "PreparedRemarkLabel"]
