import os
from pathlib import Path

QDRANT_STORAGE = 'qdrant_storage'
QDRANT_COLLECTION_MP = "finrank_mpnet"

def _find_project_root() -> Path:
    for parent in [Path(__file__)] + list(Path(__file__).parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("pyproject.toml not found")

PROJECT_ROOT = _find_project_root()
CONTEXT_DIR = PROJECT_ROOT / "src" / "dearborn" / "context"

# Root of all corpus data. Override to relocate the corpus off the repo
# volume (external disk, EC2 workspace) without touching call sites.
DATA_ROOT = Path(os.environ.get("DEARBORN_DATA_ROOT", PROJECT_ROOT / "data"))
RAW_DIR = DATA_ROOT / "raw"
RAW_TRANSCRIPT_DIR = RAW_DIR / "transcripts"
FINRANK_DIR = DATA_ROOT / "FinRank"
