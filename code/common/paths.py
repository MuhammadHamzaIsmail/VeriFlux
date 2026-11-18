"""Central paths and global settings shared by every script."""
from pathlib import Path
import os, random
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPO = DATA / "keepup_repo"
XLSX = REPO / "KeepUp-Dataset" / "Keepup.xlsx"
EVIDENCE_7Z = REPO / "KeepUp-Dataset" / "evidence" / "evidence.7z"
EVIDENCE_DIR = DATA / "evidence"
PROCESSED = DATA / "processed"
CACHE = ROOT / "results" / "cache"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PREDS = ROOT / "results" / "predictions"

for p in (PROCESSED, CACHE, TABLES, FIGURES, PREDS, EVIDENCE_DIR):
    p.mkdir(parents=True, exist_ok=True)

MODELS = DATA / "models"


def model_path(repo_id: str) -> str:
    """Local copy (downloaded by code/00_download_models.sh) or the hub id as fallback."""
    local = MODELS / repo_id.replace("/", "__")
    return str(local) if (local / "config.json").exists() else repo_id


XLMR = model_path("FacebookAI/xlm-roberta-base")
BGE_M3 = model_path("BAAI/bge-m3")
NORGLM = model_path("NorGLM/Entailment")
XNLI_MINI = model_path("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
SBERT_MINI = model_path("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANKER = model_path("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

SEED = 42
N_FOLDS = 5
N_REPEATS = 3          # repeated grouped CV -> 15 test folds per model
SEEDS = [42, 43, 44]


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
