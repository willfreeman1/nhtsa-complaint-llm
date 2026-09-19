"""Repo-relative paths used by Stage 0 tools."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
PROD_OUT = OUTPUT / "prod"
CHECKPOINTS = ROOT / "checkpoints"
INCOMING = DATA / "incoming"

DEBERTA_CKPT = CHECKPOINTS / "deberta_rung2_full_62k_plus_rare" / "final"
QWEN_ADAPTER = CHECKPOINTS / "llm_rung3_full_62k_plus_rare"
GOLD_V3 = OUTPUT / "gold_eval_set_v3.json"


def ensure_prod_out() -> Path:
    PROD_OUT.mkdir(parents=True, exist_ok=True)
    return PROD_OUT
