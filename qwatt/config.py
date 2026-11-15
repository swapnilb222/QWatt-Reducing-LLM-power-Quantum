"""Central settings for Q-Watt. Every module reads settings from here.

Secrets come from .env (gitignored). Never print them.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
DB_PATH = Path(os.getenv("QWATT_DB", str(ROOT / "results" / "qwatt.sqlite")))

CALIBRATION_JSON = RESULTS_DIR / "calibration.json"
ANGLES_JSON = RESULTS_DIR / "angles.json"
HYPERPARAMS_JSON = RESULTS_DIR / "hyperparams.json"
SPLITS_JSON = RESULTS_DIR / "splits.json"
SELECTION_EVAL_JSON = RESULTS_DIR / "selection_eval.json"
SOLVER_QUALITY_JSON = RESULTS_DIR / "solver_quality.json"
DEMO_QUESTIONS_JSON = RESULTS_DIR / "demo_questions.json"

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- secrets / endpoints (read lazily; do not log) ----
QR_TOKEN = os.getenv("QR_TOKEN", "")
QR_ACCOUNT = os.getenv("QR_ACCOUNT", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
TIGER_DATABASE_URL = os.getenv("TIGER_DATABASE_URL", "")
# Set QWATT_FAKE_LLM=1 to develop without LM Studio. Rows are tagged llm_backend='fake'.
FAKE_LLM = os.getenv("QWATT_FAKE_LLM", "0") == "1"


@dataclass
class Settings:
    seed: int = 7
    # pool / QUBO
    n_pool: int = 16
    red_threshold: float = 0.6       # theta
    lam: float = 1.0                 # redundancy weight
    nu: float = 0.5                  # bridge reward weight
    kappa: float = 1.0               # relevance per joule exchange rate
    budget_tokens: int = 120         # B, context token budget for selection arms
    # QAOA
    p: int = 1
    shots: int = 1024
    # generation
    max_tokens: int = 48
    temperature: float = 0.0
    answer_words: int = 12
    # benchmark controls
    cooldown_max_s: float = 15.0
    idle_band_w: float = 1.5         # within this many watts of idle = cooled down
    sampler_hz: float = 4.0
    # chars per token fallback when tokenizer unavailable (calibrated later)
    chars_per_token: float = 4.0
    extra: dict = field(default_factory=dict)

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_settings() -> Settings:
    """Defaults, overridden by results/hyperparams.json (frozen after dev tuning)."""
    s = Settings()
    if HYPERPARAMS_JSON.exists():
        data = json.loads(HYPERPARAMS_JSON.read_text())
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
    # NOTE: calibration.json records the measured chars/token (~4.6 for Gemma 4), but selection keeps the
    # value the QUBO was tuned with (frozen in hyperparams.json / default 4.0). Token estimates in the QUBO are
    # relative costs; the energy ledger always uses the API's real usage token counts.
    return s


def load_calibration() -> dict | None:
    if CALIBRATION_JSON.exists():
        return json.loads(CALIBRATION_JSON.read_text())
    return None


def eps_in_default() -> float:
    """Joules per input token from calibration, or a labeled placeholder."""
    cal = load_calibration()
    if cal and cal.get("eps_in"):
        return float(cal["eps_in"])
    return 0.05  # PLACEHOLDER until scripts/calibrate.py runs
