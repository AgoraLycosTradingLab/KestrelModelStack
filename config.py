# config.py
from pathlib import Path

# =====================================================
# Base paths
# =====================================================
BASE_DIR = Path(__file__).parent

ARTIFACT_DIR = BASE_DIR / "artifacts"

SP500_CSV = BASE_DIR / "sp500_master.csv"
NASDAQ_CSV = BASE_DIR / "nasdaq_master.csv"

# =====================================================
# Data / Universe
# =====================================================
PRICE_PERIOD = "2y"

MIN_HISTORY_DAYS = 260
MIN_DOLLAR_VOL = 20_000_000

TOP_N = 30

# =====================================================
# Yahoo Finance download controls (IMPORTANT)
# =====================================================
# These directly control rate-limit behavior
YF_BATCH_SIZE = 75        # lower = safer, higher = faster
YF_PAUSE_SECONDS = 1.0    # increase if you still see throttling

# =====================================================
# Layer 2 — Factor Engine Weights
# =====================================================
WEIGHTS = {
    "Risk-On": {
        "Momentum": 0.35,
        "Trend": 0.25,
        "Quality": 0.20,
        "LowVol": 0.10,
        "Value": 0.10,
    },
    "Transition": {
        "Quality": 0.30,
        "LowVol": 0.25,
        "Momentum": 0.20,
        "Value": 0.15,
        "Trend": 0.10,
    },
    "Risk-Off": {
        "LowVol": 0.35,
        "Quality": 0.30,
        "Value": 0.20,
        "Momentum": 0.10,
        "Trend": 0.05,
    },
}

# =====================================================
# Layer 3 — Signal Aggregator
# =====================================================
L3_TOP_N = 30
L3_MIN_PRICE = 5.0
L3_MAX_NAMES = 15
L3_MIN_TREND_SCORE = 0.45

# =====================================================
# Layer 4 — Risk & Sizing
# =====================================================
L4_PORTFOLIO_VALUE = 100_000.0

L4_GROSS_TARGETS = {
    "Risk-On": 0.90,
    "Transition": 0.50,
    "Risk-Off": 0.20,
}

L4_ATR_MULT = {
    "Risk-On": 2.5,
    "Transition": 2.0,
    "Risk-Off": 1.6,
}

L4_MAX_WEIGHT = 0.08
L4_MIN_WEIGHT = 0.01

# =====================================================
# Layer 5 — Execution / Reporting
# =====================================================
# Broker-agnostic for now
L5_ORDER_TYPE = "MKT"     # informational (next-open market style)
L5_MIN_SHARES = 1
