# layer3_signal_aggregator.py
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class SignalResult:
    signals: pd.DataFrame
    diagnostics: dict


def signal_aggregator(
    layer2_table: pd.DataFrame,
    close: pd.DataFrame,
    regime: str,
    confidence: float,
    top_n: int = 30,
    min_price: float = 5.0,
    max_names: int = 15,
    min_trend_score: float = 0.45,
) -> SignalResult:
    """
    Layer 3 (v0): Turn Layer 2 ranks into actionable candidates.
    Price-only filters; no fundamentals; no earnings calendar.
    """

    # Expect columns from Layer 2: Momentum, Trend, LowVol, Quality, Value, Composite
    df = layer2_table.copy()

    # Ensure we’re only using tickers we have price data for
    df = df[df.index.isin(close.columns)]

    # Restrict to Top-N by composite (cheap + deterministic)
    df = df.head(top_n)

    last_px = close.iloc[-1]

    # --- Filters (v0) ---
    # 1) Minimum price filter
    df["Last"] = last_px.reindex(df.index)
    df = df[df["Last"] >= float(min_price)]

    # 2) Trend sanity: avoid structurally broken charts
    # (in Transition/Risk-Off you really want trend >= threshold)
    if regime in ("Transition", "Risk-Off"):
        df = df[df["Trend"] >= float(min_trend_score)]

    # 3) Simple "skip day" logic: if macro confidence is very low, reduce max_names
    if confidence < 0.10:
        max_names = max(5, min(max_names, 8))

    # --- Signal scoring (separate from composite) ---
    # In Transition: favor LowVol + Quality(placeholder) + Trend
    # In Risk-On: favor Momentum + Trend
    # In Risk-Off: favor LowVol + Trend + Value(placeholder)
    if regime == "Risk-On":
        df["SignalScore"] = 0.55 * df["Momentum"] + 0.35 * df["Trend"] + 0.10 * df["LowVol"]
    elif regime == "Risk-Off":
        df["SignalScore"] = 0.55 * df["LowVol"] + 0.30 * df["Trend"] + 0.15 * df["Value"]
    else:  # Transition
        df["SignalScore"] = 0.45 * df["LowVol"] + 0.35 * df["Trend"] + 0.20 * df["Quality"]

    df = df.sort_values("SignalScore", ascending=False).head(int(max_names))

    # Side logic (v0): long-only list. Shorts can be added later.
    df["Side"] = "LONG"
    df["EntryType"] = "NEXT_OPEN"  # or "CLOSE" depending on your execution choice

    # Notes / explainability (handy for logs)
    df["Notes"] = (
        "Regime=" + regime
        + ", conf=" + df["SignalScore"].map(lambda x: f"{x:.2f}")
        + ", trend=" + df["Trend"].map(lambda x: f"{x:.2f}")
        + ", lowvol=" + df["LowVol"].map(lambda x: f"{x:.2f}")
    )

    signals = df[["Side", "SignalScore", "EntryType", "Notes", "Last"]].copy()
    diagnostics = {
        "top_n_input": int(top_n),
        "min_price": float(min_price),
        "max_names": int(max_names),
        "min_trend_score": float(min_trend_score),
        "post_filter_count": int(signals.shape[0]),
    }
    return SignalResult(signals=signals, diagnostics=diagnostics)
