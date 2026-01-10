"""
DISCLAIMER:
This code is provided for educational and research purposes only.
It does not constitute financial advice, investment advice,
trading advice, or a recommendation to buy or sell any security.

Agora Lycos Trading Lab makes no guarantees regarding accuracy,
performance, or profitability. Use at your own risk.
Past performance is not indicative of future results.
"""


# layer4_risk_sizing.py
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass
class PortfolioPlanResult:
    portfolio: pd.DataFrame
    diagnostics: dict


# -----------------------------
# Helpers
# -----------------------------
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr14_from_ohlc(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ohlc: columns MultiIndex (ticker -> OHLC) or dict-like is handled upstream.
    Here we expect a DataFrame with columns: ["High","Low","Close"] for ONE ticker.
    """
    tr = _true_range(ohlc["High"], ohlc["Low"], ohlc["Close"])
    return tr.rolling(period).mean()

def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    r = close.pct_change()
    return r.rolling(window).std()

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def cap_and_redistribute(w: pd.Series, gross: float, max_weight: float) -> pd.Series:
    """
    Water-fill cap enforcement.
    Guarantees: w_i <= max_weight for all i.
    Attempts: sum(w) ~= gross (if feasible under caps).
    """
    w = w.astype(float).copy()

    if w.sum() <= 0:
        return w

    # First scale to desired gross
    w = w / w.sum() * float(gross)

    for _ in range(100):
        over = w > max_weight
        if not over.any():
            break

        excess = float((w[over] - max_weight).sum())
        w[over] = float(max_weight)

        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0 or excess <= 0:
            break

        # redistribute excess proportionally among under-cap names
        w[under] = w[under] + excess * (w[under] / under_sum)

    # Final hard guarantee (no renormalization after this)
    w = w.clip(upper=float(max_weight))

    return w


# -----------------------------
# Main Layer 4
# -----------------------------
def risk_and_sizing(
    signals: pd.DataFrame,
    close: pd.DataFrame,
    ohlc: dict[str, pd.DataFrame] | None,
    regime: str,
    confidence: float,
    portfolio_value: float = 100_000.0,
    gross_targets: dict[str, float] | None = None,
    max_weight: float = 0.08,
    min_weight: float = 0.01,
    atr_mult_by_regime: dict[str, float] | None = None,
) -> PortfolioPlanResult:
    """
    Inputs:
      signals: DataFrame indexed by ticker with at least columns:
               ["Side","SignalScore","EntryType","Last"] (as created by Layer 3)
      close:   DataFrame of close prices (index=dates, columns=tickers)
      ohlc:    optional dict[ticker] -> DataFrame with columns ["Open","High","Low","Close"]
              If None, ATR will be approximated using close-only volatility.
      regime/confidence: from Layer 1
    Output:
      portfolio plan with weights + stop levels + optional shares
    """

    if gross_targets is None:
        gross_targets = {"Risk-On": 0.90, "Transition": 0.50, "Risk-Off": 0.20}

    if atr_mult_by_regime is None:
        atr_mult_by_regime = {"Risk-On": 2.5, "Transition": 2.0, "Risk-Off": 1.6}

    if signals.empty:
        return PortfolioPlanResult(
            portfolio=pd.DataFrame(),
            diagnostics={"reason": "no_signals", "regime": regime, "confidence": float(confidence)},
        )

    # Align tickers
    tickers = [t for t in signals.index if t in close.columns]
    sig = signals.loc[tickers].copy()

    # Entry reference price = last close (EOD system)
    last = close.iloc[-1].reindex(sig.index)
    sig["EntryRefPrice"] = last
    sig["Side"] = sig.get("Side", "LONG")

    # Compute 20d vol for sizing
    vol20 = close[sig.index].pct_change().rolling(20).std().iloc[-1]
    sig["Vol20"] = vol20

    # ATR (preferred) if OHLC provided; else approximate ATR using close-vol
    atr_mult = _safe_float(atr_mult_by_regime.get(regime, 2.0), 2.0)
    atr14_vals = pd.Series(index=sig.index, dtype=float)

    if ohlc is not None:
        for t in sig.index:
            if t not in ohlc:
                atr14_vals.loc[t] = np.nan
                continue
            df = ohlc[t][["High", "Low", "Close"]].dropna()
            if df.shape[0] < 20:
                atr14_vals.loc[t] = np.nan
                continue
            atr14_vals.loc[t] = atr14_from_ohlc(df, period=14).iloc[-1]
    else:
        # Approx ATR: ATR ≈ close * vol20 * sqrt(14) (rough, but usable v0)
        atr14_vals = (sig["EntryRefPrice"] * sig["Vol20"] * math.sqrt(14)).astype(float)

    sig["ATR14"] = atr14_vals

    # Stop price (long-only): entry - k*ATR
    sig["StopPrice"] = sig["EntryRefPrice"] - atr_mult * sig["ATR14"]

    # Risk per share (for sanity checks)
    sig["RiskPerShare"] = sig["EntryRefPrice"] - sig["StopPrice"]

    # Gross exposure target with confidence throttle
    # low confidence => scale gross down
    gross_base = _safe_float(gross_targets.get(regime, 0.50), 0.50)
    gross = gross_base * (0.5 + 0.5 * float(confidence))  # [0.5..1.0] scaling
    gross = max(0.0, min(1.5, gross))

    # Vol-inverse weights: w_i ∝ 1/vol20
    inv_vol = 1.0 / sig["Vol20"].replace(0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if inv_vol.sum() <= 0:
        # fallback: equal weights
        raw_w = pd.Series(1.0, index=sig.index)
    else:
        raw_w = inv_vol / inv_vol.sum()

    w = raw_w * gross
    w = w.where(w >= float(min_weight), 0.0)
    w = cap_and_redistribute(w, gross=gross, max_weight=float(max_weight))
    
    # Absolute cap guarantee
    w = w.clip(upper=float(max_weight))

    sig["Weight"] = w

    # Optional shares (round down)
    sig["DollarAlloc"] = sig["Weight"] * float(portfolio_value)
    sig["Shares"] = (sig["DollarAlloc"] / sig["EntryRefPrice"]).fillna(0).astype(int)

    # Final ordering
    sig = sig.sort_values("Weight", ascending=False)

    diagnostics = {
        "regime": regime,
        "confidence": float(confidence),
        "gross_base": float(gross_base),
        "gross_final": float(gross),
        "names_in": int(signals.shape[0]),
        "names_out": int((sig["Weight"] > 0).sum()),
        "portfolio_value": float(portfolio_value),
        "atr_mult": float(atr_mult),
        "max_weight": float(max_weight),
        "min_weight": float(min_weight),
    }

    portfolio = sig[
        [
            "Side",
            "SignalScore",
            "EntryType",
            "EntryRefPrice",
            "Weight",
            "DollarAlloc",
            "Shares",
            "ATR14",
            "StopPrice",
            "Vol20",
            "RiskPerShare",
            "Notes",
        ]
    ].copy()

    return PortfolioPlanResult(portfolio=portfolio, diagnostics=diagnostics)
