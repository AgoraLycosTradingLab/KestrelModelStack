"""
DISCLAIMER:
This code is provided for educational and research purposes only.
It does not constitute financial advice, investment advice,
trading advice, or a recommendation to buy or sell any security.

Agora Lycos Trading Lab makes no guarantees regarding accuracy,
performance, or profitability. Use at your own risk.
Past performance is not indicative of future results.
"""


# run_kestrel.py
from __future__ import annotations

import sys
from pathlib import Path

# Force imports to resolve from this script's folder first (prevents config.py shadowing)
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install: pip install yfinance")

import config
print("CONFIG LOADED FROM:", config.__file__)
print("CONFIG L4_MAX_WEIGHT:", getattr(config, "L4_MAX_WEIGHT", None))

import layer1_macro_gate as l1
from layer2_factor_engine import factor_engine
from layer3_signal_aggregator import signal_aggregator
from layer4_risk_sizing import risk_and_sizing
from layer5_execution_report import build_orders_and_report

# Prove which Layer 4 is loaded
import layer4_risk_sizing as l4
print("LAYER4 LOADED FROM:", l4.__file__)


# -----------------------------
# Utilities
# -----------------------------
def normalize_ticker_for_yf(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")

def pick_ticker_column(df: pd.DataFrame) -> str:
    for c in ["Symbol", "symbol", "Ticker", "ticker", "SYMBOL"]:
        if c in df.columns:
            return c
    return df.columns[0]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download_universe_prices(
    tickers: list[str],
    period: str,
    batch_size: int = 75,
    pause_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Batched yfinance downloader to reduce rate-limit failures.
    Returns:
      close: DataFrame (dates x tickers)
      vol:   DataFrame (dates x tickers) or None if unavailable
    """
    all_close: list[pd.DataFrame] = []
    all_vol: list[pd.DataFrame] = []

    # de-dupe & keep stable order
    tickers = list(dict.fromkeys(tickers))

    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        batch_total = (total + batch_size - 1) // batch_size

        try:
            data = yf.download(
                tickers=batch,
                period=period,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as e:
            print(f"[WARN] Batch {batch_num}/{batch_total} failed ({len(batch)} tickers): {e}")
            time.sleep(pause_seconds)
            continue

        # Parse close/volume from returned structure
        if isinstance(data.columns, pd.MultiIndex):
            close_b = data.xs("Close", axis=1, level=1)
            vol_b = None
            if "Volume" in data.columns.get_level_values(1):
                vol_b = data.xs("Volume", axis=1, level=1)
        else:
            # single-ticker edge case
            close_b = data[["Close"]].rename(columns={"Close": batch[0]})
            vol_b = data[["Volume"]].rename(columns={"Volume": batch[0]}) if "Volume" in data.columns else None

        close_b = close_b.dropna(how="all").ffill()
        if not close_b.empty:
            all_close.append(close_b)

        if vol_b is not None:
            vol_b = vol_b.reindex(close_b.index).ffill()
            if not vol_b.empty:
                all_vol.append(vol_b)

        # polite pause to reduce throttling
        time.sleep(pause_seconds)

    if not all_close:
        return pd.DataFrame(), None

    close = pd.concat(all_close, axis=1)
    close = close.loc[:, ~close.columns.duplicated()].sort_index().ffill()

    vol = None
    if all_vol:
        vol = pd.concat(all_vol, axis=1)
        vol = vol.loc[:, ~vol.columns.duplicated()].reindex(close.index).ffill()

    # history requirement (same logic you had)
    keep = close.columns[close.count() >= config.MIN_HISTORY_DAYS]
    close = close[keep]
    if vol is not None:
        vol = vol[keep]

    return close, vol

def liquidity_filter(close: pd.DataFrame, vol: pd.DataFrame | None, min_dollar_vol: float) -> pd.DataFrame:
    if vol is None:
        return close
    dollar_vol = (close * vol).rolling(20).mean().iloc[-1]
    liquid = dollar_vol[dollar_vol >= min_dollar_vol].index
    return close[liquid]


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # ---------- Layer 1 ----------
    layer1_tickers = list(l1.TICKERS.keys())
    l1_closes = l1.fetch_closes(layer1_tickers, period=config.PRICE_PERIOD)
    res1 = l1.compute_macro_gate(l1_closes)

    asof = res1.asof
    regime = res1.regime
    confidence = float(res1.confidence)

    # ---------- Artifacts folder ----------
    artifact_day = Path(config.ARTIFACT_DIR) / asof
    ensure_dir(artifact_day)

    # Save Layer 1 artifact
    layer1_payload = {
        "asof": asof,
        "regime": regime,
        "confidence": confidence,
        "risk_on_score": float(res1.risk_on_score),
        "raw": res1.raw.to_dict(),
        "components_01": res1.components_01.to_dict(),
    }
    (artifact_day / "layer1.json").write_text(json.dumps(layer1_payload, indent=2))

    # ---------- Universe ----------
    sp = pd.read_csv(config.SP500_CSV)
    na = pd.read_csv(config.NASDAQ_CSV)
    sp_col = pick_ticker_column(sp)
    na_col = pick_ticker_column(na)

    tickers = pd.Series(
        pd.concat([sp[sp_col], na[na_col]], ignore_index=True).dropna().unique()
    ).astype(str)

    tickers = sorted(set(normalize_ticker_for_yf(t) for t in tickers))

    # ---------- Prices (batched) ----------
    close, vol = download_universe_prices(
        tickers,
        period=config.PRICE_PERIOD,
        batch_size=getattr(config, "YF_BATCH_SIZE", 75),
        pause_seconds=getattr(config, "YF_PAUSE_SECONDS", 1.0),
    )
    if close.shape[1] == 0:
        raise SystemExit("No universe tickers have sufficient history after cleaning.")

    close = liquidity_filter(close, vol, config.MIN_DOLLAR_VOL)
    if close.shape[1] == 0:
        raise SystemExit("Universe empty after liquidity filter.")

    # ---------- Layer 2 ----------
    scores_dict, composite = factor_engine(close, regime, confidence, config.WEIGHTS)

    # Expand scalar factors into Series so output table is consistent
    idx = close.columns
    for k in ["Quality", "Value"]:
        if not isinstance(scores_dict[k], pd.Series):
            scores_dict[k] = pd.Series(scores_dict[k], index=idx)

    out = pd.DataFrame(scores_dict)
    out["Composite"] = composite
    out.index.name = "Ticker"
    out = out.sort_values("Composite", ascending=False)

    out.reset_index().to_csv(artifact_day / "layer2.csv", index=False)

    # ---------- Layer 3 ----------
    sig_res = signal_aggregator(
        layer2_table=out,
        close=close,
        regime=regime,
        confidence=confidence,
        top_n=getattr(config, "L3_TOP_N", 30),
        min_price=getattr(config, "L3_MIN_PRICE", 5.0),
        max_names=getattr(config, "L3_MAX_NAMES", 15),
        min_trend_score=getattr(config, "L3_MIN_TREND_SCORE", 0.45),
    )

    sig_res.signals.reset_index().to_csv(artifact_day / "layer3_signals.csv", index=False)
    (artifact_day / "layer3_diagnostics.json").write_text(json.dumps(sig_res.diagnostics, indent=2))

    # ---------- Layer 4 ----------
    port_val = float(getattr(config, "L4_PORTFOLIO_VALUE", 100_000.0))

    l4_res = risk_and_sizing(
        signals=sig_res.signals,  # indexed by ticker
        close=close,
        ohlc=None,  # upgrade later: pass real OHLC dict for true ATR
        regime=regime,
        confidence=confidence,
        portfolio_value=port_val,
        gross_targets=getattr(config, "L4_GROSS_TARGETS", None),
        max_weight=float(getattr(config, "L4_MAX_WEIGHT", 0.08)),
        min_weight=float(getattr(config, "L4_MIN_WEIGHT", 0.01)),
        atr_mult_by_regime=getattr(config, "L4_ATR_MULT", None),
    )

    # Hard assertion to catch cap violations
    mw = float(getattr(config, "L4_MAX_WEIGHT", 0.08))
    if l4_res.portfolio is not None and not l4_res.portfolio.empty:
        max_seen = float(l4_res.portfolio["Weight"].max())
        print("L4 max_weight:", mw, " | max_seen:", max_seen)
        assert max_seen <= mw + 1e-9, f"Layer4 cap violated: max_seen={max_seen} > max_weight={mw}"

    l4_res.portfolio.reset_index().to_csv(artifact_day / "layer4_portfolio.csv", index=False)
    (artifact_day / "layer4_diagnostics.json").write_text(json.dumps(l4_res.diagnostics, indent=2))

    # ---------- Layer 5 ----------
    l5_order_type = getattr(config, "L5_ORDER_TYPE", "MKT")
    l5_min_shares = int(getattr(config, "L5_MIN_SHARES", 1))

    l5_res = build_orders_and_report(
        portfolio=l4_res.portfolio,         # indexed by ticker (Layer 4 output)
        regime=regime,
        confidence=confidence,
        portfolio_value=port_val,
        artifact_dir=config.ARTIFACT_DIR,
        asof=asof,
        order_type=l5_order_type,
        min_shares=l5_min_shares,
    )

    (artifact_day / "layer5_diagnostics.json").write_text(json.dumps(l5_res.diagnostics, indent=2))

    # ---------- Console report ----------
    print("\n=== AGORA LYCOS — KESTREL PIPELINE (Option A Runner) ===")
    print(f"As of:       {asof}")
    print(f"Regime:      {regime}")
    print(f"Confidence:  {confidence:.3f}")
    print(f"Universe:    {close.shape[1]} tickers (post filters)")

    print(f"\n--- Top {config.TOP_N} (Layer 2 Composite) ---")
    print(out.head(config.TOP_N).round(4).to_string())

    print(f"\n--- Layer 3 Signals ({sig_res.signals.shape[0]}) ---")
    print(sig_res.signals.round(4).to_string())

    if l4_res.portfolio is not None and not l4_res.portfolio.empty:
        print(f"\n--- Layer 4 Portfolio Plan ({(l4_res.portfolio['Weight'] > 0).sum()} names) ---")
        print(l4_res.portfolio.round(4).to_string())
    else:
        print("\n--- Layer 4 Portfolio Plan ---\n(empty)\n")

    print(f"\n--- Layer 5 Orders ({l5_res.orders.shape[0]}) ---")
    print(l5_res.orders.to_string())

    print(f"\nSaved artifacts to: {artifact_day}\n")


if __name__ == "__main__":
    main()
