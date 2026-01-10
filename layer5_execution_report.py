"""
DISCLAIMER:
This code is provided for educational and research purposes only.
It does not constitute financial advice, investment advice,
trading advice, or a recommendation to buy or sell any security.

Agora Lycos Trading Lab makes no guarantees regarding accuracy,
performance, or profitability. Use at your own risk.
Past performance is not indicative of future results.
"""


# layer5_execution_report.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class ExecutionResult:
    orders: pd.DataFrame
    report_text: str
    diagnostics: dict


def build_orders_and_report(
    portfolio: pd.DataFrame,
    regime: str,
    confidence: float,
    portfolio_value: float,
    artifact_dir: str | Path,
    asof: str,
    order_type: str = "MKT",          # MKT (next open) or LOC (market-on-close) etc.
    min_shares: int = 1,
) -> ExecutionResult:
    """
    Layer 5 (v0): No broker integration.
    Produces:
      - orders table (Ticker, Side, Qty, OrderType, Notes)
      - daily report string
      - saves artifacts into artifacts/YYYY-MM-DD/
    """
    artifact_day = Path(artifact_dir) / asof
    artifact_day.mkdir(parents=True, exist_ok=True)

    if portfolio is None or portfolio.empty:
        orders = pd.DataFrame(columns=["Ticker", "Side", "Qty", "OrderType", "Notes"])
        report = (
            "AGORA LYCOS — KESTREL DAILY REPORT\n"
            f"As of: {asof}\n"
            f"Regime: {regime}\n"
            f"Confidence: {confidence:.3f}\n\n"
            "No positions generated (empty portfolio).\n"
        )
        diagnostics = {"reason": "empty_portfolio"}
        # Save empties
        orders.to_csv(artifact_day / "layer5_orders.csv", index=False)
        (artifact_day / "daily_report.txt").write_text(report)
        return ExecutionResult(orders=orders, report_text=report, diagnostics=diagnostics)

    # Expect portfolio indexed by ticker
    p = portfolio.copy()
    if p.index.name != "Ticker":
        # if it is not indexed by ticker, try to detect and set
        if "Ticker" in p.columns:
            p = p.set_index("Ticker")

    # Build orders (long-only v0)
    orders = pd.DataFrame(index=p.index)
    orders.index.name = "Ticker"
    orders["Side"] = p.get("Side", "LONG")
    orders["Qty"] = p.get("Shares", 0).fillna(0).astype(int)
    orders["OrderType"] = order_type

    # Filter out tiny orders
    orders = orders[orders["Qty"] >= int(min_shares)].copy()

    # Notes for execution/logging
    def _note_row(t: str) -> str:
        row = p.loc[t]
        entry = row.get("EntryRefPrice", float("nan"))
        stop = row.get("StopPrice", float("nan"))
        w = row.get("Weight", float("nan"))
        return f"w={w:.3f}, entry_ref={entry:.2f}, stop={stop:.2f}"

    orders["Notes"] = [ _note_row(t) for t in orders.index ]

    # Report
    gross = float(p.get("Weight").sum()) if "Weight" in p.columns else float("nan")
    n_names = int((p.get("Weight", 0) > 0).sum()) if "Weight" in p.columns else int(p.shape[0])

    top = p.sort_values("Weight", ascending=False).head(10).copy()

    report_lines = []
    report_lines.append("AGORA LYCOS — KESTREL DAILY REPORT")
    report_lines.append(f"As of: {asof}")
    report_lines.append(f"Regime: {regime}")
    report_lines.append(f"Confidence: {confidence:.3f}")
    report_lines.append(f"Portfolio value: ${portfolio_value:,.0f}")
    report_lines.append(f"Gross target (realized): {gross:.3f}")
    report_lines.append(f"Names: {n_names}")
    report_lines.append("")
    report_lines.append("Top positions (by weight):")
    for t, r in top.iterrows():
        report_lines.append(
            f"  {t:>6}  w={r['Weight']:.3f}  shares={int(r['Shares'])}  "
            f"entry_ref={r['EntryRefPrice']:.2f}  stop={r['StopPrice']:.2f}"
        )

    report_lines.append("")
    report_lines.append("Orders:")
    if orders.empty:
        report_lines.append("  (none)")
    else:
        for t, r in orders.iterrows():
            report_lines.append(f"  {t:>6}  {r['Side']}  qty={int(r['Qty'])}  {r['OrderType']}  {r['Notes']}")

    report = "\n".join(report_lines) + "\n"

    diagnostics = {
        "orders_count": int(orders.shape[0]),
        "names_count": int(n_names),
        "gross": float(gross),
        "order_type": order_type,
        "min_shares": int(min_shares),
    }

    # Save artifacts
    orders.reset_index().to_csv(artifact_day / "layer5_orders.csv", index=False)
    (artifact_day / "daily_report.txt").write_text(report)

    return ExecutionResult(orders=orders, report_text=report, diagnostics=diagnostics)
