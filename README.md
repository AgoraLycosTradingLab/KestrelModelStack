"""""
General Disclaimer
This material is provided for informational and educational purposes only and does not constitute investment advice, financial advice, trading advice, or a recommendation to buy or sell any securities or financial instruments.

Agora Lycos Trading Lab is a research-focused entity. All content reflects research opinions, models, and historical analysis, which may be incomplete, incorrect, or change without notice. Past performance is not indicative of future results.

No representation is made regarding the accuracy, completeness, or suitability of the information provided. Use of any information is at your own risk.
""""""


The Kestrel Model Stack is a fully integrated, end-to-end systematic trading pipeline that converts macro conditions into executable portfolio orders through five disciplined layers:

Layer 1 – Macro Regime Gate
Classifies the market into Risk-On, Risk-Off, or Transition using cross-asset signals (equities, volatility, credit, rates, USD, oil). Outputs regime, confidence, and diagnostic scores to control downstream exposure and factor emphasis

Layer 2 – Factor Engine
Scores the liquid equity universe across core factor families (Momentum, Trend, Low Volatility, Quality, Value). Factor weights adapt dynamically to the macro regime and confidence, producing a composite ranking of stocks.

Layer 3 – Signal Aggregator
Transforms factor rankings into actionable candidates by applying price, trend, and regime-aware filters. Generates a concise, explainable list of long signals with regime-specific signal scoring.

Layer 4 – Risk & Position Sizing
Converts signals into a portfolio plan using volatility-aware sizing, regime-scaled gross exposure, ATR-based stops, and strict position caps. Outputs weights, dollar allocations, shares, and risk metrics.

Layer 5 – Execution & Reporting
Produces broker-agnostic order tickets, a daily portfolio report, and a full artifact trail (CSV/JSON) for auditing, backtesting, and simulation.

Together, the Kestrel Model Stack enforces a macro-first, risk-controlled, rules-based workflow designed for research transparency, repeatability, and disciplined execution across market regimes.
