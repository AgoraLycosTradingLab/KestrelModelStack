"""
DISCLAIMER:
This code is provided for educational and research purposes only.
It does not constitute financial advice, investment advice,
trading advice, or a recommendation to buy or sell any security.

Agora Lycos Trading Lab makes no guarantees regarding accuracy,
performance, or profitability. Use at your own risk.
Past performance is not indicative of future results.
"""



# layer2_factor_engine.py
import numpy as np
import pandas as pd

def zrank(s: pd.Series) -> pd.Series:
    return (s.rank(pct=True) - 0.5) * 2  # approx z in [-1,1]

def squash(x):
    return 1 / (1 + np.exp(-x))

def momentum_score(px):
    r63  = px.pct_change(63)
    r126 = px.pct_change(126)
    dma200 = px / px.rolling(200).mean() - 1
    z = zrank(r63) + zrank(r126) + zrank(dma200)
    return squash(z)

def trend_score(px):
    hi52 = px / px.rolling(252).max()
    ma20 = px.rolling(20).mean()
    ma50 = px.rolling(50).mean()
    ma200= px.rolling(200).mean()
    ladder = (ma20 > ma50).astype(int) + (ma50 > ma200).astype(int)
    z = zrank(hi52) + ladder
    return squash(z)

def lowvol_score(px):
    vol20 = px.pct_change().rolling(20).std()
    vol60 = px.pct_change().rolling(60).std()
    z = -zrank(vol20) - zrank(vol60)
    return squash(z)

def factor_engine(prices, regime, confidence, weights):
    scores = {}
    scores["Momentum"] = momentum_score(prices).iloc[-1]
    scores["Trend"]    = trend_score(prices).iloc[-1]
    scores["LowVol"]   = lowvol_score(prices).iloc[-1]
    scores["Quality"]  = 0.5
    scores["Value"]    = 0.5

    eff_w = {
        k: v * (0.5 + 0.5 * confidence)
        for k, v in weights[regime].items()
    }

    composite = sum(scores[f] * eff_w[f] for f in eff_w)
    return scores, composite
