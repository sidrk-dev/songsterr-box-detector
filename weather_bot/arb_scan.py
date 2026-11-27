"""Scan historical quotes for within-event arbitrage.

Hourly index ladders ("above X.99" rungs): buying YES on a lower strike at ask a_lo and NO on a
higher strike at (1 - b_hi) pays at least $1 in every outcome, so it is risk-free when
a_lo + (1 - b_hi) + fees < 1, i.e. b_hi - a_lo > fees.
Daily-high buckets are mutually exclusive and exhaustive: buying YES on every bucket pays
exactly $1 (risk-free if sum of asks + fees < 1); buying NO on every bucket pays n-1.
"""
import json, pathlib, numpy as np, pandas as pd
from backtest import fee
DATA = pathlib.Path(__file__).parent / "data"

def quotes_at_minutes(candles):
    """Forward-filled bid/ask per market on a common minute grid."""
    frames = {}
    for tk, cs in candles.items():
        if not cs: continue
        s = pd.DataFrame([(c[0], c[1], c[2]) for c in cs], columns=["t", "bid", "ask"]).set_index("t")
        frames[tk] = s[~s.index.duplicated(keep="last")]
    if not frames: return None
    grid = sorted(set().union(*[f.index for f in frames.values()]))
    return {tk: f.reindex(grid).ffill() for tk, f in frames.items()}, grid

def hourly_ladders():
    hits = []
    for f in (DATA / "candles").glob("*/*.json"):
        c = json.loads(f.read_text()); q = quotes_at_minutes(c)
        if q is None: continue
        fr, grid = q
        tks = sorted(fr, key=lambda t: float(t.split("-T")[-1]))
        for i in range(len(tks)):
            for j in range(i + 1, len(tks)):
                lo, hi = fr[tks[i]], fr[tks[j]]
                gap = hi.bid - lo.ask
                for t in gap[gap > 0].index:
                    a, b = lo.ask[t], hi.bid[t]
                    cost = a + (1 - b) + fee(10, a) / 10 + fee(10, 1 - b) / 10
                    if cost < 1: hits.append((f.stem, tks[i], tks[j], t, a, b, 1 - cost))
    return pd.DataFrame(hits, columns=["event", "lo", "hi", "t", "ask_lo", "bid_hi", "profit"])

def daily_buckets(period=60):
    hits = []
    for f in (DATA / f"dcandles{period}").glob("*/*.json"):
        c = json.loads(f.read_text()); q = quotes_at_minutes(c)
        if q is None: continue
        fr, grid = q
        A = pd.DataFrame({k: v.ask for k, v in fr.items()}); B = pd.DataFrame({k: v.bid for k, v in fr.items()})
        n = A.shape[1]
        for t in A.index:
            a, b = A.loc[t], B.loc[t]
            if a.isna().any() or b.isna().any(): continue
            cy = a.sum() + sum(fee(10, x) / 10 for x in a)
            cn = (1 - b).sum() + sum(fee(10, 1 - x) / 10 for x in b)
            if cy < 1: hits.append((f.stem, t, "all_yes", 1 - cy))
            if cn < n - 1: hits.append((f.stem, t, "all_no", (n - 1) - cn))
    return pd.DataFrame(hits, columns=["event", "t", "kind", "profit"])

if __name__ == "__main__":
    d = daily_buckets(); print("daily bucket arbs:", len(d), "events", d.event.nunique() if len(d) else 0)
    if len(d): print(d.sort_values("profit", ascending=False).head(10))
    h = hourly_ladders(); print("hourly ladder arbs:", len(h), "events", h.event.nunique() if len(h) else 0)
    if len(h): print(h.sort_values("profit", ascending=False).head(10)); h.to_parquet(DATA / "arb_hourly.parquet")
