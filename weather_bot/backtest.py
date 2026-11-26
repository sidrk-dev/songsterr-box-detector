"""Walk-forward backtest of the rung-sniping strategy against historical Kalshi quotes.

Assumptions (deliberately conservative):
  * We are a taker: buy YES at the ask, buy NO at (1 - yes bid).
  * Kalshi taker fee per order: ceil_to_cent(0.07 * C * P * (1 - P)).
  * Quotes are the last 1-minute candle close at or before the decision minute.
  * Depth is unknown historically, so fills are capped at MAX_FILL contracts per rung.
  * The model for week w is trained only on events that settled before week w started.
"""
import math, sys, json, pathlib
import numpy as np
import pandas as pd
from model import LadderModel, brier, informative

DATA = pathlib.Path(__file__).parent / "data"
MAX_FILL = 25


def fee(contracts, price):
    return math.ceil(0.07 * contracts * price * (1 - price) * 100 - 1e-9) / 100


def walk_forward(df, min_weeks=3):
    df = df.copy()
    df["wk"] = (df.S - df.S.min()).dt.days // 7
    parts = []
    for w in range(min_weeks, int(df.wk.max()) + 1):
        te = df[df.wk == w]
        tr = df[df.S < te.S.min() - pd.Timedelta(hours=1)]
        m = LadderModel().fit(tr)
        parts.append(te.assign(p=m.predict(te)))
    return pd.concat(parts)


def signals(r, thr):
    """One candidate trade per rung (YES or NO) where model edge vs executable price >= thr."""
    r = r.dropna(subset=["bid", "ask"]).copy()
    r = r[(r.ask > r.bid)]
    yes_edge = r.p - r.ask
    no_edge = r.bid - r.p
    r["side"] = np.where(yes_edge >= no_edge, "yes", "no")
    r["edge"] = np.maximum(yes_edge, no_edge)
    r["price"] = np.where(r.side == "yes", r.ask, 1 - r.bid)
    r["pwin"] = np.where(r.side == "yes", r.p, 1 - r.p)
    r["win"] = np.where(r.side == "yes", r.y, 1 - r.y)
    r = r[(r.edge >= thr) & (r.price >= 0.03) & (r.price <= 0.97)]
    return r.sort_values(["S", "edge"], ascending=[True, False])


def simulate(sig, bankroll=20.0, kelly=0.25, max_event_frac=0.25, max_fill=MAX_FILL):
    """Compound a bankroll through the signals in time order; positions settle at S."""
    cash, open_pos, log = bankroll, [], []
    for S, g in sig.groupby("S", sort=True):
        # settle everything that closed before this decision time
        D = g.D.iloc[0]
        still = []
        for pos in open_pos:
            if pos["S"] <= D:
                cash += pos["payout"]
            else:
                still.append(pos)
        open_pos = still
        equity = cash + sum(p["cost"] for p in open_pos)
        budget = min(cash, max_event_frac * equity)
        for _, t in g.iterrows():
            # Kelly fraction for a binary contract bought at price q with win prob p
            q, p = t.price, t.pwin
            f = max(0.0, (p - q) / (1 - q)) * kelly
            n = int(min(max_fill, (f * equity) // q, budget // (q + 0.02)))
            if n < 1:
                continue
            c = n * q + fee(n, q)
            if c > budget:
                continue
            budget -= c; cash -= c
            payout = n * 1.0 if t.win else 0.0
            open_pos.append(dict(S=t.S, cost=c, payout=payout))
            log.append(dict(S=t.S, ticker=t.ticker, side=t.side, n=n, price=q, fee=fee(n, q),
                            p=p, edge=t.edge, win=t.win, cost=c, pnl=payout - c))
    cash += sum(p["payout"] for p in open_pos)
    return cash, pd.DataFrame(log)


def per_contract_stats(sig):
    """Flat 1-contract-per-signal stats (independent of sizing)."""
    pnl = sig.win - sig.price - [fee(1, q) for q in sig.price]
    return dict(n=len(sig), hit=float(sig.win.mean()) if len(sig) else np.nan,
                avg_price=float(sig.price.mean()) if len(sig) else np.nan,
                avg_edge_claimed=float(sig.edge.mean()) if len(sig) else np.nan,
                pnl_per_contract=float(pnl.mean()) if len(sig) else np.nan,
                total_pnl_1lot=float(pnl.sum()))


def main(path=DATA / "rungs.parquet"):
    df = pd.read_parquet(path)
    df["D"] = df.S - pd.to_timedelta(df.D_off, unit="m")
    r = walk_forward(df)
    r.to_parquet(DATA / "wf_preds.parquet")
    q = r.dropna(subset=["bid", "ask"])
    q = q[q.ask > q.bid]
    mid = (q.bid + q.ask) / 2
    qi = q[(q.bid > 0.02) | (q.ask < 0.98)]
    print(f"rungs with live quotes: {len(q)}   contested (not 0/100): {len(qi)}")
    print(f"Brier on contested rungs  model={brier(qi.p, qi.y):.4f}  market_mid={brier(((qi.bid+qi.ask)/2), qi.y):.4f}")
    cut = r.S.min() + (r.S.max() - r.S.min()) / 2
    print(f"\nthreshold sweep (first half = selection, second half = holdout, split {cut:%Y-%m-%d})")
    for thr in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        a = per_contract_stats(signals(r[r.S < cut], thr))
        b = per_contract_stats(signals(r[r.S >= cut], thr))
        print(f" thr={thr:.2f}  A: n={a['n']:5d} pnl/c={a['pnl_per_contract']:+.4f}  |  "
              f"B: n={b['n']:5d} hit={b['hit']:.3f} pnl/c={b['pnl_per_contract']:+.4f} total1lot={b['total_pnl_1lot']:+.2f}")
    return r


if __name__ == "__main__":
    main()
