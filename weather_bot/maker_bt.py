"""Backtest a model-filtered passive (maker) strategy on the public trade tape.

Assumed behaviour: at every decision time the bot posts a resting order 1c better than the
best quote, only on rungs/sides where the walk-forward model probability of that side
winning exceeds our price by `margin`. Any later taker trade that hit the existing best
quote on that side is assumed to have filled us first instead (we are alone at the better
price), up to `cap` contracts per trade. Positions are held to settlement.
Maker fee: these series use Kalshi fee_type "quadratic" (taker-only); a sensitivity
case charges 0.0175*P*(1-P) per contract.
"""
import numpy as np, pandas as pd, pathlib
from flow import load_trades

DATA = pathlib.Path(__file__).parent / "data"


def predictions():
    a = pd.read_parquet(DATA / "wf_all.parquet")
    b = pd.read_parquet(DATA / "wf_late.parquet")
    p = pd.concat([a[a.D_off != 15], b])[["ticker", "D_off", "p", "bid", "ask"]]
    return p.drop_duplicates(["ticker", "D_off"])


def attach(trades, preds):
    """Attach the most recent model prediction made strictly before each trade."""
    trades = trades.copy()
    offs = np.array(sorted(preds.D_off.unique()))          # minutes before S
    # most recent decision before trade: smallest offset >= mins_left
    idx = np.searchsorted(offs, trades.mins_left.values, side="left")
    ok = idx < len(offs)
    trades = trades[ok].copy()
    trades["D_off"] = offs[idx[ok]]
    return trades.merge(preds, on=["ticker", "D_off"], how="inner")


def evaluate(t, margin, improve=0.01, cap=10, maker_fee=0.0, max_left=60, min_left=1):
    t = t[(t.mins_left <= max_left) & (t.mins_left >= min_left)]
    our_price = t.maker_price + improve
    p_side = np.where(t.maker_side == "yes", t.p, 1 - t.p)
    sel = (p_side - our_price >= margin) & (our_price < 0.99) & (our_price > 0.01)
    t = t[sel].assign(our_price=our_price[sel])
    n = np.minimum(t["count"], cap)
    fee = maker_fee * t.our_price * (1 - t.our_price)
    pnl = (t.maker_win - t.our_price - fee) * n
    return t.assign(n=n, pnl=pnl)


def summary(x):
    if len(x) == 0:
        return dict(fills=0)
    return dict(fills=len(x), contracts=float(x.n.sum()), pnl=float(x.pnl.sum()),
                per_contract=float(x.pnl.sum() / x.n.sum()), events=x.event.nunique(),
                win=float(np.average(x.maker_win, weights=x.n)), avg_price=float(np.average(x.our_price, weights=x.n)))


if __name__ == "__main__":
    tr = attach(load_trades(), predictions())
    print("trades with predictions:", len(tr), "markets:", tr.ticker.nunique(), "span", tr.S.min(), tr.S.max())
    cut = tr.S.min() + (tr.S.max() - tr.S.min()) / 2
    rows = []
    for margin in (0.0, 0.02, 0.05, 0.08, 0.12):
        for (lo, hi) in ((1, 60), (1, 10), (10, 30), (30, 60)):
            a = summary(evaluate(tr[tr.S < cut], margin, min_left=lo, max_left=hi))
            b = summary(evaluate(tr[tr.S >= cut], margin, min_left=lo, max_left=hi))
            rows.append(dict(margin=margin, window=f"{lo}-{hi}m", A_fills=a.get("fills"), A_pc=a.get("per_contract"),
                             B_fills=b.get("fills"), B_contracts=b.get("contracts"), B_pc=b.get("per_contract"),
                             B_pnl=b.get("pnl"), B_win=b.get("win"), B_price=b.get("avg_price")))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(4).to_string())
