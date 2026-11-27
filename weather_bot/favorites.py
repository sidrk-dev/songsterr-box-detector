"""Favorite-side strategy analysis: buy whichever side of a rung is priced in [lo, hi]."""
import numpy as np, pandas as pd
from backtest import fee

def favorite_rows(r, lo, hi):
    r = r.dropna(subset=["bid", "ask"]); r = r[r.ask > r.bid]
    a = r[(r.ask >= lo) & (r.ask <= hi)].assign(side="yes", price=lambda x: x.ask, win=lambda x: x.y,
                                                  pside=lambda x: x.p)
    b = r[((1 - r.bid) >= lo) & ((1 - r.bid) <= hi)].assign(side="no", price=lambda x: 1 - x.bid,
                                                              win=lambda x: 1 - x.y, pside=lambda x: 1 - x.p)
    return pd.concat([a, b])

def stats(t, n_contracts):
    f = np.array([fee(n_contracts, q) for q in t.price]) / n_contracts
    pnl = t.win - t.price - f
    return pd.Series(dict(n=len(t), win=t.win.mean(), price=t.price.mean(), gross=(t.win - t.price).mean(),
                          net=pnl.mean(), se=pnl.std() / np.sqrt(max(len(t), 1)), events=t.event.nunique()))
