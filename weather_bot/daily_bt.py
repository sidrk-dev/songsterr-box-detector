"""Backtest the daily-high model against hourly Kalshi quotes (out of sample: Sep 19 - Nov 26 2026)."""
import json, math, pathlib
import numpy as np, pandas as pd
from daily_data import DAILY
from daily_model import STD_OFFSET
from daily_bias import load
from backtest import fee

DATA = pathlib.Path(__file__).parent / "data"


def market_meta():
    out = {}
    for series in DAILY:
        for m in json.loads((DATA / f"settled_{series}.json").read_text()):
            out[m["ticker"]] = dict(strike_type=m["strike_type"], floor=m.get("floor_strike"), cap=m.get("cap_strike"),
                                    date=pd.Timestamp(m["event_ticker"].split("-")[1][:7]) if False else None)
    return out


def event_date(ev):
    return pd.to_datetime(ev.split("-")[1], format="%y%b%d")


def prob_table(test):
    """(series, date, hour) -> function v -> P(high >= v)."""
    tab = {}
    for (s, d, h), g in test.groupby(["series", "date", "hour"]):
        tab[(s, d, h)] = (dict(zip(g.v, g.p)), g.v.min(), g.v.max())
    return tab


def p_ge(entry, v):
    d, lo, hi = entry
    if v < lo: return 0.9995
    if v > hi: return 0.0005
    return d[v]


def market_prob(entry, mt):
    if mt["strike_type"] == "greater":
        return p_ge(entry, int(mt["floor"]) + 1)
    if mt["strike_type"] == "less":
        return 1 - p_ge(entry, int(mt["cap"]))
    return max(0.0005, p_ge(entry, int(mt["floor"])) - p_ge(entry, int(mt["cap"]) + 1))


def build():
    test = pd.read_parquet(DATA / "daily_test.parquet")
    tab = prob_table(test)
    q = load(60)
    meta = market_meta()
    q["date"] = q.event.map(event_date)
    rows = []
    for r in q.itertuples():
        st, tz = DAILY[r.series]
        t_lst = pd.Timestamp(r.t, unit="s") + pd.Timedelta(hours=STD_OFFSET[tz])
        h = int(round((t_lst - r.date).total_seconds() / 3600))
        e = tab.get((r.series, r.date, h))
        if e is None:
            continue
        rows.append(dict(series=r.series, event=r.event, ticker=r.ticker, t=r.t, hour=h, bid=r.bid, ask=r.ask,
                         vol=r.vol, y=r.y, p=market_prob(e, meta[r.ticker])))
    return pd.DataFrame(rows)


def signals(df, thr, lot=10):
    yes_e = df.p - df.ask; no_e = df.bid - df.p
    side = np.where(yes_e >= no_e, "yes", "no")
    price = np.where(side == "yes", df.ask, 1 - df.bid)
    pwin = np.where(side == "yes", df.p, 1 - df.p)
    win = np.where(side == "yes", df.y, 1 - df.y)
    f = np.array([fee(lot, x) / lot for x in price])
    edge = pwin - price - f
    s = df.assign(side=side, price=price, pwin=pwin, win=win, fee=f, edge=edge, pnl=win - price - f)
    return s[(s.edge >= thr) & (s.price >= 0.02) & (s.price <= 0.98)]


if __name__ == "__main__":
    df = build(); df.to_parquet(DATA / "daily_bt_rows.parquet")
    c = df[(df.bid + df.ask) / 2 > 0.03]; c = c[(c.bid + c.ask) / 2 < 0.97]
    print(len(df), "quote-hours;", "contested", len(c))
    for h, g in c.groupby(pd.cut(c.hour, [5, 10, 13, 16, 19, 22, 27])):
        print(h, len(g), "brier model %.4f market %.4f" % (((g.p - g.y) ** 2).mean(), (((g.bid + g.ask) / 2 - g.y) ** 2).mean()))
    for thr in (0.02, 0.05, 0.1, 0.15):
        s = signals(df, thr)
        # one trade per market: first signal in time (realistic: we would hold the position)
        s1 = s.sort_values("t").drop_duplicates("ticker")
        cut = s1.t.min() + (s1.t.max() - s1.t.min()) / 2
        a, b = s1[s1.t < cut], s1[s1.t >= cut]
        print(f"thr {thr:.2f}  first-signal-per-market: A n={len(a)} pnl/c={a.pnl.mean():+.4f} | "
              f"B n={len(b)} hit={b.win.mean():.3f} price={b.price.mean():.3f} pnl/c={b.pnl.mean():+.4f} tot={b.pnl.sum():+.2f}")
        print("   by hour:", s1.groupby(pd.cut(s1.hour, [5, 10, 13, 16, 19, 22, 27]), observed=True).pnl.agg(["size", "mean"]).round(3).to_dict("index"))
