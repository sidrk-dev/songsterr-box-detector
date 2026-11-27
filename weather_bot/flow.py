"""Trade-tape analysis: how profitable is the order flow we would trade against as a maker?

For every public trade we know the taker's side and price. The resting maker took the
opposite side. If we had quoted 1 cent better than the resting order, we would have
been filled instead, at a price 1 cent worse for the taker.
"""
import json, pathlib
import numpy as np, pandas as pd

DATA = pathlib.Path(__file__).parent / "data"


def outcomes():
    out = {}
    for f in DATA.glob("settled_*.json"):
        for m in json.loads(f.read_text()):
            out[m["ticker"]] = (1 if m["result"] == "yes" else 0, pd.Timestamp(m["close_time"]), m["event_ticker"],
                                f.stem.replace("settled_", ""))
    return out


def load_trades():
    oc = outcomes()
    rows = []
    for f in (DATA / "trades").glob("*.json"):
        tk = f.stem
        if tk not in oc:
            continue
        y, S, ev, city = oc[tk]
        for t, px, n, side in json.loads(f.read_text()):
            rows.append((tk, ev, city, S, t, px, n, side, y))
    df = pd.DataFrame(rows, columns=["ticker", "event", "city", "S", "time", "yes_price", "count", "taker_side", "y"])
    df["time"] = pd.to_datetime(df.time, utc=True, format="ISO8601")
    df["mins_left"] = (df.S - df.time).dt.total_seconds() / 60
    # maker side and the price the maker paid for its side
    df["maker_side"] = np.where(df.taker_side == "yes", "no", "yes")
    df["maker_price"] = np.where(df.maker_side == "yes", df.yes_price, 1 - df.yes_price)
    df["maker_win"] = np.where(df.maker_side == "yes", df.y, 1 - df.y)
    df["maker_pnl"] = df.maker_win - df.maker_price          # per contract, before fees
    return df


if __name__ == "__main__":
    df = load_trades()
    print(len(df), "trades on", df.ticker.nunique(), "markets")
    w = lambda g: pd.Series(dict(trades=len(g), contracts=g["count"].sum(),
                                 maker_pnl_per_contract=np.average(g.maker_pnl, weights=g["count"]),
                                 improved_by_1c=np.average(g.maker_pnl - 0.01, weights=g["count"])))
    df["px_bucket"] = pd.cut(df.maker_price, [0, .05, .2, .4, .6, .8, .95, 1])
    df["t_bucket"] = pd.cut(df.mins_left, [-1, 2, 5, 10, 20, 30, 45, 61, 999])
    print(w(df))
    print(df.groupby("t_bucket", observed=True).apply(w).round(4))
    print(df.groupby("px_bucket", observed=True).apply(w).round(4))
    print(df.groupby("city").apply(w).round(4))
