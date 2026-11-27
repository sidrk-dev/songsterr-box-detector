"""Model-free check of daily-high market pricing: PnL of taking each side by price and time."""
import json, pathlib, numpy as np, pandas as pd
from backtest import fee
DATA = pathlib.Path(__file__).parent / "data"

def load(period=60):
    rows = []
    for f in (DATA / f"dcandles{period}").glob("*/*.json"):
        series = f.parent.name
        mk = {m["ticker"]: m for m in json.loads((DATA / f"settled_{series}.json").read_text())}
        for tk, cs in json.loads(f.read_text()).items():
            m = mk.get(tk)
            if not m or m["result"] not in ("yes", "no"): continue
            y = 1 if m["result"] == "yes" else 0
            close = pd.Timestamp(m["close_time"]).timestamp()
            for t, b, a, v in cs:
                if b is None or a is None or a <= b: continue
                rows.append((series, m["event_ticker"], tk, t, (close - t) / 3600, b, a, v, y))
    return pd.DataFrame(rows, columns=["series", "event", "ticker", "t", "h_to_close", "bid", "ask", "vol", "y"])

if __name__ == "__main__":
    df = load()
    df["yes_pnl"] = df.y - df.ask - [fee(10, q) / 10 for q in df.ask]
    df["no_pnl"] = (1 - df.y) - (1 - df.bid) - [fee(10, 1 - q) / 10 for q in df.bid]
    print(len(df), "quote-hours,", df.ticker.nunique(), "markets")
    df["hb"] = pd.cut(df.h_to_close, [0, 6, 12, 18, 24, 30, 36, 48])
    pb = [0, .05, .15, .3, .5, .7, .85, .95, 1]
    pd.set_option("display.width", 220)
    se = lambda x: x.std() / np.sqrt(len(x))
    for col, price in (("yes_pnl", "ask"), ("no_pnl", None)):
        p = df.ask if col == "yes_pnl" else 1 - df.bid
        t = df.assign(pb=pd.cut(p, pb)).groupby(["pb"], observed=True)[col].agg(["size", "mean", se]).round(4)
        print("==", col, "by price"); print(t)
        t = df.assign(pb=pd.cut(p, pb)).pivot_table(index="pb", columns="hb", values=col, aggfunc="mean", observed=True).round(3)
        print(t)
