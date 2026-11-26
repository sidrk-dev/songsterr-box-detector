"""Build the rung-level dataset (features + market quotes + outcome) for backtesting."""
import json, pathlib, sys
import numpy as np
import pandas as pd
from config import CITIES
from features import CityState, load_metars, rung_rows

DATA = pathlib.Path(__file__).parent / "data"


def index_series(markets):
    """UTC settlement hour -> settled index value (°F)."""
    d = {}
    for m in markets:
        v = m.get("expiration_value")
        if v not in (None, ""):
            d[pd.Timestamp(m["close_time"])] = float(v)
    return pd.Series(d).sort_index()


def quote_at(candles, t_unix):
    """Last yes bid/ask quoted at or before t (from 1-min candles that only exist on change)."""
    bid = ask = None
    for ts, b, a, _ in candles:
        if ts > t_unix:
            break
        bid, ask = b, a
    return bid, ask


def build(decision_offsets_min=(54,), cities=None):
    metars = load_metars(pd.read_csv(DATA / "metar.csv"))
    out = []
    for city in cities or CITIES:
        mk = json.loads((DATA / f"settled_{city}.json").read_text())
        idx = index_series(mk)
        st = CityState(city, idx, metars)
        by_ev = {}
        for m in mk:
            by_ev.setdefault(m["event_ticker"], []).append(m)
        for ev, ms in by_ev.items():
            S = pd.Timestamp(ms[0]["close_time"])
            if S not in idx.index:
                continue
            cf = DATA / "candles" / city / f"{ev}.json"
            candles = json.loads(cf.read_text()) if cf.exists() else None
            ms = sorted(ms, key=lambda m: m["floor_strike"])
            strikes = [m["floor_strike"] for m in ms]
            for off in decision_offsets_min:
                D = S - pd.Timedelta(minutes=off)
                f = st.features(S, D)
                if f is None:
                    continue
                rows = rung_rows(f, strikes)
                rows["city_ticker"], rows["event"], rows["S"], rows["D_off"] = city, ev, S, off
                rows["ticker"] = [m["ticker"] for m in ms]
                rows["strike"] = strikes
                rows["y"] = [1 if m["result"] == "yes" else 0 for m in ms]
                rows["settle"] = idx[S]
                bids, asks = [], []
                for m in ms:
                    c = candles.get(m["ticker"]) if candles else None
                    b, a = quote_at(c, int(D.timestamp())) if c else (None, None)
                    bids.append(b); asks.append(a)
                rows["bid"], rows["ask"] = bids, asks
                out.append(rows)
    df = pd.concat(out, ignore_index=True)
    return df


if __name__ == "__main__":
    offs = tuple(int(x) for x in sys.argv[1:]) or (54,)
    df = build(offs)
    df.to_parquet(DATA / "rungs.parquet")
    print(df.shape, df.groupby(["city_ticker", "D_off"]).event.nunique())
    print(df[["bid", "ask"]].notna().mean())
