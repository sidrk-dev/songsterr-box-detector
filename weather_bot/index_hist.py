"""Download Kalshi's minute-resolution Weather Index series (public live_data endpoint)."""
import json, pathlib, sys, time
import pandas as pd
import requests

API = "https://api.elections.kalshi.com/trade-api/v2/live_data/weather/"
DATA = pathlib.Path(__file__).parent / "data"
INDEX_CITY = {"KXTEMPNYCHS": "nyc", "KXTEMPMIAH": "miami", "KXTEMPLAXHS": "la-coastal", "KXTEMPCHIHS": "chicago"}


def fetch(city, frm_ms, to_ms):
    for i in range(20):
        r = requests.get(API + city, params={"from": frm_ms, "to": to_ms}, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1 + i); continue
        r.raise_for_status()
        return r.json()["timeseries"]
    raise RuntimeError(city)


def recent(city, last_sec=7200):
    r = requests.get(API + city, params={"last_sec": last_sec}, timeout=30)
    r.raise_for_status()
    return to_series(r.json()["timeseries"])


def to_series(ts):
    if not ts:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(x["t"], unit="ms", tz="UTC"): x["v"] for x in ts if x.get("v") is not None})
    return s.sort_index()


if __name__ == "__main__":
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-10-12", tz="UTC")
    end = pd.Timestamp.now(tz="UTC")
    for series, city in INDEX_CITY.items():
        pts, t = [], start
        while t < end:
            t2 = min(t + pd.Timedelta(hours=24), end)
            pts += fetch(city, int(t.timestamp() * 1000), int(t2.timestamp() * 1000) - 1)
            t = t2
        s = to_series(pts)
        s.rename("v").to_frame().to_parquet(DATA / f"minute_{series}.parquet")
        print(series, city, len(s), s.index.min(), s.index.max())
