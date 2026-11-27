"""Download the public trade tape for a list of market tickers (newest first, resumable)."""
import json, pathlib, random, sys, time
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor

API = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
DATA = pathlib.Path(__file__).parent / "data"
OUT = DATA / "trades"
S = requests.Session()


def get(params):
    for i in range(60):
        try:
            r = S.get(API, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(0.5 + random.random()); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError(params)


def fetch(tk):
    f = OUT / f"{tk}.json"
    if f.exists():
        return
    out, cur = [], None
    while True:
        p = dict(ticker=tk, limit=1000)
        if cur: p["cursor"] = cur
        d = get(p)
        out += [[t["created_time"], float(t["yes_price_dollars"]), float(t["count_fp"]), t["taker_side"]] for t in d["trades"]]
        cur = d.get("cursor")
        if not cur or not d["trades"]:
            break
    f.write_text(json.dumps(out))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tks = pd.read_csv(DATA / "contested_tickers.csv").iloc[:, 0].tolist()
    # ticker contains the date (e.g. -26NOV2614-); sort newest first
    def key(t):
        d = t.split("-")[1]
        return pd.to_datetime(d[:7], format="%y%b%d").value + int(d[7:9] or 0)
    tks.sort(key=key, reverse=True)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(fetch, tks))
    print("done", len(tks))
