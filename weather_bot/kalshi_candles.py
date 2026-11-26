"""Download 1-minute yes bid/ask candles for every settled hourly event (open -> close)."""
import json, random, sys, time, pathlib, datetime as dt, requests
from concurrent.futures import ThreadPoolExecutor
API = "https://api.elections.kalshi.com/trade-api/v2"
DATA = pathlib.Path(__file__).parent / "data"
S = requests.Session()

def get(path, **params):
    for i in range(40):
        try:
            r = S.get(API + path, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2); continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(0.5 + random.random()); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError(path)

def ts(s): return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())

def fetch(series, ev, close):
    out = DATA / "candles" / series / f"{ev}.json"
    if out.exists(): return
    c = ts(close)
    d = get(f"/series/{series}/events/{ev}/candlesticks", start_ts=c - 3600, end_ts=c, period_interval=1)
    res = {}
    for tk, cs in zip(d["market_tickers"], d["market_candlesticks"]):
        res[tk] = [[x["end_period_ts"],
                    float(x["yes_bid"].get("close_dollars", "nan")) if x.get("yes_bid") else None,
                    float(x["yes_ask"].get("close_dollars", "nan")) if x.get("yes_ask") else None,
                    float(x.get("volume_fp") or 0)] for x in cs]
    out.write_text(json.dumps(res))

if __name__ == "__main__":
    for series in sys.argv[1:]:
        (DATA / "candles" / series).mkdir(parents=True, exist_ok=True)
        m = json.loads((DATA / f"settled_{series}.json").read_text())
        evs = sorted({(x["event_ticker"], x["close_time"]) for x in m})
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda e: fetch(series, *e), evs))
        print(series, "done", len(evs), flush=True)
