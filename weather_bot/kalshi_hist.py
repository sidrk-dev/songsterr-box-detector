"""Download all settled markets for the hourly Kalshi Weather Index series."""
import json, sys, time, requests, pathlib
API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = pathlib.Path(__file__).parent / "data"

def get(path, **params):
    for i in range(6):
        r = requests.get(API + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** i); continue
        r.raise_for_status(); return r.json()
    r.raise_for_status()

def settled_markets(series):
    out, cursor = [], None
    while True:
        p = dict(series_ticker=series, status="settled", limit=1000)
        if cursor: p["cursor"] = cursor
        d = get("/markets", **p)
        out += d["markets"]; cursor = d.get("cursor")
        if not cursor or not d["markets"]: break
    return out

if __name__ == "__main__":
    for s in sys.argv[1:]:
        m = settled_markets(s)
        (OUT / f"settled_{s}.json").write_text(json.dumps(m))
        times = sorted(x["close_time"] for x in m)
        print(s, len(m), times[0] if times else None, times[-1] if times else None)
