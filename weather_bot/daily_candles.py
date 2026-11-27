"""Hourly yes bid/ask/trade candles for settled daily-high events (open -> close)."""
import json, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
from kalshi_candles import get, ts, DATA

def fetch(series, ev, op, cl, period):
    out = DATA / f"dcandles{period}" / series / f"{ev}.json"
    if out.exists(): return
    d = get(f"/series/{series}/events/{ev}/candlesticks", start_ts=ts(op), end_ts=ts(cl), period_interval=period)
    res = {}
    for tk, cs in zip(d["market_tickers"], d["market_candlesticks"]):
        f = lambda x, k: float(x[k]["close_dollars"]) if x.get(k) and "close_dollars" in x[k] else None
        res[tk] = [[x["end_period_ts"], f(x, "yes_bid"), f(x, "yes_ask"), float(x.get("volume_fp") or 0)] for x in cs]
    out.write_text(json.dumps(res))

if __name__ == "__main__":
    period = int(sys.argv[1])
    for series in sys.argv[2:]:
        (DATA / f"dcandles{period}" / series).mkdir(parents=True, exist_ok=True)
        m = json.loads((DATA / f"settled_{series}.json").read_text())
        evs = sorted({(x["event_ticker"], x["open_time"], x["close_time"]) for x in m})
        with ThreadPoolExecutor(3) as ex:
            list(ex.map(lambda e: fetch(series, *e, period), evs))
        print(series, len(evs), flush=True)
