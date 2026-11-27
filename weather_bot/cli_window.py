"""Minute quotes between the NWS CLI publication and market close, for each daily event."""
import json, pathlib, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from kalshi_candles import get, DATA
from daily_data import DAILY

OUT = DATA / "cliwin"

def windows():
    cli = pd.read_csv(DATA / "cli.csv"); cli["date"] = pd.to_datetime(cli.date)
    code = {st: s for s, (st, _) in DAILY.items()}
    ev = []
    for series in DAILY:
        st = DAILY[series][0]
        m = json.loads((DATA / f"settled_{series}.json").read_text())
        for e, close in sorted({(x["event_ticker"], x["close_time"]) for x in m}):
            d = pd.to_datetime(e.split("-")[1], format="%y%b%d")
            row = cli[(cli.station == st) & (cli.date == d)]
            if row.empty or not str(row.high.iloc[0]).lstrip("-").isdigit(): continue
            pub = pd.to_datetime(row["product"].iloc[0][:12], format="%Y%m%d%H%M", utc=True)
            ev.append((series, e, pub, pd.Timestamp(close), int(row.high.iloc[0])))
    return ev

def fetch(series, e, pub, close, high):
    f = OUT / f"{e}.json"
    if f.exists(): return
    st = int(pub.timestamp()) - 1800; en = int(close.timestamp())
    if en <= st: f.write_text("{}"); return
    d = get(f"/series/{series}/events/{e}/candlesticks", start_ts=st, end_ts=en, period_interval=1)
    res = {}
    for tk, cs in zip(d["market_tickers"], d["market_candlesticks"]):
        g = lambda x, k: float(x[k]["close_dollars"]) if x.get(k) and "close_dollars" in x[k] else None
        res[tk] = [[x["end_period_ts"], g(x, "yes_bid"), g(x, "yes_ask"), float(x.get("volume_fp") or 0),
                    float(x["yes_ask"]["low_dollars"]) if x.get("yes_ask") and "low_dollars" in x["yes_ask"] else None,
                    float(x["yes_bid"]["high_dollars"]) if x.get("yes_bid") and "high_dollars" in x["yes_bid"] else None] for x in cs]
    f.write_text(json.dumps(res))

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ev = windows()
    pd.DataFrame(ev, columns=["series", "event", "pub", "close", "high"]).to_parquet(DATA / "cli_pub.parquet")
    with ThreadPoolExecutor(3) as ex: list(ex.map(lambda x: fetch(*x), ev))
    print("events", len(ev))
