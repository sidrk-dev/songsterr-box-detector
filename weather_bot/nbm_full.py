"""Full NBS guidance (3-hourly tmp + daily txn/xnd) for all daily-market stations (resumable)."""
import io, pathlib, time, requests, pandas as pd, sys
DATA = pathlib.Path(__file__).parent / "data"; PART = DATA / "nbs_full"

def fetch(st, y):
    f = PART / f"{st}_{y}.parquet"
    if f.exists(): return pd.read_parquet(f)
    for i in range(12):
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py",
                         params=dict(station="K" + st, model="NBS", sts=f"{y}-01-01T00:00Z", ets=f"{y+1}-01-01T00:00Z", format="csv"), timeout=900)
        if r.status_code == 429: time.sleep(30 * (i + 1)); continue
        r.raise_for_status(); break
    d = pd.read_csv(io.StringIO(r.text), usecols=["runtime", "ftime", "tmp", "txn", "xnd"]).assign(station=st)
    d["runtime"] = pd.to_datetime(d.runtime, utc=True); d["ftime"] = pd.to_datetime(d.ftime, utc=True)
    d.to_parquet(f); time.sleep(3); return d

if __name__ == "__main__":
    PART.mkdir(exist_ok=True)
    from daily_data import DAILY; from lows import LOWS
    sts = sorted({st for st, _ in list(DAILY.values()) + list(LOWS.values())})
    for st in sts:
        for y in (2023, 2024, 2025, 2026): fetch(st, y)
        print(st, flush=True)
