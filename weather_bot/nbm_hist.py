"""Download NBM station guidance (NBS) daily max forecasts from the IEM MOS archive (resumable)."""
import io, pathlib, time, requests, pandas as pd
from daily_data import DAILY
DATA = pathlib.Path(__file__).parent / "data"
PART = DATA / "nbs_parts"

def fetch(station, y):
    f = PART / f"{station}_{y}.parquet"
    if f.exists(): return pd.read_parquet(f)
    for i in range(10):
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py",
                         params=dict(station="K" + station, model="NBS", sts=f"{y}-01-01T00:00Z",
                                     ets=f"{y+1}-01-01T00:00Z", format="csv"), timeout=600)
        if r.status_code == 429: time.sleep(30 * (i + 1)); continue
        r.raise_for_status(); break
    d = pd.read_csv(io.StringIO(r.text), usecols=["runtime", "ftime", "txn", "xnd", "tmp"])
    d = d[d.txn.notna()].assign(station=station); d.to_parquet(f); time.sleep(3); return d

if __name__ == "__main__":
    PART.mkdir(exist_ok=True)
    parts = [fetch(st, y) for st, _ in DAILY.values() for y in (2023, 2024, 2025, 2026)]
    df = pd.concat(parts)
    df["runtime"] = pd.to_datetime(df.runtime, utc=True); df["ftime"] = pd.to_datetime(df.ftime, utc=True)
    df.to_parquet(DATA / "nbs.parquet"); print(df.shape, df.runtime.min(), df.runtime.max())
