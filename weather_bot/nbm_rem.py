"""Remaining-day NBM forecast features.

For each (series, date, hour) snapshot: take the latest NBS run published by then (runtime
<= t - latency) and compute the forecast extreme of the 3-hourly 2 m temperature over
the rest of the climate day (t, LST midnight]. For lows, temperatures are negated so the
"remaining max" of -T is minus the remaining min.
"""
import pathlib
import numpy as np, pandas as pd
from daily_model import STD_OFFSET

DATA = pathlib.Path(__file__).parent / "data"
LAT = pd.Timedelta(minutes=90)


def load_full(stations):
    parts = [pd.read_parquet(f) for f in (DATA / "nbs_full").glob("*.parquet") if f.stem.split("_")[0] in stations]
    d = pd.concat(parts).dropna(subset=["tmp"])
    return d


def add_remaining(snap, cities, negate=False, nbs=None):
    code = {s: st for s, (st, _) in cities.items()}
    off = {s: STD_OFFSET[tz] for s, (_, tz) in cities.items()}
    if nbs is None:
        nbs = load_full(set(code.values()))
    sign = -1.0 if negate else 1.0
    res = []
    for series, g in snap[["series", "date", "hour"]].drop_duplicates().groupby("series"):
        st = code[series]; o = pd.Timedelta(hours=off[series])
        n = nbs[nbs.station == st]
        n = n.assign(runtime=n.runtime.dt.tz_convert(None), ftime=n.ftime.dt.tz_convert(None))
        runs = np.sort(n.runtime.unique().astype("datetime64[ns]"))
        by_run = {np.datetime64(r, "ns"): (grp.ftime.values.astype("datetime64[ns]"), sign * grp.tmp.values.astype(float)) for r, grp in n.groupby("runtime")}
        day0 = pd.to_datetime(g.date) - o        # LST midnight in UTC
        t = day0 + pd.to_timedelta(g.hour, unit="h")
        t_av = (t - LAT).values.astype("datetime64[ns]")
        end = (day0 + pd.Timedelta(hours=24)).values.astype("datetime64[ns]")
        tv = t.values.astype("datetime64[ns]")
        idx = np.searchsorted(runs, t_av, side="right") - 1
        rem, nxt, age = [], [], []
        for i, j in enumerate(idx):
            if j < 0:
                rem.append(np.nan); nxt.append(np.nan); age.append(np.nan); continue
            ft, tp = by_run[runs[j]]
            m = (ft > tv[i]) & (ft <= end[i])
            rem.append(tp[m].max() if m.any() else np.nan)
            k = np.argmax(ft > tv[i]) if (ft > tv[i]).any() else None
            nxt.append(tp[k] if k is not None else np.nan)
            age.append((tv[i] - runs[j]) / np.timedelta64(1, "h"))
        res.append(g.assign(fc_rem=rem, fc_next=nxt, fc_age=age))
    r = pd.concat(res)
    snap = snap.merge(r, on=["series", "date", "hour"], how="left")
    snap["fc_rem_minus_mx"] = snap.fc_rem - snap.mx
    snap["fc_next_minus_cur"] = snap.fc_next - (snap.mx + snap.cur_minus_mx)
    return snap
