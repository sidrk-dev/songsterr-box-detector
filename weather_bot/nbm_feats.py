"""Attach the latest available NBM daily-max forecast to each (station, day, hour) row."""
import numpy as np, pandas as pd
from daily_data import DAILY
from daily_model import STD_OFFSET

NBM_LATENCY = pd.Timedelta(minutes=90)


def add_nbm(rows, nbs):
    code = {s: st for s, (st, _) in DAILY.items()}
    off = {s: STD_OFFSET[tz] for s, (_, tz) in DAILY.items()}
    key = rows[["series", "date", "hour"]].drop_duplicates().copy()
    key["st"] = key.series.map(code)
    key["t_avail"] = (pd.to_datetime(key.date).dt.tz_localize("UTC") + pd.to_timedelta(key.hour, unit="h")
                      - pd.to_timedelta(key.series.map(off), unit="h") - NBM_LATENCY)
    key["ftime"] = pd.to_datetime(key.date).dt.tz_localize("UTC") + pd.Timedelta(days=1)
    n = nbs.rename(columns={"station": "st"}).sort_values("runtime")
    n = n[n.ftime.dt.hour == 0]
    out = []
    for st, g in key.groupby("st"):
        ng = n[n.st == st]
        g = g.sort_values("t_avail")
        m = pd.merge_asof(g, ng[["runtime", "ftime", "txn", "xnd"]].sort_values("runtime"),
                          left_on="t_avail", right_on="runtime", by="ftime", direction="backward")
        out.append(m)
    k = pd.concat(out)
    k["nbm_age"] = (k.t_avail - k.runtime).dt.total_seconds() / 3600
    rows = rows.merge(k[["series", "date", "hour", "txn", "xnd", "nbm_age"]], on=["series", "date", "hour"], how="left")
    rows["nbm_minus_mx"] = rows.txn - rows.mx
    if "v" in rows:
        rows["k_vs_nbm"] = rows.v - rows.txn
    return rows
