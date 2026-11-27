"""Daily-high model: P(official CLI high >= v | observations so far today).

The NWS climate day is midnight-to-midnight local *standard* time. At any time t during
the day the high is bounded below by the running maximum of observed temperatures
(hourly T-groups plus the 6-hour max groups in synoptic METARs). The model learns how
much more the temperature is likely to rise given time of day, current temperature
relative to the running max, trend, moisture, cloud and season.
"""
import math, pathlib, re
import numpy as np, pandas as pd, xgboost as xgb
from daily_data import DAILY
from features import parse_metar, c_to_f

DATA = pathlib.Path(__file__).parent / "data"
STD_OFFSET = {"America/Los_Angeles": -8, "America/Phoenix": -7, "America/New_York": -5,
              "America/Chicago": -6, "America/Denver": -7}
STATION_ID = {s: i for i, (s, _) in enumerate(DAILY.values())}
_MAX6 = re.compile(r"\s1([01])(\d{3})\s")
FEATS = ["k", "station", "hour", "mx", "cur_minus_mx", "trend1", "trend3", "dewdep", "sky", "wind", "precip",
         "doy_sin", "doy_cos", "prev_high_minus_mx", "obs_n", "nbm_minus_mx", "k_vs_nbm", "xnd", "nbm_age",
         "fc_rem_minus_mx", "fc_next_minus_cur", "fc_age", "k_vs_fc"]


def parse_all(path=DATA / "daily_metar.csv"):
    df = pd.read_csv(path)
    p = pd.DataFrame([parse_metar(x) for x in df.metar])
    m6 = df.metar.str.split(" RMK ").str[1].fillna("").apply(
        lambda r: (lambda m: (int(m.group(2)) / 10 * (-1 if m.group(1) == "1" else 1)) if m else np.nan)(_MAX6.search(" " + r + " ")))
    out = pd.concat([df[["station"]], p], axis=1)
    out["max6_c"] = m6
    out["valid"] = pd.to_datetime(df.valid, utc=True)
    return out.dropna(subset=["temp_c"]).sort_values("valid").reset_index(drop=True)


def station_frames(obs):
    res = {}
    for series, (st, tz) in DAILY.items():
        g = obs[obs.station == st].copy()
        g["lst"] = g.valid + pd.Timedelta(hours=STD_OFFSET[tz])
        g["temp_f"] = c_to_f(g.temp_c)
        g["max6_f"] = c_to_f(g.max6_c)
        res[st] = g.set_index("valid").sort_index()
    return res


def build_rows(frames, cli, hours=range(6, 27), rungs=range(-3, 11), days=None, lag_min=3):
    """One row per (station, day, hour, rung)."""
    cli = cli.copy(); cli["date"] = pd.to_datetime(cli.date)
    hi = {(r.station, r.date): int(r.high) for r in cli.itertuples() if str(r.high).lstrip("-").isdigit()}
    out = []
    for series, (st, tz) in DAILY.items():
        g = frames[st]
        off = pd.Timedelta(hours=STD_OFFSET[tz])
        lst_idx = g.index + off
        vals = g.temp_f.values; m6 = g.max6_f.values
        dates = sorted({d for (s, d) in hi if s == st})
        if days is not None:
            dates = [d for d in dates if days[0] <= d <= days[1]]
        for d in dates:
            H = hi.get((st, d)); prev = hi.get((st, d - pd.Timedelta(days=1)))
            if H is None:
                continue
            H = int(H)
            day0 = d  # naive LST midnight
            a = np.searchsorted(lst_idx.tz_localize(None), day0)
            b = np.searchsorted(lst_idx.tz_localize(None), day0 + pd.Timedelta(hours=24))
            if b - a < 12:
                continue
            for h in hours:
                t_lst = day0 + pd.Timedelta(hours=h) - pd.Timedelta(minutes=lag_min)
                e = np.searchsorted(lst_idx.tz_localize(None), t_lst, side="right")
                e = min(e, b)
                if e - a < 2:
                    continue
                v = vals[a:e]
                # 6h max groups: only those whose 6h window lies inside the day
                mm = m6[a:e]
                lsts = lst_idx[a:e].tz_localize(None)
                ok6 = (lsts - pd.Timedelta(hours=6)) >= day0 - pd.Timedelta(minutes=10)
                mx = np.nanmax(np.concatenate([v, mm[ok6 & ~np.isnan(mm)]])) if True else v.max()
                cur = v[-1]
                tt = lsts[-1]
                def back(hrs):
                    j = np.searchsorted(lsts, tt - pd.Timedelta(hours=hrs), side="right") - 1
                    return cur - v[j] if j >= 0 else np.nan
                row = g.iloc[e - 1]
                base = dict(series=series, station=STATION_ID[st], date=d, hour=h, mx=mx, cur_minus_mx=cur - mx,
                            trend1=back(1), trend3=back(3), dewdep=cur - c_to_f(row.dewp_c) if np.isfinite(row.dewp_c) else np.nan,
                            sky=row.sky, wind=row.wind_kt, precip=row.precip,
                            doy_sin=math.sin(2 * math.pi * d.dayofyear / 365.25), doy_cos=math.cos(2 * math.pi * d.dayofyear / 365.25),
                            prev_high_minus_mx=(prev - mx) if (prev is not None and h >= 2) else np.nan,
                            obs_n=e - a, high=H)
                if rungs is None:
                    out.append(base)
                    continue
                m0 = int(math.floor(mx + 0.5))
                for r in rungs:
                    v_ = m0 + r
                    out.append(dict(base, v=v_, k=v_ - mx, y=int(H >= v_)))
    return pd.DataFrame(out)


def expand_rungs(snap, below=3, around=9):
    """One row per candidate threshold v. Covers [round(mx)-below, ...] and round(NBM max) +/- around."""
    parts = []
    m0 = np.floor(snap.mx + 0.5).astype(int)
    c = np.floor(snap.txn.fillna(snap.mx) + 0.5).astype(int)
    lo = np.minimum(m0 - below, c - around)
    lo = np.maximum(lo, m0 - below)           # thresholds below running max are certain
    hi = np.maximum(c + around, m0 + below)
    width = int((hi - lo).max()) + 1
    for r in range(width):
        v = lo + r
        ok = v <= hi
        s = snap[ok].copy(); vv = v[ok]
        s["v"] = vv; s["k"] = vv - s.mx; s["k_vs_nbm"] = vv - s.txn
        s["k_vs_fc"] = vv - s.fc_rem if "fc_rem" in s else np.nan
        s["y"] = (s.high >= vv).astype(int)
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


class DailyModel:
    def __init__(self):
        mono = tuple(-1 if f in ("k", "k_vs_nbm", "k_vs_fc") else 0 for f in FEATS)
        self.clf = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, min_child_weight=20, tree_method="hist", n_jobs=4,
                                     monotone_constraints=mono, eval_metric="logloss")

    def fit(self, df):
        self.clf.fit(df[FEATS], df.y); return self

    def predict(self, df):
        return np.clip(self.clf.predict_proba(df[FEATS])[:, 1], 0.0005, 0.9995)

    def save(self, p): self.clf.save_model(p)

    @classmethod
    def load(cls, p):
        m = cls(); m.clf.load_model(p); return m
