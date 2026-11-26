"""Feature engineering shared by backtest and live bot.

An "event" is one hourly settlement time S (UTC) for one city's Kalshi Weather Index.
Features are computed as of a decision time D (< S) using only information that is
public at D:
  * the Kalshi index settlement values of earlier hours (published ~5 min after each hour)
  * METAR/SPECI reports from the index member stations observed at least METAR_LAG
    minutes before D
"""
import math, re
import numpy as np
import pandas as pd
from config import CITIES

METAR_LAG = pd.Timedelta(minutes=3)      # assumed publication latency of a METAR
INDEX_LAG = pd.Timedelta(minutes=6)      # index value for hour H usable from H+6min
MINUTE_LAG = pd.Timedelta(minutes=6)     # minute index point for t usable from t+6min (observed ~5-6)

_T_GROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_TD_MAIN = re.compile(r"\s(M?\d{2})/(M?\d{2})?\s")
_WIND = re.compile(r"\s(\d{3}|VRB)(\d{2,3})(G\d{2,3})?KT\s")
_SKY = re.compile(r"\b(FEW|SCT|BKN|OVC|VV)(\d{3})")
_SKY_FRAC = {"FEW": 0.2, "SCT": 0.45, "BKN": 0.75, "OVC": 1.0, "VV": 1.0}


def c_to_f(c): return c * 9.0 / 5.0 + 32.0


def parse_metar(raw: str) -> dict:
    """Extract temperature/dewpoint (°C, tenths when T-group present), wind, sky."""
    out = dict(temp_c=np.nan, dewp_c=np.nan, wind_kt=np.nan, sky=np.nan, ceiling=np.nan, precip=0.0)
    if not isinstance(raw, str):
        return out
    s = " " + raw + " "
    m = _T_GROUP.search(s)
    if m:
        t = int(m.group(2)) / 10.0 * (-1 if m.group(1) == "1" else 1)
        d = int(m.group(4)) / 10.0 * (-1 if m.group(3) == "1" else 1)
        out["temp_c"], out["dewp_c"] = t, d
    else:
        body = s.split(" RMK ")[0] + " "
        m = _TD_MAIN.search(body)
        if m:
            def conv(x): return -int(x[1:]) if x.startswith("M") else int(x)
            out["temp_c"] = conv(m.group(1))
            if m.group(2):
                out["dewp_c"] = conv(m.group(2))
    m = _WIND.search(s)
    if m:
        out["wind_kt"] = float(m.group(2))
    body = s.split(" RMK ")[0]
    layers = _SKY.findall(body)
    if layers:
        out["sky"] = max(_SKY_FRAC[c] for c, _ in layers)
        ceil = [int(h) * 100 for c, h in layers if c in ("BKN", "OVC", "VV")]
        out["ceiling"] = min(ceil) if ceil else 12000
    elif " CLR " in body or " SKC " in body or " CAVOK " in body:
        out["sky"], out["ceiling"] = 0.0, 12000
    if re.search(r"\s[-+]?(VC)?(TS|SH)?(RA|SN|DZ|PL|GR|UP)", body):
        out["precip"] = 1.0
    return out


def load_metars(df: pd.DataFrame) -> pd.DataFrame:
    """df has columns station, valid (UTC), metar -> parsed, sorted frame."""
    parsed = pd.DataFrame([parse_metar(x) for x in df["metar"]], index=df.index)
    out = pd.concat([df[["station"]], parsed], axis=1)
    out["valid"] = pd.to_datetime(df["valid"], utc=True)
    return out.dropna(subset=["temp_c"]).sort_values("valid").reset_index(drop=True)


def solar_elev(lat, lon, t: pd.Timestamp) -> float:
    """Approximate solar elevation angle in degrees."""
    doy = t.dayofyear
    hour = t.hour + t.minute / 60.0
    decl = 23.44 * math.sin(math.radians(360 / 365 * (doy - 81)))
    ha = (hour + lon / 15.0 - 12) * 15
    la, de, h = map(math.radians, (lat, decl, ha))
    return math.degrees(math.asin(math.sin(la) * math.sin(de) + math.cos(la) * math.cos(de) * math.cos(h)))


CITY_LATLON = {"KXTEMPNYCHS": (40.75, -73.9), "KXTEMPMIAH": (25.9, -80.2),
               "KXTEMPLAXHS": (33.9, -118.3), "KXTEMPCHIHS": (41.9, -87.9)}
CITY_ID = {c: i for i, c in enumerate(CITIES)}


class CityState:
    """Holds per-city history (index series + METARs) and builds feature rows."""

    def __init__(self, city: str, index: pd.Series, metars: pd.DataFrame, minute: pd.Series | None = None):
        self.city = city
        self.cfg = CITIES[city]
        self.index = index.sort_index()            # UTC hour -> index °F
        self.minute = minute.sort_index() if minute is not None and len(minute) else None
        st = list(self.cfg["stations"])
        self.m = {s: g.set_index("valid").sort_index() for s, g in metars[metars.station.isin(st)].groupby("station")}

    def idx_at(self, t, D):
        """Index value for hour t if it was published by decision time D."""
        if t + INDEX_LAG > D:
            return np.nan
        return self.index.get(t, np.nan)

    def metar_snapshot(self, asof):
        """Offset-adjusted network means from latest METAR per station observed <= asof."""
        vals, dews, winds, skys, ceils, prec, ages = [], [], [], [], [], [], []
        for s, off in self.cfg["stations"].items():
            g = self.m.get(s)
            if g is None:
                continue
            g = g.loc[:asof]
            if g.empty:
                continue
            row = g.iloc[-1]
            age = (asof - g.index[-1]).total_seconds() / 60
            if age > 120:
                continue
            vals.append(row.temp_c - off); dews.append(row.dewp_c); winds.append(row.wind_kt)
            skys.append(row.sky); ceils.append(row.ceiling); prec.append(row.precip); ages.append(age)
        if not vals:
            return None
        n = len(self.cfg["stations"])
        bc = sum(self.cfg["stations"].values()) / n
        return dict(p=c_to_f(np.mean(vals) + bc), spread=np.std(vals) * 1.8, n=len(vals),
                    dewdep=c_to_f(np.nanmean(vals) + bc) - c_to_f(np.nanmean(dews)) if np.isfinite(np.nanmean(dews)) else np.nan,
                    wind=np.nanmean(winds), sky=np.nanmean(skys) if np.isfinite(np.nanmean(skys)) else np.nan,
                    ceil=np.nanmedian(ceils) if np.isfinite(np.nanmedian(ceils)) else np.nan,
                    precip=np.mean(prec), age=np.mean(ages))

    def features(self, S: pd.Timestamp, D: pd.Timestamp) -> dict | None:
        """Feature row for settlement hour S as of decision time D."""
        H = pd.Timedelta(hours=1)
        # most recent published index hour
        last = S - H
        while last + INDEX_LAG > D:
            last -= H
        i0 = self.idx_at(last, D)
        if not np.isfinite(i0):
            return None
        i_hour = i0
        lead_min = (S - last).total_seconds() / 60
        mf = {}
        if self.minute is not None:
            # minute index points are published ~5 min after their event minute
            w = self.minute.loc[D - pd.Timedelta(minutes=90): D - MINUTE_LAG]
            if len(w) == 0 or (D - w.index[-1]) > pd.Timedelta(minutes=20):
                return None
            t_now, i0 = w.index[-1], float(w.iloc[-1])
            lead_min = (S - t_now).total_seconds() / 60
            for k in (5, 15, 30, 60):
                prev = w.loc[:t_now - pd.Timedelta(minutes=k)]
                mf[f"slope{k}"] = i0 - float(prev.iloc[-1]) if len(prev) else np.nan
            last15 = w.loc[t_now - pd.Timedelta(minutes=15):]
            mf["vol15"] = float(last15.diff().abs().sum())
            mf["i_minus_hour"] = i0 - i_hour
        f = dict(city=CITY_ID[self.city], lead=lead_min, i0=i0, **mf)
        for k in (1, 2, 3, 6):
            f[f"di{k}"] = i_hour - self.idx_at(last - k * H, D)
        # same hour yesterday: realised change from (S-24h - lead) to S-24h
        y1, y0 = self.idx_at(S - 24 * H, D), self.idx_at(last - 24 * H, D)
        f["yday_delta"] = y1 - y0
        f["yday_level"] = y1 - i0
        if self.minute is not None:
            # yesterday's realised change over the same remaining lead
            ys = self.minute.get(S - 24 * H, np.nan)
            yb = self.minute.loc[:S - 24 * H - pd.Timedelta(minutes=lead_min)]
            f["yday_delta_m"] = ys - float(yb.iloc[-1]) if len(yb) else np.nan
        m_now = self.metar_snapshot(D - METAR_LAG)
        m_prev = self.metar_snapshot(D - METAR_LAG - H)
        if m_now is None:
            return None
        f["metar_minus_i0"] = m_now["p"] - i0
        f["metar_age"] = m_now["age"]
        f["metar_trend"] = m_now["p"] - m_prev["p"] if m_prev else np.nan
        for k in ("spread", "n", "dewdep", "wind", "sky", "ceil", "precip"):
            f["m_" + k] = m_now[k]
        if m_prev:
            f["sky_trend"] = (m_now["sky"] - m_prev["sky"]) if np.isfinite(m_now["sky"]) and np.isfinite(m_prev["sky"]) else np.nan
        lat, lon = CITY_LATLON[self.city]
        e_s, e_0 = solar_elev(lat, lon, S), solar_elev(lat, lon, last)
        f["sun_s"], f["sun_d"] = e_s, e_s - e_0
        loc = S.tz_convert(self.cfg["tz"])
        f["hour_sin"], f["hour_cos"] = math.sin(2 * math.pi * loc.hour / 24), math.cos(2 * math.pi * loc.hour / 24)
        f["doy_sin"] = math.sin(2 * math.pi * S.dayofyear / 365.25)
        return f


FEATURES = None  # filled lazily by model module


def rung_rows(feat: dict, strikes) -> pd.DataFrame:
    """Expand an event feature row into one row per rung (strike), with relative strike k."""
    rows = []
    for x in strikes:
        r = dict(feat)
        r["k"] = x - feat["i0"]          # how far the index must move to exceed strike
        # strike distance in units of typical move over the remaining lead (~0.2°F/sqrt(min))
        r["kz"] = r["k"] / (0.2 * math.sqrt(max(feat["lead"], 1.0)))
        rows.append(r)
    return pd.DataFrame(rows)
