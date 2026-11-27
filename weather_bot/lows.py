"""Daily-LOW markets: reuse the daily-high pipeline on negated temperatures.

low = -max(-T), so with T' = -T the running max of T' and the 6-hour minimum groups
(negated) play the role of the high-case inputs. NBM min forecasts (valid 12Z of the day)
are negated likewise. Probabilities are mapped back: P(low >= v) = 1 - P(H' >= -v + 1).
"""
import io, json, pathlib, re, time, pickle, sys
import numpy as np, pandas as pd, requests
import daily_model as dm, nbm_feats as nf
from daily_model import DailyModel, expand_rungs
from features import parse_metar, c_to_f
from backtest import fee

DATA = pathlib.Path(__file__).parent / "data"
TZ = {"OKC": "America/Chicago", "EWR": "America/New_York", "BOS": "America/New_York", "SEA": "America/Los_Angeles",
      "DEN": "America/Denver", "PHX": "America/Phoenix", "SFO": "America/Los_Angeles", "SDF": "America/New_York",
      "HOU": "America/Chicago", "AUS": "America/Chicago", "SAN": "America/Los_Angeles", "MIA": "America/New_York",
      "DFW": "America/Chicago", "LAS": "America/Los_Angeles", "MDW": "America/Chicago", "SAT": "America/Chicago",
      "LAX": "America/Los_Angeles", "PHL": "America/New_York", "TTN": "America/New_York", "MSY": "America/Chicago",
      "NYC": "America/New_York", "DCA": "America/New_York", "ATL": "America/New_York"}
LOWS = {s: (st, TZ[st]) for s, st in json.loads((DATA / "extra_series.json").read_text()).items() if s.startswith("KXLOW")}
_MIN6 = re.compile(r"\s2([01])(\d{3})\s")


def fetch_cli():
    rows = []
    for st in sorted({st for st, _ in LOWS.values()}):
        for y in range(2023, 2027):
            for i in range(5):
                r = requests.get("https://mesonet.agron.iastate.edu/json/cli.py", params=dict(station="K" + st, year=y), timeout=120)
                if r.status_code == 429: time.sleep(20); continue
                break
            rows += [dict(station=st, date=x["valid"], high=x["low"], product=x["product"]) for x in r.json()["results"]]
    pd.DataFrame(rows).to_csv(DATA / "cli_low.csv", index=False)   # 'high' column holds the LOW


def fetch_metar():
    from daily_data import metars
    have = {st for st, _ in dm.DAILY.values()}
    need = {s: v for s, v in LOWS.items() if v[0] not in have}
    if need:
        metars(which=need, out="daily_metar3.csv")


def frames():
    parts = [DATA / "daily_metar.csv", DATA / "daily_metar2.csv", DATA / "daily_metar3.csv"]
    df = pd.concat([pd.read_csv(p) for p in parts if p.exists()])
    df = df[df.station.isin({st for st, _ in LOWS.values()})]
    df = df.reset_index(drop=True)
    out = pd.DataFrame([parse_metar(x) for x in df.metar])
    rmk = df.metar.str.split(" RMK ").str[1].fillna("")
    out["min6_c"] = rmk.apply(lambda r: (lambda m: (int(m.group(2)) / 10 * (-1 if m.group(1) == "1" else 1))
                                         if m else np.nan)(_MIN6.search(" " + r + " ")))
    out["station"] = df.station
    out["valid"] = pd.to_datetime(df.valid, utc=True)
    out = out.dropna(subset=["temp_c"])
    res = {}
    for st, g in out.groupby("station"):
        g = g.copy()
        g["temp_f"] = -c_to_f(g.temp_c)           # negated: running max of this = -(running min)
        g["max6_f"] = -c_to_f(g.min6_c)
        res[st] = g.set_index("valid").sort_index()
    return res


def patch():
    dm.DAILY = LOWS
    nf.DAILY = LOWS
    dm.STATION_ID = {st: i for i, st in enumerate(sorted({st for st, _ in LOWS.values()}))}


def add_nbm_low(snap, nbs):
    """NBM min for climate day d is the txn valid at 12Z of day d; negate it."""
    code = {s: st for s, (st, _) in LOWS.items()}
    off = {s: dm.STD_OFFSET[tz] for s, (_, tz) in LOWS.items()}
    key = snap[["series", "date", "hour"]].drop_duplicates().copy()
    key["st"] = key.series.map(code)
    key["t_avail"] = (pd.to_datetime(key.date).dt.tz_localize("UTC") + pd.to_timedelta(key.hour, unit="h")
                      - pd.to_timedelta(key.series.map(off), unit="h") - nf.NBM_LATENCY)
    key["ftime"] = pd.to_datetime(key.date).dt.tz_localize("UTC") + pd.Timedelta(hours=12)
    n = nbs.rename(columns={"station": "st"}); n = n[n.ftime.dt.hour == 12].sort_values("runtime")
    out = []
    for st, g in key.groupby("st"):
        m = pd.merge_asof(g.sort_values("t_avail"), n[n.st == st][["runtime", "ftime", "txn", "xnd"]],
                          left_on="t_avail", right_on="runtime", by="ftime", direction="backward")
        out.append(m)
    k = pd.concat(out); k["txn"] = -k.txn
    k["nbm_age"] = (k.t_avail - k.runtime).dt.total_seconds() / 3600
    snap = snap.merge(k[["series", "date", "hour", "txn", "xnd", "nbm_age"]], on=["series", "date", "hour"], how="left")
    snap["nbm_minus_mx"] = snap.txn - snap.mx
    return snap


def fetch_nbm():
    from nbm_hist import fetch, PART
    PART.mkdir(exist_ok=True)
    have = pd.read_parquet(DATA / "nbs.parquet")
    need = sorted({st for st, _ in LOWS.values()} - set(have.station.unique()))
    parts = [fetch(st, y) for st in need for y in (2023, 2024, 2025, 2026)]
    if parts:
        p = pd.concat(parts); p["runtime"] = pd.to_datetime(p.runtime, utc=True); p["ftime"] = pd.to_datetime(p.ftime, utc=True)
        have = pd.concat([have, p]); have.to_parquet(DATA / "nbs.parquet")


def low_ge(entry, v):
    from daily_bt import p_ge
    return 1 - p_ge(entry, -v + 1)


def market_prob_low(entry, mt):
    if mt["strike_type"] == "greater":
        return low_ge(entry, int(mt["floor"]) + 1)
    if mt["strike_type"] == "less":
        return 1 - low_ge(entry, int(mt["cap"]))
    return max(0.0005, low_ge(entry, int(mt["floor"])) - low_ge(entry, int(mt["cap"]) + 1))


def backtest():
    from daily_bt import prob_table, event_date, signals
    from daily_bias import load
    test = pd.read_parquet(DATA / "low_test.parquet"); tab = prob_table(test)
    q = load(60); q = q[q.series.isin(LOWS)]
    meta = {}
    for s in LOWS:
        for m in json.loads((DATA / f"settled_{s}.json").read_text()):
            meta[m["ticker"]] = dict(strike_type=m["strike_type"], floor=m.get("floor_strike"), cap=m.get("cap_strike"))
    q["date"] = q.event.map(event_date)
    rows = []
    for r in q.itertuples():
        st, tz = LOWS[r.series]
        t_lst = pd.Timestamp(r.t, unit="s") + pd.Timedelta(hours=dm.STD_OFFSET[tz])
        h = int(round((t_lst - r.date).total_seconds() / 3600))
        e = tab.get((r.series, r.date, h))
        if e is None: continue
        rows.append(dict(series=r.series, event=r.event, ticker=r.ticker, t=r.t, hour=h, bid=r.bid, ask=r.ask,
                         vol=r.vol, y=r.y, p=market_prob_low(e, meta[r.ticker])))
    b = pd.DataFrame(rows); b.to_parquet(DATA / "low_bt_rows.parquet")
    b["mid"] = (b.bid + b.ask) / 2; c = b[(b.mid > .03) & (b.mid < .97)]
    print(len(b), "quote-hours; contested", len(c))
    for h, g in c.groupby(pd.cut(c.hour, [0, 4, 7, 10, 14, 19, 27]), observed=True):
        print(h, len(g), "brier model %.4f market %.4f" % (((g.p - g.y) ** 2).mean(), ((g.mid - g.y) ** 2).mean()))
    for thr in (0.05, 0.1, 0.15, 0.2):
        s = signals(b, thr).sort_values("t").drop_duplicates("ticker")
        cut = s.t.median()
        print(f"thr {thr}: n={len(s)} pnl/c={s.pnl.mean():+.4f} halves {s[s.t<cut].pnl.mean():+.4f}/{s[s.t>=cut].pnl.mean():+.4f} tot={s.pnl.sum():+.2f}")
        print("   by hour", s.groupby(pd.cut(s.hour, [0, 4, 7, 10, 14, 19, 27]), observed=True).pnl.agg(["size", "mean"]).round(3).to_dict("index"))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "bt":
        backtest()
    if step == "data":
        fetch_cli(); fetch_metar(); fetch_nbm()
    elif step == "train":
        patch()
        fr = frames()
        cli = pd.read_csv(DATA / "cli_low.csv")
        cli = cli[cli.high.astype(str).str.lstrip("-").str.isdigit()].copy()
        cli["high"] = -cli.high.astype(int)
        snap = dm.build_rows(fr, cli, hours=range(1, 27), rungs=None)
        snap = add_nbm_low(snap, pd.read_parquet(DATA / "nbs.parquet"))
        rows = expand_rungs(snap); print(rows.shape)
        tr = rows[rows.date < "2026-09-15"]; te = rows[rows.date >= "2026-09-17"]
        m = DailyModel().fit(tr); te = te.assign(p=m.predict(te))
        te.to_parquet(DATA / "low_test.parquet"); m.save(str(DATA / "low_model.json"))
