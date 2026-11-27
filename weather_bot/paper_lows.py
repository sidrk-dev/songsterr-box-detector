"""Live PAPER trader for Kalshi daily-LOW temperature markets (evening strategy).

Pre-registered rules (fixed before any live data, from the Sep-Nov 2026 backtest):
  * trade only when the climate day's local-standard hour is >= MIN_HOUR (21:00 LST) up to 02:59 LST next day
  * model = DailyModel trained on negated temperatures (lows.py), features exactly as in training
  * take a side when model edge after taker fees >= EDGE (0.15), price between 2c and 98c
  * one position per market, quarter-Kelly sizing, max 25% of equity per event, $20 start
No orders are ever sent. Only public endpoints are used (Kalshi, aviationweather.gov, IEM).
Run at about :02 past every hour. State: paper_lows/state.json, fills.csv, settlements.csv, snapshots.csv
"""
import io, json, math, pathlib, time, datetime as dt
import numpy as np, pandas as pd, requests
import daily_model as dm, nbm_feats as nf
from lows import LOWS, _MIN6, market_prob_low, add_nbm_low, patch
from nbm_rem import add_remaining
from daily_model import DailyModel, expand_rungs
from daily_bt import prob_table, signals
from features import parse_metar, c_to_f
from backtest import fee

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "paper_lows"
API = "https://api.elections.kalshi.com/trade-api/v2"
MIN_HOUR, MAX_HOUR, EDGE = 21, 26, 0.15
BANKROLL, KELLY, EVENT_FRAC, MAX_FILL = 20.0, 0.25, 0.25, 25
S = requests.Session(); S.headers["User-Agent"] = "weather-paper-bot"


def get(url, **p):
    for i in range(8):
        try:
            r = S.get(url, params=p, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 2 * i); continue
            r.raise_for_status(); return r
        except requests.RequestException:
            time.sleep(2 + 2 * i)
    raise RuntimeError(url)


def live_frames(stations):
    ids = ",".join("K" + s for s in stations)
    d = get("https://aviationweather.gov/api/data/metar", ids=ids, format="json", hours=36).json()
    rows = []
    for x in d:
        raw = x.get("rawOb") or ""
        rec = parse_metar(raw)
        rmk = raw.split(" RMK ")[1] if " RMK " in raw else ""
        m = _MIN6.search(" " + rmk + " ")
        rec["min6_c"] = (int(m.group(2)) / 10 * (-1 if m.group(1) == "1" else 1)) if m else np.nan
        rec["station"] = x["icaoId"][1:]; rec["valid"] = pd.Timestamp(x["obsTime"], unit="s", tz="UTC")
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["temp_c"])
    res = {}
    for st, g in df.groupby("station"):
        g = g.copy(); g["temp_f"] = -c_to_f(g.temp_c); g["max6_f"] = -c_to_f(g.min6_c)
        res[st] = g.set_index("valid").sort_index()
    return res


def live_cli(stations, year):
    rows = []
    for st in stations:
        for r in get("https://mesonet.agron.iastate.edu/json/cli.py", station="K" + st, year=year).json()["results"][-5:]:
            if str(r["low"]).lstrip("-").isdigit():
                rows.append(dict(station=st, date=r["valid"], high=-int(r["low"])))
    return pd.DataFrame(rows)


def live_nbs(stations, now):
    parts = []
    for st in stations:
        t = get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py", station="K" + st, model="NBS",
                sts=(now - pd.Timedelta(hours=30)).strftime("%Y-%m-%dT%H:%MZ"), ets=now.strftime("%Y-%m-%dT%H:%MZ"),
                format="csv").text
        d = pd.read_csv(io.StringIO(t), usecols=["runtime", "ftime", "tmp", "txn", "xnd"]).assign(station=st)
        parts.append(d)
    n = pd.concat(parts)
    n["runtime"] = pd.to_datetime(n.runtime, utc=True); n["ftime"] = pd.to_datetime(n.ftime, utc=True)
    return n


def load_state():
    f = OUT / "state.json"
    if f.exists(): return json.loads(f.read_text())
    return dict(cash=BANKROLL, start=BANKROLL, open=[], traded=[], runs=[],
                started=dt.datetime.now(dt.timezone.utc).isoformat())


def append(name, rows):
    if rows:
        f = OUT / name; pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)


def settle(st):
    done, still = [], []
    for p in st["open"]:
        m = get(f"{API}/markets/{p['ticker']}").json()["market"]
        if m.get("result") in ("yes", "no"):
            pay = p["n"] * 1.0 if m["result"] == p["side"] else 0.0
            st["cash"] += pay
            done.append(dict(p, result=m["result"], value=m.get("expiration_value"), payout=pay, pnl=round(pay - p["cost"], 4)))
        else:
            still.append(p)
    st["open"] = still; append("settlements.csv", done); return done


def run(now=None):
    OUT.mkdir(exist_ok=True)
    now = now or pd.Timestamp.now(tz="UTC").floor("min")
    patch()
    model = DailyModel.load(str(ROOT / "models" / "low_model.json"))
    st = load_state(); settled = settle(st)
    # which series are inside the trading window right now
    todo = []
    for series, (stn, tz) in LOWS.items():
        lst = (now + pd.Timedelta(hours=dm.STD_OFFSET[tz])).tz_localize(None)
        h = lst.hour if lst.hour >= MIN_HOUR else lst.hour + 24
        day = lst.normalize() if lst.hour >= MIN_HOUR else lst.normalize() - pd.Timedelta(days=1)
        if MIN_HOUR <= h <= MAX_HOUR:
            todo.append((series, stn, day, h))
    notes, fills, snaps = [], [], []
    if todo:
        stations = sorted({t[1] for t in todo})
        fr = live_frames(stations); nbs = live_nbs(stations, now)
        cli = live_cli(stations, now.year)
        equity = st["cash"] + sum(p["cost"] for p in st["open"])
        for series, stn, day, h in todo:
            ev = f"{series}-{day:%y%b%d}".upper()
            if stn not in fr:
                notes.append(f"{ev}: no METARs"); continue
            c = pd.concat([cli[cli.station == stn], pd.DataFrame([dict(station=stn, date=f"{day:%Y-%m-%d}", high=0)])])
            c = c.drop_duplicates(["station", "date"], keep="first")
            # build_rows needs the label row for `day`; 0 is a placeholder (labels unused live)
            c.loc[(c.station == stn) & (c.date == f"{day:%Y-%m-%d}"), "high"] = 0
            sub = {series: LOWS[series]}
            dm.DAILY = sub; nf.DAILY = sub
            snap = dm.build_rows({stn: fr[stn]}, c, hours=[h], rungs=None, days=(day, day), lag_min=3)
            dm.DAILY = LOWS; nf.DAILY = LOWS
            if snap.empty:
                notes.append(f"{ev}: insufficient obs"); continue
            snap = add_nbm_low(snap, nbs[nbs.txn.notna()])
            snap = add_remaining(snap, sub, negate=True, nbs=nbs)
            rows = expand_rungs(snap); rows["p"] = model.predict(rows)
            entry = prob_table(rows)[(series, day, h)]
            mk = get(f"{API}/markets", event_ticker=ev, status="open").json().get("markets", [])
            if not mk:
                notes.append(f"{ev}: no open markets"); continue
            q = []
            for m in mk:
                mt = dict(strike_type=m["strike_type"], floor=m.get("floor_strike"), cap=m.get("cap_strike"))
                q.append(dict(series=series, event=ev, ticker=m["ticker"], t=int(now.timestamp()), hour=h,
                              bid=float(m["yes_bid_dollars"]), ask=float(m["yes_ask_dollars"]),
                              ask_size=float(m.get("yes_ask_size_fp") or 0), bid_size=float(m.get("yes_bid_size_fp") or 0),
                              y=0, p=market_prob_low(entry, mt)))
            q = pd.DataFrame(q)
            snaps += [dict(run=now.isoformat(), low_so_far=-snap.mx.iloc[0], nbm_min=-snap.txn.iloc[0],
                           fc_rem_min=-snap.fc_rem.iloc[0], **r) for r in q.drop(columns=["y"]).to_dict("records")]
            sig = signals(q[q.ask > q.bid], EDGE)
            budget = min(st["cash"], EVENT_FRAC * equity)
            for _, t in sig.iterrows():
                if t.ticker in st["traded"]: continue
                depth = t.ask_size if t.side == "yes" else t.bid_size
                kf = max(0.0, (t.pwin - t.price) / (1 - t.price)) * KELLY
                n = int(min(MAX_FILL, depth, (kf * equity) // t.price, budget // (t.price + 0.02)))
                if n < 1: continue
                cost = round(n * t.price + fee(n, t.price), 4)
                if cost > budget: continue
                budget -= cost; st["cash"] -= cost
                pos = dict(ticker=t.ticker, event=ev, side=t.side, n=n, price=float(t.price), fee=fee(n, t.price),
                           cost=cost, model_p=round(float(t.p), 4), edge=round(float(t.edge), 4), hour=h,
                           filled_at=now.isoformat())
                st["open"].append(pos); st["traded"].append(t.ticker); fills.append(pos)
            notes.append(f"{ev} h={h} low_so_far={-snap.mx.iloc[0]:.1f} nbm_rem={-snap.fc_rem.iloc[0]:.0f} signals={len(sig)}")
    st["traded"] = st["traded"][-3000:]
    append("fills.csv", fills); append("snapshots.csv", snaps)
    eq = st["cash"] + sum(p["cost"] for p in st["open"])
    st["runs"].append(dict(t=now.isoformat(), cash=round(st["cash"], 4), equity_at_cost=round(eq, 4),
                           fills=len(fills), settled=len(settled), windows=len(todo)))
    (OUT / "state.json").write_text(json.dumps(st, indent=1))
    print("\n".join(notes) or "no series in trading window")
    print(f"cash={st['cash']:.2f} open={len(st['open'])} equity(at cost)={eq:.2f} fills={len(fills)} settled={len(settled)}")


if __name__ == "__main__":
    run()
