"""Paper trader for Kalshi hourly Weather Index markets.

Run once per hour shortly after markets open (e.g. at :05). Each run:
  1. settles open paper positions whose markets have been determined,
  2. for every city, builds features for the event that settles at the next top of hour,
  3. compares model probabilities with the live order book and records paper fills.
State lives in paper/state.json and paper/trades.csv so it can be committed between runs.
No orders are ever sent to Kalshi; only public, unauthenticated GET endpoints are used.
"""
import json, math, pathlib, time, datetime as dt
import numpy as np
import pandas as pd
import requests
from config import CITIES
from features import CityState, load_metars, rung_rows
from model import LadderModel
from backtest import fee, signals

ROOT = pathlib.Path(__file__).parent
PAPER = ROOT / "paper"
API = "https://api.elections.kalshi.com/trade-api/v2"
METAR_API = "https://aviationweather.gov/api/data/metar"
STRAT = json.loads((ROOT / "models" / "strategy.json").read_text())
S_ = requests.Session()


def get(url, **params):
    for i in range(10):
        try:
            r = S_.get(url, params=params, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(1 + i); continue
            r.raise_for_status(); return r.json()
        except requests.RequestException:
            time.sleep(1 + i)
    raise RuntimeError(url)


def load_state():
    f = PAPER / "state.json"
    if f.exists():
        return json.loads(f.read_text())
    return dict(start_cash=STRAT["bankroll"], cash=STRAT["bankroll"], open=[], traded_events=[],
                started=dt.datetime.now(dt.timezone.utc).isoformat(), runs=[])


def save_state(st):
    PAPER.mkdir(exist_ok=True)
    (PAPER / "state.json").write_text(json.dumps(st, indent=1))


def append_trades(rows, name="trades.csv"):
    if not rows:
        return
    f = PAPER / name
    df = pd.DataFrame(rows)
    df.to_csv(f, mode="a", header=not f.exists(), index=False)


def settle(st):
    done, still = [], []
    for pos in st["open"]:
        m = get(f"{API}/markets/{pos['ticker']}")["market"]
        if m.get("result") in ("yes", "no"):
            won = (m["result"] == pos["side"])
            payout = pos["n"] * 1.0 if won else 0.0
            st["cash"] += payout
            done.append(dict(pos, result=m["result"], index_value=m.get("expiration_value"),
                             payout=payout, pnl=round(payout - pos["cost"], 4),
                             settled_at=dt.datetime.now(dt.timezone.utc).isoformat()))
        else:
            still.append(pos)
    st["open"] = still
    append_trades(done, "settlements.csv")
    return done


def index_history(series, now):
    lo = int((now - pd.Timedelta(hours=27)).timestamp())
    d = get(f"{API}/markets", series_ticker=series, min_close_ts=lo, limit=1000)
    s = {}
    for m in d["markets"]:
        v = m.get("expiration_value")
        if v not in (None, ""):
            s[pd.Timestamp(m["close_time"])] = float(v)
    return pd.Series(s).sort_index()


def live_metars(now):
    ids = ",".join("K" + s for c in CITIES.values() for s in c["stations"])
    d = get(METAR_API, ids=ids, format="json", hours=4)
    df = pd.DataFrame([dict(station=x["icaoId"][1:], valid=pd.Timestamp(x["obsTime"], unit="s", tz="UTC"),
                            metar=x["rawOb"]) for x in d])
    return load_metars(df)


def run(now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    model = LadderModel.load(str(ROOT / "models" / "ladder.json"))
    st = load_state()
    settled = settle(st)
    metars = live_metars(now)
    equity = st["cash"] + sum(p["cost"] for p in st["open"])
    fills, notes = [], []
    for city in CITIES:
        S = now.ceil("h")
        mins_left = (S - now).total_seconds() / 60
        if not (STRAT["min_lead"] <= mins_left <= STRAT["max_lead"]):
            notes.append(f"{city}: {mins_left:.0f} min to settle, outside trading window"); continue
        mk = get(f"{API}/markets", series_ticker=city, status="open", limit=200)["markets"]
        mk = [m for m in mk if pd.Timestamp(m["close_time"]) == S]
        if not mk:
            notes.append(f"{city}: no open event for {S}"); continue
        ev = mk[0]["event_ticker"]
        if ev in st["traded_events"]:
            notes.append(f"{city}: {ev} already handled"); continue
        stt = CityState(city, index_history(city, now), metars)
        f = stt.features(S, now)
        if f is None:
            notes.append(f"{city}: features unavailable (index/METAR missing)"); continue
        mk = sorted(mk, key=lambda m: m["floor_strike"])
        rows = rung_rows(f, [m["floor_strike"] for m in mk])
        rows["ticker"] = [m["ticker"] for m in mk]
        rows["bid"] = [float(m["yes_bid_dollars"]) for m in mk]
        rows["ask"] = [float(m["yes_ask_dollars"]) for m in mk]
        rows["ask_size"] = [float(m.get("yes_ask_size_fp") or 0) for m in mk]
        rows["bid_size"] = [float(m.get("yes_bid_size_fp") or 0) for m in mk]
        rows["p"] = model.predict(rows)
        rows["y"] = 0; rows["S"] = S; rows["D"] = now
        sig = signals(rows, STRAT["edge_threshold"])
        budget = min(st["cash"], STRAT["max_event_frac"] * equity)
        for _, t in sig.iterrows():
            q, p = t.price, t.pwin
            depth = t.ask_size if t.side == "yes" else t.bid_size
            kf = max(0.0, (p - q) / (1 - q)) * STRAT["kelly"]
            n = int(min(STRAT["max_fill"], depth, (kf * equity) // q, budget // (q + 0.02)))
            if n < 1:
                continue
            c = round(n * q + fee(n, q), 4)
            if c > budget:
                continue
            budget -= c; st["cash"] -= c
            pos = dict(ticker=t.ticker, event=ev, city=city, side=t.side, n=n, price=q, fee=fee(n, q),
                       cost=c, model_p=round(float(t.p), 4), edge=round(float(t.edge), 4),
                       i0=f["i0"], settle_time=S.isoformat(), filled_at=now.isoformat())
            st["open"].append(pos); fills.append(pos)
        st["traded_events"].append(ev)
        best = rows.assign(mid=(rows.bid + rows.ask) / 2)[["ticker", "p", "bid", "ask"]]
        notes.append(f"{city}: {ev} i0={f['i0']:.2f} signals={len(sig)} fills={sum(1 for x in fills if x['event']==ev)}")
        append_trades([dict(run=now.isoformat(), event=ev, **r) for r in best.to_dict("records")], "quotes.csv")
    st["traded_events"] = st["traded_events"][-500:]
    append_trades(fills, "fills.csv")
    equity = st["cash"] + sum(p["cost"] for p in st["open"])
    st["runs"].append(dict(t=now.isoformat(), cash=round(st["cash"], 4), equity_at_cost=round(equity, 4),
                           fills=len(fills), settled=len(settled)))
    save_state(st)
    print("\n".join(notes))
    print(f"cash={st['cash']:.2f} open={len(st['open'])} equity(at cost)={equity:.2f} "
          f"new fills={len(fills)} settled={len(settled)}")


if __name__ == "__main__":
    run()
