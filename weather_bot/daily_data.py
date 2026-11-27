"""Training data for the daily-high model: IEM METAR history + official NWS CLI highs."""
import json, pathlib, requests, pandas as pd
DATA = pathlib.Path(__file__).parent / "data"
DAILY = {"KXHIGHLAX": ("LAX", "America/Los_Angeles"), "KXHIGHTPHX": ("PHX", "America/Phoenix"),
         "KXHIGHMIA": ("MIA", "America/New_York"), "KXHIGHCHI": ("MDW", "America/Chicago"),
         "KXHIGHTSFO": ("SFO", "America/Los_Angeles"), "KXHIGHDEN": ("DEN", "America/Denver"),
         "KXHIGHPHIL": ("PHL", "America/New_York"), "KXHIGHTDC": ("DCA", "America/New_York"),
         "KXHIGHTDAL": ("DFW", "America/Chicago"), "KXHIGHTHOU": ("HOU", "America/Chicago")}

def metars(start="2023-08-25", end="2026-11-28"):
    p = [("station", s) for s, _ in DAILY.values()] + [
        ("data", "metar"), ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("missing", "M"),
        ("direct", "no"), ("report_type", "3"), ("report_type", "4"),
        ("year1", start[:4]), ("month1", start[5:7]), ("day1", start[8:10]),
        ("year2", end[:4]), ("month2", end[5:7]), ("day2", end[8:10])]
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", params=p, timeout=1800)
    r.raise_for_status(); (DATA / "daily_metar.csv").write_text(r.text)
    print("metar rows", len(r.text.splitlines()))

def cli():
    rows = []
    for s, _ in DAILY.values():
        for y in range(2023, 2027):
            d = requests.get("https://mesonet.agron.iastate.edu/json/cli.py", params=dict(station="K" + s, year=y), timeout=120).json()
            rows += [dict(station=s, date=x["valid"], high=x["high"], product=x["product"]) for x in d["results"]]
    df = pd.DataFrame(rows); df.to_csv(DATA / "cli.csv", index=False); print("cli rows", len(df))

if __name__ == "__main__":
    cli(); metars()
