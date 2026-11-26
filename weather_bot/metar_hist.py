"""Download raw METAR/SPECI history from Iowa Environmental Mesonet."""
import sys, pathlib, requests, io
from config import ALL_STATIONS
DATA = pathlib.Path(__file__).parent / "data"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

def fetch(start, end, out):
    p = [("station", s) for s in ALL_STATIONS] + [
        ("data", "metar"), ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"), ("report_type", "4"),
        ("year1", start[:4]), ("month1", start[5:7]), ("day1", start[8:10]),
        ("year2", end[:4]), ("month2", end[5:7]), ("day2", end[8:10])]
    r = requests.get(URL, params=p, timeout=600); r.raise_for_status()
    out.write_text(r.text); print(out, len(r.text.splitlines()))

if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2], DATA / "metar.csv")
