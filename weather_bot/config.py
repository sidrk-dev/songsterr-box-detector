"""Index configurations from Kalshi Weather Index Methodology appendices (equal weights)."""
CITIES = {
    "KXTEMPNYCHS": dict(tz="America/New_York", stations={
        "LGA": 1.0, "EWR": 1.0, "TEB": 1.0, "CDW": 0.0, "HPN": -1.0, "FRG": 0.0, "SMQ": -1.0, "ISP": 0.0}),
    "KXTEMPMIAH": dict(tz="America/New_York", stations={
        "MIA": 0.0, "OPF": -0.5, "FLL": 0.0, "FXE": 0.0, "PMP": 0.0}),
    "KXTEMPLAXHS": dict(tz="America/Los_Angeles", stations={
        "HHR": 0.0, "LAX": -1.0, "SMO": -1.0, "LGB": 0.5, "SNA": 1.0}),
    "KXTEMPCHIHS": dict(tz="America/Chicago", stations={
        "MDW": 1.0, "ORD": 0.5, "PWK": 0.0, "DPA": -0.5, "ARR": -1.0}),
}
ALL_STATIONS = sorted({s for c in CITIES.values() for s in c["stations"]})
