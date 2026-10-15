# -*- coding: utf-8 -*-
"""Timebase literature lookup (OpenAlex + Crossref abstracts). ASCII-only PowerShell is avoided; run with python."""
import json, sys, urllib.parse, urllib.request

QUERIES = [
    ("BT-hop", "Bluetooth frequency hopping 1600 hops per second dwell time 625 microseconds"),
    ("Zigbee-duty", "IEEE 802.15.4 Zigbee duty cycle measurement packet interval 2.4 GHz"),
    ("WiFi-beacon", "IEEE 802.11 beacon interval 100 ms measurement traffic burst duration"),
    ("DECT-frame", "DECT ETSI EN 300 175 TDMA frame 10 ms 24 slots structure"),
    ("Welder-RFI", "welding equipment radio disturbance CISPR broadband emission repetition rate"),
    ("Corona-RFI", "corona discharge radio interference power line pulse repetition rate measurement"),
    ("VFD-RFI", "variable frequency drive conducted emission PWM switching frequency interference"),
    ("MWO-RFI", "microwave oven leakage emission spectrum 2.45 GHz measurement"),
]
OUT = "benchmark_v3/evidence/_lit_timebase.txt"

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "em-benchmark/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))

def abstract_of(w):
    idx = w.get("abstract_inverted_index")
    if not idx:
        return ""
    pairs = [(p, word) for word, ps in idx.items() for p in ps]
    pairs.sort()
    s = " ".join(wd for _p, wd in pairs)
    return s[:700] + " ..." if len(s) > 700 else s

lines = []
for key, q in QUERIES:
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q)
           + "&filter=from_publication_date:2005-01-01&per-page=4"
           + "&select=title,publication_year,doi,abstract_inverted_index,cited_by_count,primary_location"
           + "&mailto=em-benchmark@example.com")
    lines.append("================ %s  ::  %s" % (key, q))
    try:
        r = get(url)
    except Exception as exc:                                  # noqa: BLE001
        lines.append("   [query failed] %s" % exc)
        lines.append("")
        continue
    for w in r.get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        lines.append("[%s] %s  <%s>  (%s cites)" % (w.get("publication_year"), w.get("title"),
                                                    loc.get("display_name"), w.get("cited_by_count")))
        lines.append("   DOI: %s" % w.get("doi"))
        ab = abstract_of(w)
        if ab:
            lines.append("   ABS: %s" % ab)
    lines.append("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("wrote %d lines -> %s" % (len(lines), OUT))