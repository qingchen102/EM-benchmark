# -*- coding: utf-8 -*-
"""Pin the Bluetooth BR/EDR hop-rate citation from the freely readable Core Spec HTML (v5.4) - baseband page."""
import re
import urllib.request

CAND = [
    "https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-54/out/en/br-edr-controller/baseband-specification.html",
    "https://www.bluetooth.com/wp-content/uploads/Files/Specification/HTML/Core-54/out/en/br-edr-controller/radio-specification.html",
]
KEY = re.compile(r"(1\s?600|625\s*[u\u00b5]s|hops?\s*(per|/)\s*second|hop\s*rate|79\s*(RF\s*)?channels?)", re.I)
OUT = r"D:\srtp\EM-benchmark\benchmark_v3\evidence\_bt_standard_hits.txt"

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("Bluetooth Core Spec v5.4 (HTML, bluetooth.com)\n")

for u in CAND:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read().decode("utf-8", "replace")
    except Exception as exc:                                     # noqa: BLE001
        print("fail %s -> %s" % (u, exc))
        continue
    txt = " ".join(re.sub(r"<[^>]+>", " ", raw).split())
    uniq, seen = [], set()
    for m in KEY.finditer(txt):
        s, e = max(0, m.start() - 240), min(len(txt), m.end() + 240)
        sn = txt[s:e]
        k = sn[:90]
        if k in seen:
            continue
        seen.add(k)
        uniq.append((m.group(0), sn))
    print("==== %s  (len=%d, uniq hits=%d)" % (u.split('/')[-1], len(txt), len(uniq)))
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write("\n== %s\n" % u)
        for kw, sn in uniq[:20]:
            fh.write("[%s]  %s\n" % (kw, sn))
    for kw, sn in uniq[:8]:
        print("  [%s] %s" % (kw, sn[:280]))
print("[写入] " + OUT)
