# -*- coding: utf-8 -*-
"""Pin down the DECT frame structure citation from the freely available ETSI EN 300 175-3 PDF."""
import re
import sys
import urllib.request

CAND = [
    "https://www.etsi.org/deliver/etsi_en/300100_300199/30017503/02.09.01_60/en_30017503v020901p.pdf",
    "https://www.etsi.org/deliver/etsi_en/300100_300199/30017503/02.08.01_60/en_30017503v020801p.pdf",
    "https://www.etsi.org/deliver/etsi_en/300100_300199/30017503/02.07.01_60/en_30017503v020701p.pdf",
    "https://www.etsi.org/deliver/etsi_en/300100_300199/30017503/02.06.01_60/en_30017503v020601p.pdf",
]
OUT_PDF = r"D:\srtp\EM-benchmark\_tmp_dect.pdf"
KEY = r"(10\s*ms|24\s*(time\s*)?slots?|frame\s+length|frame\s+duration|slot\s+duration|416)"

url_ok = None
for u in CAND:
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if data[:4] == b"%PDF" and len(data) > 50_000:
            with open(OUT_PDF, "wb") as f:
                f.write(data)
            url_ok = u
            print("downloaded %d bytes from %s" % (len(data), u))
            break
        print("skip (not a PDF): %s  head=%r" % (u, data[:40]))
    except Exception as exc:                                     # noqa: BLE001
        print("fail %s -> %s" % (u, exc))

if not url_ok:
    sys.exit("no ETSI PDF available")

import pymupdf                                                    # noqa: E402

doc = pymupdf.open(OUT_PDF)
print("pages =", doc.page_count)
hits = []
for i, page in enumerate(doc):
    txt = page.get_text()
    if not txt:
        continue
    for m in re.finditer(KEY, txt, flags=re.IGNORECASE):
        s = max(0, m.start() - 160)
        e = min(len(txt), m.end() + 160)
        snippet = " ".join(txt[s:e].split())
        hits.append((i + 1, m.group(0), snippet))
    if len(hits) > 60:
        break

seen = set()
lines = ["ETSI EN 300 175-3 (%s)" % url_ok, ""]
for pg, kw, sn in hits:
    k = (kw.lower(), sn[:80])
    if k in seen:
        continue
    seen.add(k)
    lines.append("p%d  [%s]  %s" % (pg, kw, sn))
    if len(lines) > 40:
        break

open(r"D:\srtp\EM-benchmark\benchmark_v3\evidence\_dect_standard_hits.txt", "w", encoding="utf-8").write(
    "\n".join(lines) + "\n")
print("\n".join(lines[:24]))
print("\n[写入] benchmark_v3/evidence/_dect_standard_hits.txt")
