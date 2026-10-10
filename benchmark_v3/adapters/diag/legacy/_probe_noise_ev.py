import json, os

for lvl in ("d1", "d2", "d3", "d4"):
    p = f"../dataset_{lvl}/{lvl}_00000/capture/private/d1_truth.json"
    if not os.path.exists(p):
        print(lvl, "MISSING")
        continue
    d = json.load(open(p, encoding="utf-8"))
    ev = d.get("interference_events")
    print("---", lvl, "noise =", d.get("noise"), "| events =", len(ev) if ev is not None else None)
    if ev:
        print("   ", json.dumps(ev[0], ensure_ascii=False)[:500])
    else:
        print("    (empty / none)")
