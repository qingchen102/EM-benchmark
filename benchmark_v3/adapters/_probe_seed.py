"""只读：dump 一个发射源/一条真值源的全部字段，找有没有可复现的随机种子。"""
import json
from pathlib import Path

B = Path(__file__).resolve().parent.parent
cap = B / "dataset_d4" / "d4_00000" / "capture"
sc = json.loads((cap / "private" / "scenario.json").read_text(encoding="utf-8"))
print("scenario 顶层键:", list(sc.keys()))
print("scenario.seed =", sc.get("seed"))
for em in sc["emitters"]:
    print(f"--- emitter {em.get('id')} 全部键: {sorted(em.keys())}")
    if em.get("id") == "interferer_1":
        print("    ", json.dumps({k: v for k, v in em.items() if k != "components"},
                                 ensure_ascii=False)[:400])
        print("     components:", json.dumps(em["components"], ensure_ascii=False)[:400])

tr = json.loads((cap / "private" / "truth.json").read_text(encoding="utf-8"))
print("\ntruth 顶层键:", list(tr.keys()))
s0 = tr["stations"][0]
for s in s0["sources"]:
    if s["emitter_id"] == "interferer_1":
        print("source 全部键:", sorted(s.keys()))
        print("  geometry:", json.dumps(s.get("geometry"), ensure_ascii=False)[:300])
        print("  channel_provenance:", json.dumps(s.get("channel_provenance"), ensure_ascii=False)[:500])
print("\ntruth.notes:", json.dumps(tr.get("notes"), ensure_ascii=False)[:400])
