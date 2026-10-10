"""只读探针：scenario 的发射源结构（preset vs 组件） + v3 辐射体生成器签名。"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
import inspect
import json
import sys
from pathlib import Path

B = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(B / "城市低空电磁信号仿真平台" / "src"))
sys.path.insert(0, str(B / "simulation"))

from em_signal_simulator_v3.emitters import EMITTERS  # noqa: E402

for d in ("d1", "d2", "d3", "d4"):
    p = B / f"dataset_{d}" / f"{d}_00000" / "capture" / "private" / "scenario.json"
    sc = json.loads(p.read_text(encoding="utf-8"))
    print(f"=== {d}  emitters={len(sc['emitters'])}")
    for em in sc["emitters"]:
        comps = em.get("components") or []
        c0 = comps[0] if comps else {}
        print("   id=%-14s role=%-12s preset=%-16s power=%s  ncomp=%d  comp0=%s" % (
            em.get("id"), em.get("role"), em.get("preset"), em.get("power_dbm"), len(comps),
            {k: c0.get(k) for k in ("kind", "bandwidth_hz", "offset_hz",
                                    "pulse_period_s", "duty_cycle")}))

print("\n=== v3 辐射体生成器签名（是否接受 bw / fs）")
for name in sorted(EMITTERS):
    gen = EMITTERS[name][0]
    sig = inspect.signature(gen)
    print("  %-18s bw=%-5s fs=%-5s %s" % (
        name, "bw" in sig.parameters, "fs" in sig.parameters, str(sig)))
