"""反拒答增强的口径单测（只读，零 API）。

检查三件事：
  1. ev2._has_final_answer 已被换成 v3 列表校验器（v2 冻结文件零改动）；
  2. 测得候选时"空列表"判无效 → 会触发 repair 轮；没测到候选时空列表合法（D1 干净样本不受影响）；
  3. _recording_tools 包住的工具会把输出记进 _EVIDENCE。

用法：python -X utf8 _check_antirefuse.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_capture_agent as rca  # noqa: E402

IT = {"freq_offset_hz": 1.5e5, "kind": "Zigbee", "intent": "incidental"}

bad = 0


def check(name, got, exp):
    global bad
    ok = got == exp
    if not ok:
        bad += 1
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  expected=%r got=%r" % (exp, got)))


check("patched_validator", rca.ev2._has_final_answer is rca.has_final_answer_capture, True)

rca._reset_evidence()
check("empty_ok_when_no_candidate", rca.has_final_answer_capture({"interferers": []}), True)

rca._EVIDENCE["candidates"] = 5
check("empty_invalid_when_candidates", rca.has_final_answer_capture({"interferers": []}), False)
check("nonempty_ok", rca.has_final_answer_capture({"interferers": [IT]}), True)
check("count_only_invalid", rca.has_final_answer_capture({"num_interferers": 2}), False)
check("not_dict_invalid", rca.has_final_answer_capture("nope"), False)

rca._reset_evidence()
check("reset_clears", rca._EVIDENCE["candidates"], 0)

calls = []


def _fake_clustering():
    return {"n_candidates": 3, "candidates": [{"a": 1}, {"b": 2}, {"c": 3}], "sources": [{"a": 1}]}


def _fake_classify(station="station_a"):
    calls.append(station)
    return {"num_interferers": 1, "interferers": [{"freq_offset_hz": 1.0, "kind": "DECT"}]}


wrapped = rca._recording_tools({"estimate_sources_by_spatial_clustering": _fake_clustering,
                                "classify_extracted_waveform": _fake_classify})
out = wrapped["estimate_sources_by_spatial_clustering"]()
check("recorder_candidates", rca._EVIDENCE["candidates"], 3)
check("recorder_passthrough", out["n_candidates"], 3)
wrapped["classify_extracted_waveform"]("station_b")
check("recorder_classify", (rca._EVIDENCE["classify"] or {}).get("num_interferers"), 1)
check("recorder_kwargs_passthrough", calls, ["station_b"])

print("\n%s" % ("%d failed" % bad if bad else "all passed"))
sys.exit(1 if bad else 0)
