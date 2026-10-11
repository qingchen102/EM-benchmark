"""_sanitize 的口径单测（只读，零 API）。

覆盖 2026-10-10 拒答修复的三类形态：
  1. 计数合法            -> n = 计数
  2. 计数坏 + 有条目      -> n = len(条目)，count_field_invalid=True（旧代码会清成 0）
  3. 计数坏 + 空条目      -> abstained=True（真拒答）
另外检查 confidence 透传、机理字段（mechanism）四值域透传/丢弃，与 schema 字段完整性。

用法：python -X utf8 _check_sanitize.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # adapters
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_capture_agent as rca  # noqa: E402

IT = {"freq_offset_hz": 1.5e5, "kind": "Zigbee", "intent": "incidental",
      "is_spoof": False, "doa_deg": 35.0, "position_m": [-650.0, -1410.0]}
IT2 = {"freq_offset_hz": -8.274e5, "kind": "Tone_jam", "intent": "malicious",
       "is_spoof": False, "doa_deg": 12.0, "position_m": [10.0, 20.0]}

CASES = [
    # name, input, expected subset
    ("count_ok_1", {"num_interferers": 1, "interferers": [IT]},
     {"num_interferers": 1, "abstained": False, "count_field_invalid": False}),
    ("count_ok_0", {"num_interferers": 0, "interferers": []},
     {"num_interferers": 0, "abstained": False, "count_field_invalid": False}),
    ("count_ok_low_conf", {"num_interferers": 2, "confidence": "LOW", "interferers": [IT, IT2]},
     {"num_interferers": 2, "abstained": False, "count_field_invalid": False, "confidence": "low"}),
    ("neg_sentinel_empty", {"num_interferers": -1, "interferers": []},
     {"num_interferers": 0, "abstained": False, "count_field_invalid": False}),
    ("roman_with_items", {"num_interferers": "III", "interferers": [IT, IT2]},
     {"num_interferers": 2, "abstained": False, "count_field_invalid": True}),
    ("neg_with_items", {"num_interferers": -1, "interferers": [IT]},
     {"num_interferers": 1, "abstained": False, "count_field_invalid": True}),
    # 2026-10-10 新契约：答案不再要求计数字段（harness 从列表派生），缺计数是正常形态
    ("list_only_items", {"interferers": [IT, IT2]},
     {"num_interferers": 2, "abstained": False, "count_field_invalid": False}),
    ("list_only_five", {"interferers": [IT, IT2, IT, IT2, IT]},
     {"num_interferers": 5, "abstained": False, "count_field_invalid": False}),
    ("empty_pred", {},
     {"num_interferers": 0, "abstained": False, "count_field_invalid": False}),
    # 显式 null 视同“没写计数”（不标 cfi，条目数胜出）
    ("null_count_with_items", {"num_interferers": None, "interferers": [IT]},
     {"num_interferers": 1, "abstained": False, "count_field_invalid": False}),
    # 计数与条目不一致：以**列表**为准（n = len(items)），保证落盘答案自洽
    ("count_lt_items", {"num_interferers": 1, "interferers": [IT, IT2]},
     {"num_interferers": 2, "abstained": False, "count_field_invalid": False}),
    # 2026-10-11 拒答语义：空列表本身是合法答案（确实测到零）；
    # 只有「工具已给出候选、模型却交空列表」才算真拒答 —— 第 4 个元素 = 模拟 _EVIDENCE["candidates"]
    ("refusal_when_candidates_measured", {"interferers": []},
     {"num_interferers": 0, "abstained": True, "count_field_invalid": False}, 3),
    ("legit_empty_no_candidates", {"interferers": []},
     {"num_interferers": 0, "abstained": False, "count_field_invalid": False}, 0),
]

REQUIRED_KEYS = ("num_interferers", "interferers", "abstained", "count_field_invalid")


def main() -> int:
    bad = 0
    total = len(CASES)
    for case in CASES:
        name, pred, exp = case[:3]
        cand = case[3] if len(case) > 3 else 0
        rca._reset_evidence()
        rca._EVIDENCE["candidates"] = cand
        out = rca._sanitize(dict(pred))
        miss_keys = [k for k in REQUIRED_KEYS if k not in out]
        diff = {k: (exp[k], out.get(k)) for k in exp if out.get(k) != exp[k]}
        ok = not miss_keys and not diff
        if not ok:
            bad += 1
        print(("PASS " if ok else "FAIL ") + name)
        if miss_keys:
            print("      missing keys: %s" % miss_keys)
        if diff:
            print("      expected->got: %s" % diff)
        print("      out keys=%s n=%r abstained=%r cfi=%r conf=%r items=%d"
              % (sorted(out), out.get("num_interferers"), out.get("abstained"),
                 out.get("count_field_invalid"), out.get("confidence"),
                 len(out.get("interferers") or [])))
    # 机理字段（2026-10-11 新增轴）：四值域内保留，域外/缺失一律丢弃（不替模型猜）
    for mech, want in (("co_channel", "co_channel"), ("ADJACENT", "adjacent"),
                       ("jam", None), (None, None)):
        total += 1
        item = dict(IT)
        if mech is not None:
            item["mechanism"] = mech
        rca._reset_evidence()
        out = rca._sanitize({"interferers": [item]})
        got = out["interferers"][0].get("mechanism") if out["interferers"] else None
        ok = got == want
        if not ok:
            bad += 1
        print(("PASS " if ok else "FAIL ") + "mechanism_%s" % (mech,))
        print("      expect=%r got=%r" % (want, got))
    print("\n%d/%d passed" % (total - bad, total))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
