"""一次性：量化「拒答」的构成——答案文件 vs run-log 原文里的 num_interferers 记号。

分类：
  ok_nonneg  原文里是非负整数（正常）
  neg        原文里是负数（显式 -1 哨兵）
  junk       原文里不是数字（如 III、空、null、字符串）
  no_raw     run-log 里没有该样本（断点续跑只留末轮）
"""
import json
import pathlib
import re

TOK = re.compile(r'"num_interferers"\s*:\s*("[^"]*"|[^,}\s]+)')
NEG = re.compile(r'"num_interferers"\s*:\s*-\d+')
OBJ = re.compile(r'\{\s*"freq_offset_hz"')

for diff in ("d1", "d2", "d3", "d4"):
    root = pathlib.Path(f"benchmark_v3/dataset_{diff}")
    dirs = sorted(p for p in root.glob("answers_*") if p.is_dir())
    for d in dirs:
        files = sorted(d.glob(f"{diff}_*.json"))
        if not files:
            continue
        log = {}
        lp = d / "_run_log.json"
        if lp.exists():
            raw = json.loads(lp.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                for x in raw:
                    if isinstance(x, dict) and x.get("sample_id"):
                        log[x["sample_id"]] = x
        cnt = {"ok_nonneg": 0, "neg": 0, "junk": 0, "no_raw": 0}
        ans_abst = 0
        ans_zero_with_raw_objs = 0
        examples = []
        for p in files:
            sid = p.stem
            o = json.loads(p.read_text(encoding="utf-8"))
            if o.get("abstained"):
                ans_abst += 1
            e = log.get(sid)
            if e is None:
                cnt["no_raw"] += 1
                continue
            t = e.get("raw_output") or ""
            m = TOK.search(t)
            tok = m.group(1) if m else ""
            is_neg = bool(NEG.search(t))
            is_int = bool(re.fullmatch(r"\d+", tok))
            if is_neg:
                cls = "neg"
            elif is_int:
                cls = "ok_nonneg"
            else:
                cls = "junk"
            cnt[cls] += 1
            n_obj = len(OBJ.findall(t))
            if cls in ("neg", "junk") and int(o.get("num_interferers") or 0) == 0 and n_obj:
                ans_zero_with_raw_objs += 1
            if cls in ("neg", "junk") and len(examples) < 3:
                examples.append((sid, tok, n_obj, t[:90]))
        print(f"[{diff}] {d.name}: n={len(files)}  answer.abstained={ans_abst}")
        print(f"      原文记号: ok={cnt['ok_nonneg']} neg={cnt['neg']} junk={cnt['junk']} 无记录={cnt['no_raw']}")
        print(f"      被清成 0 但原文里列了干扰对象: {ans_zero_with_raw_objs}")
        for sid, tok, n_obj, head in examples:
            print(f"        e.g. {sid} token={tok!r} 原文对象数(300字内)={n_obj} :: {head}")
