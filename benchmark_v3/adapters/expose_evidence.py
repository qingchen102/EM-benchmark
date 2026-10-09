"""受害者层公开视图落地器：把真值 interference_events 投影进 **Agent 可见**的
capture/public/observations.json（场景设计规范 v1.0 §7）。

只动 public/observations.json —— 不碰 IQ、不碰 private、不碰 index。因此：
  * IQ 逐字节不变（脚本内对 iq_station_a.npy 做前后 MD5 自证）；
  * 答案键（private/d1_truth.json）与评分口径（eval_dataset.py）完全不参与，只读；
  * 幂等：重复运行结果一致；--mechanism 切换暴露粒度（full / coarse / none）。
  * --coop（默认开）同时恢复 public 的 cooperative_sources 公开先验：平台每次 replay
    都会整体重写 observations.json 并把它写成 []（cityrf/runner.py:102），必须在
    replay 之后恢复；两条路径的 _target_bw_normalized 等价性见 _probe_prior.py。

用法：
    python expose_evidence.py ../dataset_d3 ../dataset_d4
    python expose_evidence.py ../dataset_d1 ../dataset_d2 ../dataset_d3 ../dataset_d4 --dry-run
    python expose_evidence.py ../dataset_d4 --mechanism coarse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import evidence_layer as el
import gen_dataset as gd              # 顺带把平台 src 插进 sys.path（digest / 公开先验）
from cityrf.export import digest  # noqa: E402

# audit_dataset.py v2-c（信息隔离）用的子串黑名单，这里同步自查一遍
BANNED_SUBSTRINGS = ("emitter", "intent", "is_jamming", "freq_offset_hz")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _refresh_summary_hash(cap: Path, rel: str) -> bool:
    """改写 public 文件后，把 run_summary.json 里该文件的摘要更新为现值。

    平台 runner 会摘要 run root 下每个文件（含 public/observations.json），
    `regen_noise.refresh_hashes` 同样；此后任何对 public 的合法改写都必须同步摘要，
    否则数据集自带的一致性对账会失败（对账脚本 _check_summary_hashes.py）。
    """
    sp = cap / "run_summary.json"
    if not sp.is_file():
        return False
    summ = json.loads(sp.read_text(encoding="utf-8"))
    hashes = summ.get("hashes") or {}
    if rel not in hashes:
        return False
    cur = digest(cap / rel)
    if hashes.get(rel) == cur:
        return False                      # 摘要已一致：不碰文件（幂等）
    hashes[rel] = cur
    summ["hashes"] = hashes
    summ["public_patch"] = {"tool": "expose_evidence.py", "file": rel}
    sp.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def process(root: Path, mechanism: str, dry_run: bool, coop: bool) -> dict:
    pub_list = sorted(root.glob("*/capture/public/observations.json"))
    st = {"root": root.name, "samples": len(pub_list), "with_events": 0, "events": 0,
          "hashes_refreshed": 0,
          "rewritten": 0, "unchanged": 0, "coop_restored": 0,
          "iq_checked": 0, "iq_unchanged": 0, "problems": [],
          "mechanisms": {}, "fields": {}}
    for op in pub_list:
        sid = op.parent.parent.parent.name
        tp = op.parent.parent / "private" / "d1_truth.json"
        if not tp.exists():
            st["problems"].append(f"{sid}: 缺 private/d1_truth.json")
            continue
        truth = json.loads(tp.read_text(encoding="utf-8"))
        old_txt = op.read_text(encoding="utf-8")
        old_obj = json.loads(old_txt)
        obs = json.loads(old_txt)
        iq = op.parent / "iq_station_a.npy"
        before = _md5(iq) if iq.exists() else None

        n = el.attach_public_events(obs, truth, mechanism)
        if n:
            st["with_events"] += 1
            st["events"] += n
            for e in obs["interference_events"]:
                bad = [k for k in e if k not in el.PUBLIC_EVENT_FIELDS]
                if bad:
                    st["problems"].append(f"{sid}: 公开事件出现非公开字段 {bad}")
                st["fields"][",".join(sorted(e))] = st["fields"].get(",".join(sorted(e)), 0) + 1
                mk = str(e.get("mechanism", "(hidden)"))
                st["mechanisms"][mk] = st["mechanisms"].get(mk, 0) + 1
        if coop:
            obs["cooperative_sources"] = [gd.cooperative_sources_prior(truth)]
            st["coop_restored"] += 1

        txt = json.dumps(obs, ensure_ascii=False)
        hit = [b for b in BANNED_SUBSTRINGS if b in txt]
        if hit:
            st["problems"].append(f"{sid}: observations 命中隔离黑名单 {hit}")
        withheld = [w for w in el.WITHHELD_EVENT_FIELDS if w in txt]
        if withheld:
            st["problems"].append(f"{sid}: observations 出现屏蔽字段 {withheld}")

        # 只在**内容**真的变了才写盘，且写盘格式与平台/现有文件一致（indent=2 + 结尾换行）——
        # 否则"内容没变但整体重排"会把 50 个文件全算成改写。
        changed = (json.dumps(old_obj, ensure_ascii=False, sort_keys=True)
                   != json.dumps(obs, ensure_ascii=False, sort_keys=True))
        if changed:
            st["rewritten"] += 1
            if not dry_run:
                op.write_text(json.dumps(obs, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
        else:
            st["unchanged"] += 1
        # 摘要同步与内容改写解耦：本次内容没变也要把 run_summary.json 里与盘上
        # 不一致的摘要修好（幂等自愈，防止 replay/旧版本遗留失效摘要）。
        if not dry_run and _refresh_summary_hash(op.parent.parent, "public/observations.json"):
            st["hashes_refreshed"] += 1
        if before is not None:
            st["iq_checked"] += 1
            if _md5(iq) == before:
                st["iq_unchanged"] += 1
    return st


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", help="数据集目录（如 ../dataset_d3）")
    ap.add_argument("--mechanism", default="full", choices=list(el.MECHANISM_MODES),
                    help="机理暴露粒度：full（§7 原样）/ coarse（统一 interference）/ none")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写盘")
    ap.add_argument("--coop", dest="coop", action="store_true", default=True,
                    help="同时恢复 public 的 cooperative_sources 公开先验（默认开）")
    ap.add_argument("--no-coop", dest="coop", action="store_false",
                    help="只做受害者层投影，不动 cooperative_sources")
    args = ap.parse_args()

    bad = 0
    for r in args.roots:
        root = Path(r)
        if not (root / "index.json").exists():
            print(f"[skip] {root} 没有 index.json")
            bad += 1
            continue
        st = process(root, args.mechanism, args.dry_run, args.coop)
        print(f"== {st['root']} ==  mechanism={args.mechanism}"
              + ("  (dry-run)" if args.dry_run else ""))
        print(f"  样本 {st['samples']}  含事件 {st['with_events']}  公开事件 {st['events']} 条"
              f"  改写文件 {st['rewritten']}  未变 {st['unchanged']}")
        if st["coop_restored"]:
            print(f"  cooperative_sources 公开先验已恢复 {st['coop_restored']} 个样本")
        if st["hashes_refreshed"]:
            print(f"  run_summary.json 摘要已同步 {st['hashes_refreshed']} 个样本")
        print(f"  IQ 完整性自证：检查 {st['iq_checked']}  未变 {st['iq_unchanged']}")
        if st["mechanisms"]:
            print(f"  机理分布 {st['mechanisms']}")
        if st["fields"]:
            print(f"  公开字段组合 {st['fields']}")
        if st["problems"]:
            bad += 1
            print(f"  !! 问题 {len(st['problems'])} 条：")
            for p in st["problems"][:10]:
                print(f"     - {p}")
        else:
            print("  信息隔离/字段自查：通过（无黑名单命中、无屏蔽字段、无越界字段）")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
