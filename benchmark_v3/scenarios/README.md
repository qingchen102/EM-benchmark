# 场景运行入口

本目录集中放置"按场景跑"的脚本。数据生成、冒烟、批量重跑都从这里进入，不再散落在 `adapters/` 或临时目录里。

## 前置条件

- Linux / WSL2（Sionna + Mitsuba 依赖 LLVM，原生 Windows 跑不了）
- 已准备平台虚拟环境：`benchmark_v3/城市低空电磁信号仿真平台/.venv-rt/`
- 在仓库根目录执行；脚本会自行解析路径

## 单个场景

```bash
bash benchmark_v3/scenarios/run_dataset.sh <scene> [want] [seed0] [pool] [out]
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `scene` | `d1`–`d12` | 必填 |
| `want` | 最终接受的样本数 | `50` |
| `seed0` | 起始随机种子 | `31000` |
| `pool` | 候选池大小 | 等于 `want`；约束强的场景要放大 |
| `out` | 输出目录 | `benchmark_v3/dataset_<scene>` |

可选环境变量：`PY=/path/to/python` 指定解释器；`DURATION=1.0` 覆盖窗长（长窗冒烟用）。

示例：

```bash
# 常规场景
bash benchmark_v3/scenarios/run_dataset.sh d5 50 51000

# d7 蜂群同频约束强：200 个候选里筛 50 个
bash benchmark_v3/scenarios/run_dataset.sh d7 50 53000 200
```

## 批量重跑 d5–d12

```bash
bash benchmark_v3/scenarios/run_all_d5_d12.sh
```

固定种子（与已入库数据集一致）：

| 场景 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 |
|---|---|---|---|---|---|---|---|---|
| seed0 | 51000 | 52000 | 53000 | 54000 | 55000 | 56000 | 57000 | 58000 |
| 候选池 | 50 | 50 | 200 | 50 | 120 | 120 | 120 | 120 |

每档最终接受 50 个样本；输出到 `benchmark_v3/dataset_d5` … `dataset_d12`。

## 冒烟

```bash
# d1–d12 每个只生成 2 个样本，验证链路
bash benchmark_v3/scenarios/smoke_all.sh

# 只冒烟某几个
bash benchmark_v3/scenarios/smoke_all.sh d7 d11
```

输出到 `benchmark_v3/_regen_scene_smoke/`，不写正式数据集。

## 产物与约定

- 数据集目录已被 `.gitignore` 排除，不提交原始 IQ；只提交 `manifest.jsonl`、`labels.json`、`README.md` 等元数据时，需单独确认体积。
- 生成器入口仍是 `benchmark_v3/adapters/gen_dataset.py`；本目录只负责场景参数、种子和输出位置，不改生成逻辑。
- 新增场景时，在 `run_all_d5_d12.sh` 的 `SEED` 表里登记种子，并在 `docs/场景设计规范.md` 补场景卡。
