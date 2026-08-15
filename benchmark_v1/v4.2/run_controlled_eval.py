#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遍历 dataset_controlled_lv1 下的所有子文件夹，分别评估每个组合，
最后汇总生成按 (调制方式, 干扰类型) 分组的准确率热力图。
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

# 从 evaluator.py 导入核心函数
from evaluator import evaluate_dataset, OpenAICompatibleAgent

# ==================== 配置 ====================
BASE_DIR = Path("dataset_controlled_lv1")   # 你生成的根目录
MODEL_NAME = "qwen2.5:7b"                   # 你的模型名
OUTPUT_REPORT = "controlled_eval_summary.json"
OUTPUT_HEATMAP = "controlled_eval_heatmap.png"

# 定义 9 种调制和 5 种干扰的顺序（保证热力图规整）
MODULATIONS = ["BPSK", "QPSK", "16QAM", "64QAM", "GFSK", "OOK", "OFDM", "FHSS", "LFM"]
JAMMING_TYPES = ["none", "single_tone", "swept", "pulse", "broadband"]
# =============================================

def run_all_subdirs():
    """遍历所有子文件夹，分别评估，返回汇总结果"""
    # 检查根目录是否存在
    if not BASE_DIR.exists():
        print(f"❌ 目录不存在: {BASE_DIR}")
        print("请先运行 generate_controlled.py 生成数据。")
        return None

    # 初始化 Agent（只创建一次，复用于所有子文件夹）
    print(f"🚀 初始化模型: {MODEL_NAME}")
    agent = OpenAICompatibleAgent(MODEL_NAME)

    # 存储结果的字典：{(mod, jam): {"correct": int, "total": int, "accuracy": float}}
    results = defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})
    all_details = []  # 存储每个样本的详细信息（可选）

    # 获取所有子文件夹（每个子文件夹对应一个组合）
    subdirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    print(f"📁 发现 {len(subdirs)} 个子文件夹，开始评估...")

    # 用 tqdm 显示进度
    for subdir in tqdm(subdirs, desc="评估子文件夹"):
        # 从文件夹名解析出 mod 和 jam，例如 "BPSK_none"
        folder_name = subdir.name
        parts = folder_name.split("_")
        if len(parts) == 2:
            mod, jam = parts
        else:
            # 如果文件夹名不规范，尝试从 metadata 中读取
            try:
                meta_path = subdir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        if meta:
                            mod = meta[0]["ground_truth"]["mod_type"]
                            jam = meta[0]["ground_truth"]["jamming_type"]
                        else:
                            continue
                else:
                    continue
            except:
                continue

        # 调用 evaluator.py 的 evaluate_dataset 函数
        try:
            report = evaluate_dataset(subdir, agent)
        except Exception as e:
            print(f"⚠️ 评估 {folder_name} 失败: {e}")
            continue

        # 提取该组合的准确率
        total = report["num_samples"]
        correct = int(total * report["jamming_classification_accuracy"])
        results[(mod, jam)]["correct"] += correct
        results[(mod, jam)]["total"] += total
        results[(mod, jam)]["accuracy"] = correct / total if total > 0 else 0.0

        # 可选：记录详细结果
        all_details.append({
            "mod": mod,
            "jam": jam,
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "report": report  # 完整报告（可能导致文件很大，可省略）
        })

    print("✅ 评估完成！")
    return results, all_details

def generate_heatmap(results):
    """生成热力图"""
    # 构建数据矩阵
    data = np.zeros((len(MODULATIONS), len(JAMMING_TYPES)))
    for (mod, jam), stats in results.items():
        if mod in MODULATIONS and jam in JAMMING_TYPES:
            i = MODULATIONS.index(mod)
            j = JAMMING_TYPES.index(jam)
            data[i, j] = stats["accuracy"] * 100  # 转为百分比

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="RdYlGn", 
                xticklabels=JAMMING_TYPES, yticklabels=MODULATIONS,
                vmin=0, vmax=100, cbar_kws={"label": "Accuracy (%)"},
                ax=ax, linewidths=0.5, linecolor='white')
    ax.set_title("Accuracy by Modulation Type and Jamming Type (High SNR)", fontsize=16)
    ax.set_xlabel("Jamming Type", fontsize=14)
    ax.set_ylabel("Modulation Type", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP, dpi=300)
    print(f"✅ 热力图已保存: {OUTPUT_HEATMAP}")
    plt.show()

def save_summary(results, all_details):
    """保存汇总报告"""
    summary = {
        "total_combinations": len(results),
        "overall_accuracy": sum(v["correct"] for v in results.values()) / 
                            sum(v["total"] for v in results.values()) if results else 0.0,
        "by_combination": {f"{mod}_{jam}": {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["accuracy"]
        } for (mod, jam), stats in results.items()},
        "per_modulation": {},
        "per_jamming": {}
    }

    # 按调制方式汇总
    for mod in MODULATIONS:
        total_correct = sum(results.get((mod, jam), {}).get("correct", 0) for jam in JAMMING_TYPES)
        total_samples = sum(results.get((mod, jam), {}).get("total", 0) for jam in JAMMING_TYPES)
        summary["per_modulation"][mod] = {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "total": total_samples
        }

    # 按干扰类型汇总
    for jam in JAMMING_TYPES:
        total_correct = sum(results.get((mod, jam), {}).get("correct", 0) for mod in MODULATIONS)
        total_samples = sum(results.get((mod, jam), {}).get("total", 0) for mod in MODULATIONS)
        summary["per_jamming"][jam] = {
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "total": total_samples
        }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ 汇总报告已保存: {OUTPUT_REPORT}")
    print(f"📊 总体准确率: {summary['overall_accuracy'] * 100:.1f}%")

def main():
    # 检查是否设置了环境变量（提醒用户）
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ 未设置 OPENAI_API_KEY，请先运行：")
        print("  $env:OPENAI_API_KEY='ollama'")
        print("  $env:OPENAI_BASE_URL='http://localhost:11434/v1'")
        return

    results, details = run_all_subdirs()
    if results is None:
        return

    generate_heatmap(results)
    save_summary(results, details)

if __name__ == "__main__":
    main()