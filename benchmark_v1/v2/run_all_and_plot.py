#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 配置区 ====================
LEVELS = [
    {"name": "Lv1_HighSNR", "dir": "dataset_lv1", "report": "report_lv1.json"},
    {"name": "Lv2_MidSNR",  "dir": "dataset_lv2", "report": "report_lv2.json"},
    {"name": "Lv3_LowSNR",  "dir": "dataset_lv3", "report": "report_lv3.json"},
]

MODEL = "qwen2.5:7b"
PLOT_OUTPUT1 = "accuracy_by_level_and_type.png"
PLOT_OUTPUT2 = "accuracy_trend_by_snr.png"
# =================================================

def evaluate_level(level_info):
    dataset_dir = level_info["dir"]
    report_file = level_info["report"]
    
    if not Path(dataset_dir).exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}")
        return None
    
    print(f"\n{'='*50}")
    print(f"▶ 正在评估 {level_info['name']} (数据集: {dataset_dir})")
    print(f"   报告将保存至: {report_file}")
    print('='*50)
    
    cmd = [
        sys.executable, "evaluator.py",
        dataset_dir,
        "--model", MODEL,
        "--output", report_file
    ]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # 关键修复：指定 encoding='utf-8'
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, encoding='utf-8', env=env)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    
    if process.returncode != 0:
        print(f"⚠️ 评估 {level_info['name']} 失败，返回码: {process.returncode}")
        return None
    
    if not Path(report_file).exists():
        print(f"❌ 报告文件未生成: {report_file}")
        return None
    
    print(f"✅ {level_info['name']} 评估完成")
    return report_file

def load_reports():
    results = {}
    for level_info in LEVELS:
        report_path = level_info["report"]
        if Path(report_path).exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results[level_info["name"]] = data
                print(f"✅ 已加载: {report_path}")
        else:
            print(f"⚠️ 报告文件 {report_path} 不存在")
    return results

def plot_results(all_data):
    if not all_data:
        print("❌ 没有可用的报告数据")
        return
    
    type_names = ["none", "single_tone", "swept", "pulse", "broadband"]
    type_labels = ["无干扰", "单音", "扫频", "脉冲", "宽带"]
    
    acc_by_level_type = {}
    levels = []
    for level_name, data in all_data.items():
        levels.append(level_name)
        stats = {t: [0, 0] for t in type_names}
        for r in data["results"]:
            exp = r["expected_jamming_type"]
            pred = r["predicted_jamming_type"]
            if exp not in stats:
                continue
            stats[exp][1] += 1
            if exp == pred:
                stats[exp][0] += 1
        acc_by_level_type[level_name] = {
            t: (stats[t][0] / stats[t][1] * 100) if stats[t][1] > 0 else 0
            for t in type_names
        }
    
    # 图1：分组柱状图
    x = np.arange(len(type_labels))
    width = 0.25
    multiplier = 0
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {'Lv1_HighSNR': '#2ecc71', 'Lv2_MidSNR': '#f1c40f', 'Lv3_LowSNR': '#e74c3c'}
    
    for level_name in levels:
        offset = width * multiplier
        acc_values = [acc_by_level_type[level_name].get(t, 0) for t in type_names]
        rects = ax.bar(x + offset, acc_values, width, label=level_name, color=colors.get(level_name, '#999'))
        for rect, val in zip(rects, acc_values):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        multiplier += 1
    
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_xlabel('干扰类型', fontsize=12)
    ax.set_title('不同信噪比级别下各类干扰的识别准确率', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width, type_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT1, dpi=300)
    print(f"✅ 图表1已保存: {PLOT_OUTPUT1}")
    
    # 图2：整体准确率趋势折线图
    ordered_levels = ["Lv1_HighSNR", "Lv2_MidSNR", "Lv3_LowSNR"]
    ordered_labels = ["高信噪比 (>5dB)", "中信噪比 (-5~5dB)", "低信噪比 (<-5dB)"]
    ordered_acc = []
    for l in ordered_levels:
        if l in all_data:
            ordered_acc.append(all_data[l].get("jamming_classification_accuracy", 0) * 100)
        else:
            ordered_acc.append(0)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(ordered_labels, ordered_acc, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2980b9')
    ax2.set_ylabel('总体准确率 (%)', fontsize=12)
    ax2.set_xlabel('信噪比区间', fontsize=12)
    ax2.set_title('模型准确率随信噪比变化的趋势', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    for i, acc in enumerate(ordered_acc):
        if acc > 0:
            ax2.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
    ax2.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT2, dpi=300)
    print(f"✅ 图表2已保存: {PLOT_OUTPUT2}")
    
    plt.show()

def main():
    print("🚀 开始批量评估三个 SNR 级别的数据集...")
    print(f"模型: {MODEL}")
    print(f"数据集: {[l['dir'] for l in LEVELS]}")
    
    for level in LEVELS:
        evaluate_level(level)
    
    print("\n📊 加载评估报告...")
    all_data = load_reports()
    if not all_data:
        print("❌ 没有可用的报告")
        return
    
    print("\n🎨 生成可视化图表...")
    plot_results(all_data)
    
    print("\n✨ 全部完成！")

if __name__ == "__main__":
    main()