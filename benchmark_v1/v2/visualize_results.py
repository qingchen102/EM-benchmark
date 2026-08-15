#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯英文版可视化：读取三个级别的报告，绘制准确率柱状图和趋势图。
要求：report_lv1.json, report_lv2.json, report_lv3.json 存在。
输出：accuracy_by_type_and_snr.png, accuracy_trend.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 定义级别顺序和对应的显示名称
LEVELS = [
    {"file": "report_lv1.json", "label": "High SNR", "short": "High"},
    {"file": "report_lv2.json", "label": "Mid SNR",  "short": "Mid"},
    {"file": "report_lv3.json", "label": "Low SNR",  "short": "Low"},
]

# 干扰类型（按顺序）
JAMMING_TYPES = ["none", "single_tone", "swept", "pulse", "broadband"]
TYPE_LABELS = ["None", "Single-tone", "Swept", "Pulse", "Broadband"]

def load_report(filepath):
    """加载 JSON 报告，如果不存在则返回 None"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 未找到文件: {filepath}")
        return None

def compute_accuracy_by_type(report):
    """从报告结果中计算每类干扰的准确率，返回字典 {type: accuracy}"""
    if report is None:
        return {}
    results = report.get("results", [])
    stats = {t: [0, 0] for t in JAMMING_TYPES}  # [correct, total]
    for r in results:
        exp = r.get("expected_jamming_type")
        pred = r.get("predicted_jamming_type")
        if exp not in stats:
            continue
        stats[exp][1] += 1
        if exp == pred:
            stats[exp][0] += 1
    acc = {}
    for t in JAMMING_TYPES:
        correct, total = stats[t]
        acc[t] = (correct / total * 100) if total > 0 else 0.0
    return acc

def plot_grouped_bar(all_data):
    """绘制分组柱状图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(TYPE_LABELS))
    width = 0.25
    colors = {'High': '#2ecc71', 'Mid': '#f1c40f', 'Low': '#e74c3c'}
    
    for i, level in enumerate(LEVELS):
        acc_dict = all_data.get(level["short"], {})
        acc_values = [acc_dict.get(t, 0.0) for t in JAMMING_TYPES]
        offset = width * i
        rects = ax.bar(x + offset, acc_values, width, label=level["label"], color=colors[level["short"]])
        # 在柱顶显示数值
        for rect, val in zip(rects, acc_values):
            if val > 0:
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Jamming Type', fontsize=12)
    ax.set_title('Accuracy by Jamming Type and SNR Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(TYPE_LABELS)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_by_type_and_snr.png', dpi=300)
    print("✅ 柱状图已保存: accuracy_by_type_and_snr.png")
    plt.show()

def plot_trend(all_data):
    """绘制总体准确率趋势折线图"""
    ordered = ["High", "Mid", "Low"]
    labels = ["High SNR", "Mid SNR", "Low SNR"]
    accuracies = []
    for key in ordered:
        data = all_data.get(key, {})
        # 计算总体准确率：从该级别的报告中读取
        # 我们通过计算平均准确率或者直接从报告中的 jamming_classification_accuracy 字段获取
        # 但报告中有总体准确率字段，可以直接使用
        # 然而我们的 all_data 只存了每个类型的准确率，所以需要重新加载报告获取总体准确率
        # 简单起见，我们在这里重新加载报告获取总体准确率
        # 或者遍历所有结果重新计算，但更简单是直接从报告的 'jamming_classification_accuracy' 取
        # 我们修改一下，在加载时保存总体准确率
        pass
    # 为了简单，我们在这里重新加载报告并提取总体准确率
    overall = []
    for level in LEVELS:
        report = load_report(level["file"])
        if report:
            overall.append(report.get("jamming_classification_accuracy", 0) * 100)
        else:
            overall.append(0.0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, overall, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2980b9')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax.set_xlabel('SNR Level', fontsize=12)
    ax.set_title('Overall Accuracy Trend vs SNR Level', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    for i, acc in enumerate(overall):
        if acc > 0:
            ax.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('accuracy_trend.png', dpi=300)
    print("✅ 趋势图已保存: accuracy_trend.png")
    plt.show()

def main():
    # 加载所有报告，并计算每类准确率
    all_data = {}
    for level in LEVELS:
        report = load_report(level["file"])
        if report:
            acc_by_type = compute_accuracy_by_type(report)
            all_data[level["short"]] = acc_by_type
            # 同时也保存总体准确率以便趋势图使用
            all_data[level["short"] + "_overall"] = report.get("jamming_classification_accuracy", 0) * 100
        else:
            all_data[level["short"]] = {t: 0.0 for t in JAMMING_TYPES}
            all_data[level["short"] + "_overall"] = 0.0
    
    if not any(all_data.values()):
        print("❌ 没有可用的报告数据，请确保 report_lv1/2/3.json 存在。")
        return
    
    print("📊 绘制分组柱状图...")
    plot_grouped_bar(all_data)
    
    print("📈 绘制趋势折线图...")
    # 趋势图直接使用存储的总体准确率
    # 为了更清晰，重新提取总体准确率
    # 但我们已在 all_data 中存储了 overall
    # 只需提取 labels 和 acc
    ordered = ["High", "Mid", "Low"]
    labels = ["High SNR", "Mid SNR", "Low SNR"]
    overall_acc = [all_data.get(k + "_overall", 0.0) for k in ordered]
    # 重新绘制趋势图（可复用上面的函数，但这里直接画）
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, overall_acc, marker='o', linestyle='-', linewidth=2, markersize=8, color='#2980b9')
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax.set_xlabel('SNR Level', fontsize=12)
    ax.set_title('Overall Accuracy Trend vs SNR Level', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    for i, acc in enumerate(overall_acc):
        if acc > 0:
            ax.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('accuracy_trend.png', dpi=300)
    print("✅ 趋势图已保存: accuracy_trend.png")
    plt.show()

if __name__ == "__main__":
    main()