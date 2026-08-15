"""v2 数据集生成命令行入口。

用法示例：
    # 全随机模式（默认）
    python simulation/generate_dataset_v2.py --count 100

    # 指定部分参数，其余随机
    python simulation/generate_dataset_v2.py --count 50 --modulation QPSK --num-sources 3

    # 全指定模式
    python simulation/generate_dataset_v2.py --count 30 --modulation OFDM --num-sources 2 --interferer-type pulse --fixed-snr 10
"""
import argparse
import random
from em_signal_simulator.factory import generate_dataset

MODULATION_LIST = ["BPSK", "QPSK", "16QAM", "64QAM", "GFSK", "OOK", "OFDM", "FHSS", "LFM"]
INTERFERER_TYPE_LIST = ["single_tone", "swept", "pulse", "broadband", "nfm"]


def parse_num_sources(value):
    if value == "random":
        return "random"
    try:
        v = int(value)
        if v in (1, 2, 3):
            return v
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"num-sources must be 1, 2, 3, or 'random'")


def main():
    p = argparse.ArgumentParser(description="Generate multi-source multi-antenna IQ benchmark samples (v2)")
    p.add_argument("--output-dir", default="dataset")
    p.add_argument("--count", type=int, default=100)
    p.add_argument(
        "--num-sources",
        type=parse_num_sources,
        default="random",
        help="总源数（目标+干扰）：1/2/3 或 random（随机）"
    )
    p.add_argument(
        "--modulation",
        default="random",
        help=f"目标调制类型：{'/'.join(MODULATION_LIST)} 或 random（随机）"
    )
    p.add_argument(
        "--interferer-type",
        default="random",
        help=f"干扰波形类型：{'/'.join(INTERFERER_TYPE_LIST)} 或 random（随机）"
    )
    snr_group = p.add_mutually_exclusive_group()
    snr_group.add_argument("--snr-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), help="每个样本从该区间均匀采样 SNR")
    snr_group.add_argument("--fixed-snr", type=float, default=None, help="所有样本使用固定 SNR")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-antennas", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=1024)
    # —— 可选物理效应（默认关闭，向后兼容）——
    p.add_argument("--lna-saturation-db", type=float, default=6.0, help="LNA 饱和门限（相对目标 RMS，dB）；有 blocking 干扰时启用 Rapp 压缩")
    p.add_argument("--lna-p", type=float, default=2.0, help="Rapp 模型平滑系数 p")
    p.add_argument("--array-gain-error-std", type=float, default=0.0, help="阵元增益误差标准差（如 0.05 = 5%）")
    p.add_argument("--array-phase-error-std-deg", type=float, default=0.0, help="阵元相位误差标准差（度，如 5.0）")
    # —— 带宽与 blocking 阈值 ——
    p.add_argument("--target-bandwidth", type=float, default=None, help="目标信号占用带宽（归一化）；默认按调制类型标准带宽")
    p.add_argument("--interferer-bandwidth-range", nargs=2, type=float, default=(0.05, 0.6),
                   metavar=("MIN", "MAX"), help="真实调制干扰的带宽采样范围（归一化，带宽为待估计参数）")
    p.add_argument("--blocking-power-threshold-db", type=float, default=10.0, help="blocking 判定功率阈值（dB，相对目标）")
    p.add_argument("--blocking-ratio-threshold", type=float, default=0.5, help="blocking 判定频偏比阈值（|freq_offset|/目标带宽）")
    p.add_argument("--min-inr-db", type=float, default=3.0, help="干扰可检测性下限：INR = 功率比+SNR >= 该值（低于下限的干扰不可检测，无意义）")
    p.add_argument("--num-workers", type=int, default=1, help="并行生成进程数（>1 时用进程池，结果与串行一致）")
    a = p.parse_args()

    if a.snr_range is None and a.fixed_snr is None:
        snr_range = (-5.0, 15.0)
        fixed_snr = None
    else:
        snr_range = a.snr_range
        fixed_snr = a.fixed_snr

    generate_dataset(
        output_dir=a.output_dir,
        count=a.count,
        num_sources=a.num_sources,
        snr_range=snr_range,
        fixed_snr=fixed_snr,
        seed=a.seed,
        modulation=a.modulation,
        interferer_type=a.interferer_type,
        num_antennas=a.num_antennas,
        num_samples=a.num_samples,
        lna_saturation_db=a.lna_saturation_db,
        lna_p=a.lna_p,
        array_gain_error_std=a.array_gain_error_std,
        array_phase_error_std_deg=a.array_phase_error_std_deg,
        target_bandwidth_normalized=a.target_bandwidth,
        interferer_bandwidth_range=tuple(a.interferer_bandwidth_range),
        blocking_power_threshold_db=a.blocking_power_threshold_db,
        blocking_ratio_threshold=a.blocking_ratio_threshold,
        min_inr_db=a.min_inr_db,
        num_workers=a.num_workers,
    )
    print(f"Generated {a.count} samples -> {a.output_dir}")


if __name__ == "__main__":
    main()