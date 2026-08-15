# generate_controlled.py
import sys
import json
from pathlib import Path
from simulation.em_signal_simulator.factory import generate_dataset

# 配置
OUTPUT_DIR = "dataset_controlled_lv1"
COUNT_PER_COMBO = 10
SNR_RANGE = (5, 20)  # 高信噪比

# 所有组合
MODULATIONS = ["BPSK", "QPSK", "16QAM", "64QAM", "GFSK", "OOK", "OFDM", "FHSS", "LFM"]
JAMMING_TYPES = ["none", "single_tone", "swept", "pulse", "broadband"]

# 创建总输出目录
Path(OUTPUT_DIR).mkdir(exist_ok=True)
all_metadata = []

# 循环生成
for mod in MODULATIONS:
    for jam in JAMMING_TYPES:
        print(f"Generating {mod} + {jam} ...")
        # 每个组合单独生成，存到子目录，避免覆盖
        sub_dir = f"{OUTPUT_DIR}/{mod}_{jam}"
        meta = generate_dataset(
            output_dir=sub_dir,
            count=COUNT_PER_COMBO,
            mixed=False,              # 关闭混合，精确指定类型
            snr_range=SNR_RANGE,
            mod_type=mod,             # 指定调制
            jamming_type=jam          # 指定干扰
        )
        # 把生成的 metadata 合并到总列表
        all_metadata.extend(meta)

# 保存合并后的总 metadata（方便 evaluator.py 读取）
with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"✅ 生成完成！共 {len(all_metadata)} 个样本，保存在 {OUTPUT_DIR}")