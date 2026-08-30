# mod_classifier —— 学习型信号体制分类器（v7 候选）

用 1D-CNN 直接吃 IQ 波形，替换 `estimate_modulation_features` 的
"手工特征 + 最近模板"后端（该范式经 oracle 实测干净上限仅 0.66，见
`../mod_upperbound_v6.log`）。

## 纪律（先于一切）

1. **训练/考试隔离**：训练数据由独立生成配置产生——种子段 500000+（考卷数据集与
   上限脚本使用 1000~ 数千段，绝不重叠）、带宽采样 U(0.02, 0.90)（考卷干扰为
   0.05~0.60）、SNR 增强 U(-10, 20) dB（考卷为 -5~15）。冻结的 500 样本不进训练。
2. **预注册验收门槛**（`eval_oracle.py`，跑之前已写死）：
   - 第 1 级 干净波形上限（独立配置生成）≥ **0.85**
   - 第 2 级 冻结 500 的 GT 干扰切片 ≥ **0.50**
   - 过闸才接入 `estimate_modulation_features`（v7）；不过闸则留档不接入。

## 文件

- `data_gen.py`   样本生成（类名 → 波形，含频偏/噪声/时移增强）
- `model.py`      ModCNN（4 段 Conv1d-BN-ReLU-Pool + 全局平均 + FC，约 0.2M 参数）
- `train.py`      训练（干净波形预生成 + 每 epoch 新鲜加噪/频偏/时移增强）
- `eval_oracle.py` 两级 oracle 验收（干净级 / 冻结集切片级），输出混淆矩阵与按 INR 分解

## 用法

```bash
python train.py --epochs 8 --steps-per-epoch 200 --batch 256   # 训练（CPU 约 20-40 分钟）
python eval_oracle.py --checkpoint checkpoint.pt               # 两级验收
```
