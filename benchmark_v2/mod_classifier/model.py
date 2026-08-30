"""ModCNN：1D-CNN 信号体制分类器（参考 RadioML 系列的入门架构）。"""
from __future__ import annotations

import torch.nn as nn

from data_gen import CLASSES


class ModCNN(nn.Module):
    """输入 (B, 2, 1024)（实部/虚部两通道，RMS 归一化）→ 10 类 softmax。"""

    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        def block(cin, cout, k, pool=True):
            layers = [nn.Conv1d(cin, cout, k, padding=k // 2),
                      nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool1d(2))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(2, 32, 7),
            block(32, 64, 5),
            block(64, 128, 5),
            block(128, 256, 3, pool=False),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
