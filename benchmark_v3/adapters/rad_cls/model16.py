"""RadCNN：16 辐射体体制分类器（复用 v8 ModCNN 架构，输入 (B,2,L) 长度无关）。"""
from __future__ import annotations

import torch.nn as nn

from data_gen16 import CLASSES16


class RadCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES16)):
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
