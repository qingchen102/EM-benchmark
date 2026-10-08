# 交付验证 · 2026-09-18

## 代码与数据迁移

- 从原 Agent 工程复制 172 个代码、配置、测试、几何及示例产物文件（复制清单见 `import_manifest.json`）。另复制两张多径方法卡，增加独立 WebUI、CLI 入口与文档。
- `src/cityrf` 仿真核心及算法模块与原复制哈希一致。独立化调整仅涉及部分测试和辅助脚本，记录在 `migration_changes.json`。
- 9 个预设配置全部通过 `Scenario` 校验。
- 三个导入案例全部通过 IQ 形状、数据类型、SHA256 和元数据审计。

## 自动检查

- Windows：`python -m pytest -q --junitxml=docs/test-results.xml`，437 passed、7 skipped；2 条旧定位 API 的弃用警告。跳过项为依赖原 Agent/原旧项目的集成测试及显式关闭的 RT 测试。
- 后续修改作业状态原子写入后，单独重跑 Web API 测试：3 passed。包含真实自由空间生成、PSD 主峰验证、产物下载、无效配置和跨源写入拒绝。
- WSL：`RUN_RT_TESTS=1 python -m pytest -q tests/test_rt.py`，4 passed；输出见 `test-rt-results.xml`。
- 独立命令行入口 `python simulate.py validate --config examples/free_space.json` 通过。

## 浏览器端实际验证

- 真实慕尼黑网格正常加载，38,936 三角面；2D / 3D 切换、设备选择、标签、阵面方向显示正常。
- 2D 拖动 target_1 后，属性中的位置从 (-150,-100,85) 更新至约 (-129.8,-87.6,85)，高度保持不变；随后切换预设恢复配置。
- 十字阵切换成 4×4 矩形阵，并同步三站，后端校验通过。
- 点击网页“运行仿真”实际执行 `run_20260918_153102_fdd934`：Sionna RT LLVM CPU，3 源/3 站/16 通道/50 ms，22 条路径，仿真核心约 2.9 s；含进程启动、审计和出图约 8.1 s。
- 新作业的 IQ、元数据和哈希通过独立审计，日志完整到 `COMPLETE`。浏览器显示三站频谱与时频图，非伪造前端曲线。
- 地图和图表在浏览器截图中检查过；检查时没有浏览器 console error/warn。

## 当前边界

- GPU RT 未验收；仅 CPU RT 与基础 DSP 已验证。
- 场景仍为静态、最长50 ms；没有新增动态轨迹或流式处理。
- 地图不展示真实反射折线，因为原信道输出没有逐跳顶点；显示的连线明确标记为布站配对。
- 本地单用户服务，每次运行一个仿真；暂未实现作业取消与服务重启后的自动续跑。
