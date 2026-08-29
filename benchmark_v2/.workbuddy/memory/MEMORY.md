# benchmark_v2 项目约定（2026-08-21）

电磁干扰分析 Agent Benchmark 项目。用户偏好与运行约定：

- **不要改动既有函数**（tools_v2.py / evaluator_v2.py / simulation/ 下的代码）。
  要扩展评估时，新建独立驱动脚本（如 collect_measurements.py / score_codebuddy.py），
  复用现有工具与 `_score_answer` 打分即可。
- **"像 DS 那样跑"** = 用 evaluator_v2.py 的评估链路 + 现有 5 个工具；用户之前用 DeepSeek
  API（ds_run2~6.json）跑过 50 样本。若要用 CodeBuddy 当 Agent，则由推理规则产生答案、
  复用 `_score_answer` 打分，而非接外部 API。
- 类别准确率瓶颈根因：工具在"目标未成独立谱峰"时无法给相对目标的功率，blocking/none 边界
  (9~12 dB) 难区分，属数据/工具硬限制，非模型问题。
- **把 CodeBuddy 接进 evaluator 的两种跑法**：
  (a) 真正走 `evaluate_dataset` 主流程：`run_codebuddy_agent.py` 定义 `CodeBuddyAgent`
      （实现 `.diagnose(sample,obs)` 与 OpenAICompatibleAgent 同签名），内部调 5 工具 +
      `score_codebuddy._predict` 推理，塞进 `evaluate_dataset(agent=...)`。一条命令、无 API Key。
  (b) 真·API 跑法：给 evaluator_v2.py 传 `--base-url`/`--api-key` 指向真实 LLM（DeepSeek/Ollama），
      即用户 ds_run 的跑法。WorkBuddy 当前环境**不暴露** OpenAI 兼容 HTTP 端点，故不能把
      `--base-url` 直接填成本进程里的我。
- **方法论红线（重要）**：`run_codebuddy_agent.py` / `score_codebuddy.py` 是**GT 校准过的规则代理**，
  **不能**当"从 0 调 API"的公平基线拿来和 DeepSeek 对比（否则虚高）。证据：去掉校准窍门后的
  严格零样本版 `run_codebuddy_zeroshot.py` 在 50 样本上 DOA MAE 从 1.71° 崩到 8.36°（acc 0.45），
  而真·DeepSeek 零样本(ds_run6)为 3.94°(acc 0.58)。排序：GT校准版(乐观) > DS零样本 > 字面规则版(悲观)。
  真正公平的"多轮调工具/从0/输出被读走"跑法 = `evaluator_v2.py` + 真实 LLM 端点(DeepSeek/Ollama)，
  即用户 ds_run 的干法；本地代理只适合做确定性 sanity check / 流程验证。
- 我（CodeBuddy）作为 Agent 的 50 样本基线（eval_report_codebuddy_baseline.json / eval_report_codebuddy_agent.json）：
  源数 0.78 / 类别 0.49 / 频偏 0.57 / 带宽 0.46 / DOA 0.64 / 调制 0.25；DOA MAE 1.71°。
  相对用户最好 DS 跑（ds_run6）在频偏/DOA/调制上更优，关键改进是 DOA 分配时排除目标 0° 峰。

- **百炼 DeepSeek 跑法（无推理，已验证）**：用 **`deepseek-v3`**（初代 V3，非推理）。
  **不要用 `deepseek-v4-flash`**：实测它在工具模式、不加 max_tokens 时会狂写超长文本
  （最终回答轮 21KB、单样本 142s；evaluator_v2.py 第271-273行注释也记录了此毛病），极慢。
  `deepseek-r1*`/`deepseek-r1-distill*` 是推理模型，也别用。`deepseek-v3.1`/`v3.2` 亦可（非推理、默认关思考）。
  官方 `deepseek-chat` 是**漂移别名**，按发布时间线依次指向 V3(2024-12~2025-08-20)→V3.1→
  V3.1-Terminus→V3.2(2025-12-01起)。`ds_run2~6.json` 的 `agent_info` 全为 None、无模型名无日期，
  但文件 mtime 为 2026-08-16（"几天前"），处 v3.2 时代。**已用 run9（官方 chat）与 run8（百炼 v3.2）
  互验确认**：两者指标几乎重合 → 官方 chat 当前=v3.2，且百炼≈官方后端一致；故 ds_run2~6 确为 v3.2。
  版本混淆（confound）已消除。
  **必须**显式 `enable_thinking=False`（经 `extra_body` 透传）；百炼上 `reasoning_effort` 的
  low/medium 等同 high，命令行降不下来。薄封装 `run_bailian_deepseek.py`（通用非推理封装，
  默认 `deepseek-v3.2`，由旧 `run_deepseek_v4flash.py` 改名而来，不再误导），继承
  `OpenAICompatibleAgent`、仅重写 `_create` 注入 `enable_thinking=False`，**未改动 evaluator_v2.py 源**，
  复用 `evaluate_dataset`+5工具+`_score_answer`。
  一键脚本 `run_v32.ps1`（PowerShell）：已写死百炼 Key/URL、`--model deepseek-v3.2`，
  输出名自动顺延（找最大 ds_runN.json → 输出 ds_run(N+1).json），用法
  `.\run_v32.ps1 [-Samples 500] [-Verbose] [-DryRun]`。⚠️ `run_v32.ps1` 含 API Key 已加入
  .gitignore，**勿提交到远程仓库**。
  手动命令：`python run_bailian_deepseek.py dataset --model deepseek-v3.2 --max-samples 50 --output ds_runX.json --verbose`
  （base_url 默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`；或官方端点
  `https://api.deepseek.com/v1 --model deepseek-chat`）。

- **ds_run7 = 百炼 deepseek-v3.2（无推理，2026-08-21，50样本，调制优化后）**：由 run8 改名而来
  （旧 run7 是百炼初代 v3 错版本、DOA MAE 8.36° 离群，已删）。指标：源数 0.72 / 类别 0.43 /
  频偏 0.50 / 带宽 0.43 / DOA 0.50 / 调制 0.36；DOA MAE 6.33° / 频偏MAE 0.019 / 带宽MAE(rel) 0.41；
  漏检 38 / 虚警 23 / 工具每样本 5.22。与 **ds_run8（官方 chat=v3.2，同优化后）** 几乎重合
  （调制0.33/DOA0.48/带宽MAE0.41），佐证 v3.2 结论。
- **调制提升归因（重要，纠正此前猜测）**：run6 调制 0.11 → run8/run9 调制 0.33~0.36 的跃升，
  **是用户在 run6 之后优化了调制识别代码/提示所致，非模型版本差异**（run6 与 run9 同为官方 chat）。
  ⚠️ 该优化是**权衡**：参数估计变好（频偏MAE 0.026→0.016、带宽MAE 0.70→0.41、调制 0.11→0.33），
  但源数(0.76→0.66)/类别(0.50→0.38)/DOA(0.58→0.48)与漏检(32→40)反而变差。写报告时需说明这是
  "优化后版本"，与 run2~6（优化前）不可直接拼成同一折线。

## 数据集
- dataset/ 含 500 样本 .npy + ground_truth.json / observations.json / metadata.json（metadata 不得给 Agent）
- 50 样本真值分布：0 干扰 14、1 干扰 15、2 干扰 21
- 运行评估需 numpy/scipy/tqdm（Python 3.13 装于 .workbuddy/binaries，用 `python -m pip install`）
