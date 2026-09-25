# 验证记录

日期：2026-09-21。开发环境：macOS arm64，Python 3.13.7。

## 已完成

- 18 项 unittest 通过：标签隔离、双跳协议、来源校验、记忆预算、搜索次数上限、成本核算、固定检索独立性、HotpotQA 转换、API usage／凭据处理，以及训练数值测试。
- 训练数值检查验证：仅生成 token 的 loss 与标准 masked causal loss 一致；正负优势的 PPO clipping；一次更新提高受奖励序列概率；采样保存的 token IDs 与 teacher-forcing log-prob 一致；LoRA 权重发生变化并可保存。
- 四组离线 demo 执行完成，文件位于 `outputs/demo-matrix/`。数据是两个虚构问题，策略是规则程序，仅用于检查编排和产物。
- 随机微型 GPT-2 模型完成两步 CPU LoRA SFT，checkpoint 位于 `outputs/cpu-sft-smoke/step-000002/`；loss 从约 3.326273 到 3.326206。只验证执行路径，不代表任务学习或模型能力。
- Python 模块编译检查通过。
- 两进程 CPU 同步测试通过：一个 rank 有梯度、另一个 rank 无本地有效动作，平均后两边执行相同更新。
- 两进程 CPU 微模型 SFT 集成测试通过，完成两步训练和 checkpoint 保存，位于 `outputs/cpu-distributed-sft/`。测试中发现 macOS 自动选择 barrier 设备的问题，已改为显式设备上的同步 collective。

## 依赖版本

CPU 数值测试使用 torch 2.14.0、transformers 4.57.6、accelerate 1.15.0、peft 0.21.0。它们是本机验证版本，不是 3090/CUDA 的已验证组合。GPU 服务器需要重新做兼容性 smoke test 并冻结环境。

## 初版时尚未验证（最新状态见后续记录）

- 完整规模的真实 HotpotQA 模型/API 评测（20 题排查结果见下文）。
- GPU 训练显存、吞吐、长序列与八卡运行。
- 完整真实环境中的 GRPO 收敛、奖励设计和跨 seed 稳定性。
- 记忆事实错误率、引用语义支持性、相对基线的算法提升。
- 完整 verl 多轮 rollout；目前只有独立最终答案 reward hook。

不得将本页的 demo／微模型测试数值当作项目的研究结果或简历指标。

## uv 环境迁移验证

项目已迁移到 uv：默认 Python 3.13（本机实测 3.13.7）、项目 `.venv`、`uv.lock` 与显式包索引。Python 3.11 managed 下载无进展后停止，改用已验证的本机解释器；包的兼容范围仍为 Python 3.11+。

- `uv sync --locked` 成功创建基础环境。
- `uv sync --locked --extra train` 成功安装训练依赖。
- `uv lock --check` 通过，锁文件与配置一致。
- `uv run --locked memsearch eval ...` 跑通，协议验证产物在 `outputs/uv-demo/`。
- `uv run --locked --extra train python -m unittest discover -s tests -v`：18 项测试全部通过。
- `uv run --locked --extra train memsearch-train --help` 正常加载新训练入口。
- 两个训练 shell 脚本的 `bash -n` 检查通过，内部已使用 uv 启动。

本次只验证本机 uv 环境；服务器 CUDA 与正式训练结果仍待验证。

## 显式记忆决策修正（2026-09-23）

默认记忆协议从整段重写改为 ADD / UPDATE / DELETE / NOOP，环境只原子执行操作；超预算拒绝，不自动淘汰。

- 在 uv 的 train 环境中运行完整测试：28 项全部通过。
- 新增 10 项记忆决策测试，覆盖稳定 ID、更新/删除/合并、NOOP、预算拒绝、整批回滚、非法提案惩罚、丢弃证据不再可见。
- 链路测试固定问题与证据，脚本策略分别 ADD 关键事实和 NOOP，经同一 Agent / collect_group 得到奖励 1 和 0、正负组内优势，并确认操作及参数 token 被保留。这是受控测试，不是学习后得到的行为。
- 原有真实微模型数值测试继续通过，包括 token log-prob、一轮梯度更新和 LoRA 保存。上述测试不等于在实际记忆任务上完成 GRPO 训练。

旧版 CPU SFT 产物使用旧摘要协议，只能作为历史训练器验证记录，不能当作新动作策略 checkpoint。正式 GPU RL 与算法效果仍未验证。


## 7B reader 约束与基线排查（2026-09-24）

来源：服务器运行者回传的 20 题 `summary.json` 和 `report` 输出。开发端未直接读取服务器完整产物；模型 revision、完整环境版本和数据指纹还需随正式实验归档。下表是调试记录，不是独立复现实验或 RL 训练结果。模型为本轮使用的 Qwen2.5-7B-Instruct，检索计划固定。

| 记忆模式 | reader 解码 | EM | F1 | 无效轨迹 |
|---|---|---:|---:|---:|
| 普通摘要 | text | 10% | 14.86% | 14/20 |
| 普通摘要 | JSON Schema | 35% | 47.89% | 3/20 |
| 显式操作 | text | 10% | 15.71% | 13/20 |
| 显式操作 | JSON Schema | 25% | 30.71% | 6/20 |

启用约束后，两组 reader 的 JSON 解析错误均清零。摘要组剩余 2 条 memory 空来源错误、1 条 reader 未知引用错误；操作组剩余 3 条 memory JSON 尾随文字错误、1 条不存在的记忆目标、1 条写入不可见来源，以及 1 条 reader 未知引用错误。约束只保证结构，不保证事实或引用正确。

此次本地代码回归共 36 项测试通过，包括 reader-only Schema 请求、默认关闭约束、保留原始截断响应和记录 finish_reason。API 测试使用模拟响应；服务约束的有效性证据来自上述服务器回传结果。

已具备尝试两步单卡训练链路检查的依据，但尚无 GPU RL 更新、7B 训练显存可行性、多卡收敛或算法收益的实测结论。API 约束只施加于冻结 reader，本地训练的 memory 仍自由采样。后续 before/summary/after 与训练必须使用一致的 reader 解码方式。


## 两步训练失败复现与采样对照（2026-09-24）

通过 SSH 在空闲 GPU 上复现原参数的两步 GRPO，冻结 reader 保持运行。新增 --trace-rollouts 保存完整调用、原始 token IDs 和 log-probs。结果与原失败一致：4 条轨迹均奖励 -0.1，无参数更新。

直接检查原始记录发现：所有失败 memory 调用均以 EOS 结束，生成长度 58–150 token，低于 512 上限；两条同位置报错的响应实际尾部不同，不是完全重复的采样结果。非法输出包括未闭合对象前插入杂乱字符、完整 JSON 后追加文字。

对两个失败任务的首轮提示词，确认重新编码的 prompt IDs 与训练记录完全相同后，分别测试贪心与温度 0.7。第一个任务两种设置仍追加代码围栏而解析失败；第二个任务的首轮两种设置均为合法 JSON。该对照只检验这两个提示词，不能证明全数据上的采样质量，也未做标签驱动的提示词优化。

目前没有证据将此次失败归因于长度截断或重复种子；也不能断言已经排除了所有实现问题。已修复诊断日志缺失和教程分支，未声称格式故障已修复。尚未执行非零优势反向传播、SFT 或长时间多卡训练。


## 训练端 Schema 修复验证（2026-09-24）

新增可选 MEMORY_OUTPUT=json_schema，API 评测与本地 memory GRPO 使用共享字段 Schema。Linux structured extra 固定 xgrammar 0.2.7。训练在 T=1 的合法 token 集合上采样；对采样时与更新时的概率都重放相同语法掩码并重新归一化，保留所有原始采样 token。来源与目标 ID 不由语法替模型选择，语义校验仍在环境中执行。

在服务器空闲 GPU 上重跑相同两题与种子（不是新的验证集效果实验）：

- 第一步：两条轨迹合法，但答案均错误，奖励 [0.0, 0.0]，没有更新。
- 第二步：两条轨迹合法，奖励 [1.0, 0.0]，产生正负优势并执行更新，无 OOM。
- 对比原两步未更新的同种子 adapter：392/392 个张量变化，最大绝对差约 1e-5，全部数值有限。
- 第二步总 loss 接近零不意味着梯度为零：初始概率比接近 1，组内正负优势的 loss 数值会抵消；实际参数变化已单独核验。
- 服务器 41 项测试全部通过，包含在线 grammar mask 与训练重放（含 EOS）一致性、约束分布归一化、真实微模型反向传播、API/训练 Schema 一致性。本机 41 项测试中 Linux XGrammar 项跳过 1 项，其余通过。

原始输出保存在服务器 outputs/diagnostic-grammar-001，参数对比为 parameter-check.json。这些是训练链路与数值验证，尚无多卡训练、长序列峰值显存或 RL 优于基线的结论。原始 text 记忆基线应与新 Schema 模式分开记录，正式比较需重新运行匹配基线。

共享 Schema 的 vLLM 接口另在两题 dev 上检查 before/summary，两组均无协议错误；这仅验证 API 兼容性，不用两题分数报告效果。


## 第一阶段端到端短程验收（2026-09-25）

2026-09-24 至 2026-09-25，在 8×RTX 3090 服务器完成第一阶段短程闭环。基座 Qwen2.5-7B-Instruct；GPU 0–5 训练，GPU 6 reader，GPU 7 合并后的 memory 服务；训练与推理均使用共享记忆 Schema。

| 组别 | EM | F1 | 无效轨迹 |
|---|---:|---:|---:|
| 未训练操作策略 | 30% | 38.21% | 2/20 |
| 普通摘要 | 35% | 47.89% | 1/20 |
| 4 步六卡 RL checkpoint | 30% | 41.55% | 1/20 |

操作组 F1 的单次差值为 +3.33 个百分点，但仍低于摘要基线 6.35 个百分点。EM 未变化。20 题与 4 步训练仅用于验收，不能证明稳定收益；没有据此选择最佳 checkpoint 或宣称收敛。

六卡各 4 步，共 96 条轨迹、4 次全局更新。训练中 9 条无效轨迹：4 条 NOOP 混用、4 条目标不存在、1 条来源不可见。未出现 JSON 解析错误。392 个 LoRA 张量均变化且数值有限，最大绝对变化约 4.01e-5。CPU 合并、GPU 7 部署和真实 chat 检查均通过。

训练后评测剩余 1 条错误为 memory 引用不可见证据，reader 无错误。三组的数据指纹、预算、reader 解码和固定检索设置通过自动对比检查。

实测产物（服务器仓库相对路径）：

- `outputs/grammar-001/train-e2e/step-000004/`：LoRA checkpoint。
- `checkpoints/grammar-001/`：合并后的完整模型。
- `outputs/grammar-001/e2e-before-20/`、`e2e-summary-20/`、`e2e-after-20/`：三组评测。
- `outputs/grammar-001/e2e-comparison.json`、`training-audit.json`：对比与训练审计。
- `outputs/grammar-001/model-provenance.json`、`code-fingerprints.json`、依赖清单与锁文件：复现记录。

41 项服务器测试通过。验收结束时 GPU 0–5 已释放，reader 与 memory 服务分别保留在 GPU 6、7；不再评测时可在各自 tmux 窗口按 Ctrl-C 关闭，不使用批量 kill。

本轮评测 dataset_sha256：`f018b5c2e89804d33eee91da970be10534f8433b294471064c11379b80992456`。完整环境版本保存在忽略提交的实验产物中，不将个人路径或账户写入公开文档。
