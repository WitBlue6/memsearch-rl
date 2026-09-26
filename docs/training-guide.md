# MemSearch RL：从配置到训练后评测

本教程完成第一阶段闭环：下载并校验数据 → 部署冻结 reader → 训练前基线 → 单卡检查 → 六卡短程 RL → 合并 LoRA → 部署记忆模型 → 训练后评测与对比。本文保留最初短程验收的步骤；后续已完成的研究实验与搜索控制器训练见下方入口。

研究对象是预算约束下的 ADD / UPDATE / DELETE / NOOP 记忆决策。固定 BM25 检索计划与 reader，只训练记忆策略 LoRA；数据为 HotpotQA distractor，每题记忆独立。默认最终答案 F1 为奖励，无效轨迹奖励 -0.1。参考训练器使用组内相对优势、clipped policy loss，KL 系数为 0。不是 full-wiki、跨会话长期记忆或完整 verl 实现。

本文先用 20 题评测与六卡各 4 步训练验收全链路；这不等于模型收敛，也不能用短程结果宣称稳定算法提升。具体实测结果见末尾与 [验证记录](validation.md)。

## 后续研究与结果入口

首次运行先完成本文的基础闭环。已经跑通后，进入 [研究实验流程](research-workflow.md)，按独立数据划分 → 公平基线与多种子学习曲线 → 预算与证据顺序实验 → SFT 冷启动 → 冻结 memory 的 controller 训练 → 锁定测试与报告的顺序开展实验。

- [研究实验流程](research-workflow.md)：研究版配置、服务和各阶段命令，主脚本为 [research_pipeline.py](../scripts/research_pipeline.py)。
- [项目贡献与实验总结](project-contributions.md)：controller／memory／reader 的职责、具体代码改进，以及已完成的 200 题 test 结果与结论边界。
- [验证记录](validation.md)：基础闭环的历史排错与验收记录。

本文第 13 节仍是最初 20 题短程验收记录，不是最新研究结果。研究主脚本的 `test` 阶段覆盖原始基线和三个 memory RL 种子；SFT 与 controller 的扩展测试需要额外通过 [research_eval.py](../scripts/research_eval.py) 运行。已有实验产物应保留，不要为跟随教程重复覆盖或重跑已完成阶段。

## 1. 安装项目环境

将项目放到服务器任意目录，以下命令均从仓库根目录执行。无需 sudo。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra train --extra structured
uv run --locked --extra train --extra structured python -m unittest discover -s tests -v
uv run --locked --extra train --extra structured python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

已有 uv 或环境可跳过对应安装。训练使用项目 Python 3.13；后面的独立推理环境使用 Python 3.12，二者在同一服务器，通过 HTTP 通信。structured extra 固定 XGrammar 版本，当前本地约束实现支持 Linux。

## 2. 创建本地实验配置

```bash
cp -n configs/experiment.env.example configs/experiment.local.env
nano configs/experiment.local.env
export MEMSEARCH_SETTINGS=configs/experiment.local.env
source "$MEMSEARCH_SETTINGS"
```

每个新 SSH/tmux 终端都需要重新设置 `MEMSEARCH_SETTINGS` 并 source 对应文件。不要一边 source 默认配置，一边运行指向另一份配置的脚本。

已有本次验收配置时，可改用 `export MEMSEARCH_SETTINGS=configs/experiment-grammar.local.env`，无需重新生成或覆盖。新实验使用未占用的 EXPERIMENT_ID。

| 配置 | 作用 |
|---|---|
| `EXPERIMENT_ID` | 本轮编号，改变模型或解码设置时使用新编号 |
| `READER_REPO` / `READER_MODEL` | 下载仓库 ID / reader 本地模型目录 |
| `POLICY_REPO` / `POLICY_MODEL` | 下载仓库 ID / 待训练基座目录；本教程与 reader 相同 |
| `BUDGET_TOKENIZER` | 用于证据与记忆预算计数的 tokenizer，所有对照组保持相同 |
| `READER_OUTPUT=json_schema` | 约束冻结 reader 回答结构 |
| `MEMORY_OUTPUT=json_schema` | 在 API 评测和本地 RL 中使用共享记忆 Schema |
| `SERVING_ENV` | 独立 vLLM 环境 |
| `TRAIN_GPUS` / `TRAIN_PROCESSES` | 示例 GPU 0–5 / 6 个进程 |
| `READER_GPU` / `READER_PORT` | 示例 GPU 6 / 8000 |
| `MEMORY_GPU` / `MEMORY_PORT` | 示例 GPU 7 / 8001 |
| `TRAIN_DATA` / `DEV_DATA` / `SMOKE_DATA` | 训练、完整 dev、小样本 dev 目录 |
| `MAX_NEW` | 训练和评测的输出上限，默认 512 |
| `MEMORY_MODEL` | 合并权重的输出目录，必须尚不存在 |

模型名只在配置中设置。示例将 reader、policy 和预算 tokenizer 都指向同一 7B 目录。下载仓库与本地目录必须匹配；不能把 1.5B 仓库下载到 7B 目录。预算 tokenizer 不额外加载一份模型权重。

API 使用角色别名 memsearch-reader / memsearch-memory。所有服务只监听本机。配置由 Bash source，仅使用可信文件；密钥通过环境变量提供，不写进文档或提交 Git。

## 3. 下载、转换和校验数据

已有验证通过的数据可跳过。脚本使用固定 Hugging Face revision，下载中断文件保留为 .part，转换结果拒绝覆盖。

```bash
bash scripts/download_hotpot.sh
uv run --no-project --python 3.13 --with pyarrow==25.0.1 python scripts/convert_hotpot_parquet.py
uv run --locked --extra train --extra structured python scripts/clean_hotpot.py
```

完整 Parquet 元数据应为 90447 / 7405 条，脚本取前 2000 / 200 条。异常支持句子整题剔除，保留原始 JSON 和 rejected 报告，不补造标签。

```bash
uv run --locked --extra train --extra structured memsearch prepare-hotpot --input data/raw/hotpot_train_hf_2000.clean.json --out data/processed/train-001
uv run --locked --extra train --extra structured memsearch prepare-hotpot --input data/raw/hotpot_dev_hf_200.clean.json --out data/processed/dev-001
uv run --locked --extra train --extra structured memsearch prepare-hotpot --input data/raw/hotpot_dev_hf_200.clean.json --out data/processed/dev-smoke-002 --limit 20
uv run --locked --extra train --extra structured memsearch validate-data --data data/processed/train-001
uv run --locked --extra train --extra structured memsearch validate-data --data data/processed/dev-001
```

该固定子集的已验证计数：训练 19207 documents / 1999 tasks；dev 1991 documents / 200 tasks。不要删除成功数据来重跑教程。

## 4. 下载模型和安装服务环境

```bash
bash scripts/experiment.sh download --role reader
bash scripts/experiment.sh download --role policy
```

两个角色使用同一仓库和目录时复用已下载文件。已有可用 vLLM 环境可跳过安装：

```bash
source "$MEMSEARCH_SETTINGS"
uv venv --python 3.12 "$SERVING_ENV"
uv --no-config pip install --python "$SERVING_ENV/bin/python" --index-url https://pypi.tuna.tsinghua.edu.cn/simple vllm ninja --torch-backend=auto
```

vLLM 与驱动兼容性以实际服务器为准，参考 [官方安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。独立服务环境不受项目 uv.lock 管理，验证后保存版本清单。每张训练卡持有完整模型副本，没有 FSDP，不能把多卡显存视作一张大卡。

## 5. 生成配置与启动 reader

每个新实验只生成一次：

```bash
bash scripts/experiment.sh config
```

生成 configs/generated/<编号>/ 下的 before/summary/after.toml 和模型设置快照。修改设置后用新编号；脚本会拒绝漂移配置及已有目录。

如果旧 reader 占用设备或端口，先进入其 tmux 窗口按 Ctrl-C，等它退出。Ctrl-Z 不释放显存；不要批量终止其他人的 GPU 进程。

```bash
tmux new -s memsearch-reader
```

在服务窗口回到仓库根目录，重新设置配置选择后启动：

```bash
export MEMSEARCH_SETTINGS=configs/experiment.local.env
bash scripts/experiment.sh serve --role reader
```

若用的是其他配置，替换这里的文件名。脚本自动设置推理环境 PATH，并检查 Ninja，避免 FlashInfer 找不到 ninja。服务就绪后用 Ctrl-b 后 d 脱离窗口，在工作终端检查：

```bash
bash scripts/experiment.sh health --role reader
```

health 检查模型别名与普通 chat 请求；下一步评测才检验 Schema 请求。已有相同 reader 正常运行时，不必重启。

## 6. 跑两组训练前基线

```bash
source "$MEMSEARCH_SETTINGS"
bash scripts/experiment.sh eval --setting before --data "$SMOKE_DATA" --out "outputs/$EXPERIMENT_ID/e2e-before-20"
bash scripts/experiment.sh eval --setting summary --data "$SMOKE_DATA" --out "outputs/$EXPERIMENT_ID/e2e-summary-20"
bash scripts/experiment.sh report --out "outputs/$EXPERIMENT_ID/e2e-before-20"
bash scripts/experiment.sh report --out "outputs/$EXPERIMENT_ID/e2e-summary-20"
```

before 使用未训练的显式操作策略，summary 使用未训练的摘要策略，均调用 reader 基座服务；reader、检索计划和预算一致。报告包含失败角色、前三条失败原文、finish_reason 和 decoding_mode。

Schema 只约束字段与 JSON 结构。非法目标、不可见来源、超预算、NOOP 混用及事实错误仍可能发生；不能将它们静默修复。持续大量失败时先看附录诊断，不直接启动长训练。

## 7. 单卡两步检查

```bash
bash scripts/experiment.sh smoke --trace-rollouts
source "$MEMSEARCH_SETTINGS"
bash scripts/experiment.sh report --out "outputs/$EXPERIMENT_ID/smoke"
```

使用 TRAIN_GPUS 的第一张卡，不覆盖旧输出。若需要重跑，用 `--out` 指定新目录。此步骤是检查模型加载、采样、奖励、反向传播及保存，不是收敛测试。

继续条件：至少观察到有效更新，输出无无法解释的异常且无 OOM。全部同分时优势为零，updated=false；不代表必须 SFT。先检查原因，再决定修复或冷启动。OOM 不能靠增加训练副本数量解决。

## 8. 六卡短程训练

在训练 tmux 窗口中使用相同配置，保留 reader 服务：

```bash
source "$MEMSEARCH_SETTINGS"
bash scripts/experiment.sh train --steps 4 --trace-rollouts --out "outputs/$EXPERIMENT_ID/train-e2e"
```

每 rank 每步一道题、每题 4 条轨迹，六卡四步共 24 个训练任务位置、96 条采样轨迹。各 rank 处理不同题目，平均 LoRA 梯度。updated 表示全局发生更新，某 rank 即使本地 zero_variance=true 也可接收其他 rank 的梯度。

```bash
bash scripts/experiment.sh report --out "outputs/$EXPERIMENT_ID/train-e2e"
```

预期六份 rank 日志各 4 行，checkpoint 为 step-000004。总 loss 接近零可能来自组内正负优势抵消，不能只凭这个数断言没有学习；结合 updated、梯度和 checkpoint 参数变化判断。

需要更长实验时用新输出目录增大 steps，同时调整后面的 adapter 路径。4 步验收不代表 100 步或任意序列长度都已验证。当前脚本最后一步保存；--adapter 仅恢复权重，优化器重新开始，不是精确断点续训。

## 9. 合并 LoRA

训练正常结束后执行：

```bash
source "$MEMSEARCH_SETTINGS"
bash scripts/experiment.sh merge --adapter "outputs/$EXPERIMENT_ID/train-e2e/step-000004"
```

CPU 合并需要足够内存与磁盘，7B FP32 权重约 30 GB 量级。输出为配置中的 MEMORY_MODEL，已有目录时拒绝覆盖。必须使用训练时的相同基座。

## 10. 部署训练后的记忆模型

新建独立 tmux 服务窗口，reader 保持运行：

```bash
tmux new -s memsearch-memory
```

在仓库根目录中选择同一配置并启动：

```bash
export MEMSEARCH_SETTINGS=configs/experiment.local.env
bash scripts/experiment.sh serve --role memory
```

服务加载 MEMORY_MODEL，使用 MEMORY_GPU/PORT 和别名 memsearch-memory。另一个终端检查：

```bash
bash scripts/experiment.sh health --role memory
```

## 11. 训练后评测

```bash
source "$MEMSEARCH_SETTINGS"
bash scripts/experiment.sh eval --setting after --data "$SMOKE_DATA" --out "outputs/$EXPERIMENT_ID/e2e-after-20"
bash scripts/experiment.sh report --out "outputs/$EXPERIMENT_ID/e2e-after-20"
```

after 只切换 memory 后端到合并 checkpoint，reader、预算、固定检索与 Schema 保持一致。不要用旧 text 模式基线作新 Schema 模式的直接对照。

## 12. 自动对比与保存环境

```bash
uv run --locked --extra train --extra structured python scripts/compare_results.py \
  --before "outputs/$EXPERIMENT_ID/e2e-before-20" \
  --summary "outputs/$EXPERIMENT_ID/e2e-summary-20" \
  --after "outputs/$EXPERIMENT_ID/e2e-after-20" \
  --out "outputs/$EXPERIMENT_ID/e2e-comparison.json"
```

脚本核对数据指纹、题数、预算 tokenizer、reader 设置和固定检索配置，然后报告 EM/F1、invalid、记忆召回和成本。不检查模型权重身份或事实蕴含，这些仍需结合 manifest、部署日志和人工抽查确认。

```bash
uv pip freeze --python .venv/bin/python > "outputs/$EXPERIMENT_ID/train-packages.txt"
uv pip freeze --python "$SERVING_ENV/bin/python" > "outputs/$EXPERIMENT_ID/serving-packages.txt"
nvidia-smi > "outputs/$EXPERIMENT_ID/gpu.txt"
cp uv.lock "outputs/$EXPERIMENT_ID/uv.lock"
```

同时保存代码版本、模型 revision 和运行配置。正式结论应扩大到完整 dev、多 seed 和预算消融。after-before 是同协议下训练前后差异；after-summary 同时涉及协议与训练差异。一次短程差值不能证明稳定收益。

## 13. 实测记录

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


## 附录 A：失败先诊断，SFT 不是必经步骤

--trace-rollouts 将原始提示词、响应、token IDs、old log-probs 和 Schema 保存在 rollouts-rank-*.jsonl。先检查 finish_reason 与生成长度，区分长度截断、错误模板、JSON 格式和动作语义错误。

之前的两步自由采样实验中，4 条轨迹均产生非法 JSON，全部奖励 -0.1。复现确认并非长度截断或完全重复采样；低温和预填开头没有可靠修复。现在的可选 Schema 在采样时屏蔽非法 token，训练时重放相同掩码并重新归一化新旧概率，保留原始 token，不做尾部删除。

单卡实测已产生 [1.0, 0.0] 奖励与真实 LoRA 更新。因此当前主线直接验证约束模式，无需先做 SFT。若后续合法轨迹仍长期全部同分，应分析任务难度、事实质量与奖励分布；SFT 仅是诊断后可选的冷启动方案，只能使用 train split，且需人工审查教师事实。采用时分别评测 SFT 与 SFT+RL，不能混算贡献。

| 故障 | 处理 |
|---|---|
| Settings changed since config generation | 检查本地设置，换新编号生成配置；保留旧结果 |
| 输出目录已存在 | 换新路径，不删除或覆盖历史实验 |
| HTTP 错误/端口不可达 | 检查对应 tmux 日志和 health，再检查 Schema 兼容性 |
| 找不到 ninja | 通过 serve 脚本启动，确保服务环境中已安装 ninja |
| Schema 输出仍 invalid | 检查引用、目标 ID、预算与长度；语法正确不保证动作正确 |
| 全组零优势 | 检查原始轨迹和奖励；不能自动跳到长训练或 SFT |
| OOM | 停止扩容运行，检查实际序列长度与单卡峰值；多副本不分摊模型 |

## 附录 B：发布与后续研究

.gitignore 排除本地配置、原始/处理后数据、虚拟环境、输出、checkpoint、权重和常见密钥文件。忽略规则不能自动移除已跟踪的文件，提交前检查暂存区和历史。公开文档不包含个人账户、SSH 地址或用户目录；运行产物可能含本机路径，应单独审查后分享。

后续 controller 训练使用 [研究实验流程](research-workflow.md) 中的 `research controller` 入口：冻结指定 memory checkpoint 和 reader，训练 SEARCH(query)／FINISH 决策。当前 API 与本地 Schema 均已支持 controller；本轮研究协议由冻结 reader 生成最终答案。该分阶段流程已有实测结果，见 [项目贡献与实验总结](project-contributions.md)；完整 verl AgentLoop、长期学习效果和统计显著性仍未得到验证。

文件入口：configs/experiment.env.example（统一配置）、scripts/experiment.sh / experiment.py（流程）、scripts/compare_results.py（对比）、src/memsearch/structured.py（共享 Schema 与概率掩码）、training.py（SFT/GRPO）。
