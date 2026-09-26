# MemSearch RL

从下载数据到训练、部署 checkpoint、对比评测，按步骤执行：[SSH 服务器训练与结果检验教程](docs/training-guide.md)。

预算约束下的记忆增强检索 Agent：先用 RL 学习如何处理新证据（ADD / UPDATE / DELETE / NOOP），再学习搜索与停止决策。

研究假设：**保留来源及未解决子问题的压缩记忆，能够帮助 Agent 找到缺失证据，在固定预算下提高多跳问答质量。** 这是待验证的假设，不是已经得到的结论。

目前提供可运行的检索与记忆环境、四组对照评测、HotpotQA 格式转换、真实模型 API 接口，以及小模型 LoRA SFT / GRPO 参考训练器。离线 demo 是虚构数据和确定性规则，仅验证代码。已完成真实数据 20 题基线排查及 reader 格式约束验证，详见 [验证记录](docs/validation.md)；已完成记忆 Schema 模式下的六卡各 4 步 RL、checkpoint 合并部署和同一 20 题训练后评测；短程操作组 F1 为 38.21% → 41.55%，仍低于摘要基线 47.89%。这只是闭环验收，尚无稳定算法提升结论。

模型、下载目录、GPU 与端口集中设置在 `configs/experiment.env.example`；复制为 `configs/experiment.local.env` 后，通过 `bash scripts/experiment.sh` 执行下载、服务、配置生成、评测及训练。详见教程。

## 本地运行

统一使用 uv 管理项目环境，`.python-version` 固定 Python 3.13，`uv.lock` 锁定依赖版本。核心 demo 不安装训练依赖，也不需要 API key。无需手动激活虚拟环境或设置 PYTHONPATH。

```bash
cd /path/to/memsearch-rl
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked memsearch eval \
  --config configs/demo.toml --data data/demo --out outputs/my-demo
uv run --locked memsearch matrix \
  --config configs/demo.toml --data data/demo --out outputs/my-matrix
```

只安装基础环境时，训练数值测试会标记为 skipped；完整测试使用 `--extra train`。`uv run --locked` 检查锁文件一致性；有意更新依赖时运行 `uv lock`。

输出目录必须不存在，避免覆盖实验。每组输出包含 `trajectories.jsonl`、逐题 `metrics.jsonl`、汇总 `summary.json`、配置和数据指纹。Demo 的四组名称只测试实验编排，没有训练后的模型，也不能用于简历中的算法提升数字。

## 真实模型和数据

需要真实 tokenizer、训练或完整数值测试时，安装 train extra：

```bash
uv sync --locked --extra train
uv run --locked --extra train python -m unittest discover -s tests -v
uv run --locked --extra train python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

GPU PyTorch/CUDA 组合需要根据驱动选择，确认 `torch.cuda.is_available()` 后再继续。模型服务建议另建环境，避免 vLLM 与训练依赖互相覆盖。先单卡 smoke test，再增加训练卡数。仓库提供跨平台依赖锁；它锁定软件版本，但不保证未知服务器驱动与所选 PyTorch wheel 兼容。uv 同步后仍需完成该服务器上的 CUDA smoke test。不要直接用 pip 覆盖 uv 管理的依赖；需要调整版本时修改 pyproject.toml 并重新生成 uv.lock。

从 [HotpotQA 官方项目](https://github.com/hotpotqa/hotpot) 获取官方 train 与 dev-distractor JSON，分别转换；不要从测试集构造训练样本。

```bash
uv run --locked memsearch prepare-hotpot --input data/raw/hotpot_train_v1.1.json \
  --out data/processed/train --limit 2000
uv run --locked memsearch prepare-hotpot --input data/raw/hotpot_dev_distractor_v1.json \
  --out data/processed/dev --limit 200
uv run --locked memsearch validate-data --data data/processed/train
```

转换后：`corpus.jsonl` 保存文档，`tasks.jsonl` 只含问题和候选文档 ID，答案与支持证据在独立的 `labels.jsonl`。Agent 不接收标签对象。

初版默认在每题给定候选文档内检索，**不是 full-wiki 开放域成绩**。`retrieval_scope="corpus"` 可以搜索整个导入语料，但该语料仍只是导入的数据，不等同于全量维基。规模增大后应替换当前内存 BM25；它用于小规模可解释实验。

启动你自己的 OpenAI-compatible 模型服务，在 `configs/api.toml` 填入地址及模型名。需要鉴权时仅通过 `MEMSEARCH_API_KEY` 环境变量传入；不要把密钥写入配置或提交。

```bash
uv run --locked --extra train memsearch eval --config configs/api.toml --data data/processed/dev --out outputs/prompt-dev
uv run --locked --extra train memsearch matrix --config configs/api.toml --data data/processed/dev --out outputs/prompt-matrix
```

配置中的 tokenizer 决定证据／记忆预算，各组保持一致。API usage 记录真实调用 token；无 usage 的服务退回配置的 tokenizer 估计计数，以 `estimated_` 前缀标记，不与服务报告值混加。离线 demo 使用 `lexical_proxy`，不是模型 token。

## 分阶段训练

### 0. SFT 冷启动（可选）

以下底层命令中的 `POLICY_MODEL` 需显式设置为策略模型路径。推荐先按教程使用统一脚本。

用教师模型在 **train split** 生成轨迹、按答案表现筛选并人工抽查事实与引用，导出某个角色：

```bash
uv run --locked memsearch export-sft --trajectories outputs/teacher-train/trajectories.jsonl \
  --metrics outputs/teacher-train/metrics.jsonl --role memory --out data/processed/memory-sft.jsonl
uv run --locked --extra train memsearch-train sft --role memory --model "$POLICY_MODEL" \
  --data data/processed/memory-sft.jsonl --out outputs/memory-sft --steps 100
```

教师轨迹必须由 `memory="decision"` 生成，completion 是操作 JSON；旧版整段 `items` 摘要数据不能直接作为该策略的冷启动数据。筛选成功答案不保证中间记忆真实；这一步不能替代内容审计。仅 completion token 参与 SFT loss。

### 1. 训练记忆模块

固定检索计划与回答模型，只优化记忆操作策略。模型看到旧记忆、新证据和预算，生成操作及参数；环境原子执行，超预算直接判无效，不自动删除。GRPO 用最终答案奖励训练这些操作 token，默认 `--memory-mode decision`。启动脚本默认训练使用 GPU 0–5，预留 GPU 6–7 给固定模型服务；服务也可以在远程机器。先修改 `configs/api.toml` 的 reader endpoint。

```bash
TRAIN_DATA=data/processed/train RUN_OUT=outputs/memory-grpo \
  bash scripts/train_memory.sh --steps 100 --group-size 4
```

可以传 `--adapter outputs/memory-sft/step-000100` 加载 SFT 适配器。该参数只恢复权重，优化器重新开始，不是精确断点续训。

### 2. 冻结记忆，训练搜索策略

先合并阶段一权重，使用独立推理服务部署，再在新配置的 `[memory_backend]` 指向该模型。控制器只输出 SEARCH 或 FINISH。

```bash
uv run --locked --extra train python scripts/merge_adapter.py --model "$POLICY_MODEL" \
  --adapter outputs/memory-grpo/step-000100 --out checkpoints/memory-stage1
TRAIN_DATA=data/processed/train RUN_OUT=outputs/controller-grpo \
  EXPERIMENT_CONFIG=configs/stage2.local.toml \
  bash scripts/train_controller.sh --steps 100 --group-size 4
```

`stage2.local.toml` 需要从 `api.toml` 复制后填入真实冻结模型服务地址。训练脚本不自动占用额外 GPU 启动推理服务。

训练以最终答案 F1 为奖励，默认不惩罚搜索次数；先获得有效学习信号，再消融 `--search-cost`。无效动作／引用会受罚。监测日志中的 `zero_variance` 与 `errors`，长期全零优势时应先修复格式或任务难度。

### 训练器的明确边界

- 这是可检查的小模型参考实现：默认全分布 T=1 采样（可选 Schema 模式在合法 token 集合归一化采样，并重放相同掩码计算新旧概率）、组内相对优势、clipped policy loss、每条轨迹按生成 token 数归一化、KL 系数为 0。
- 采样和 loss 保留原始 token IDs，观察／检索结果不计入策略 loss。
- 多进程每卡一个 LoRA 模型副本；显式平均梯度，允许不同 rank 的轨迹步数不同。默认 **6 卡训练＋预留 2 卡服务**，不是把 192GB 显存当作单卡。
- 没有 vLLM rollout 加速、异步流水线或 FSDP；不能直接扩到需要跨卡容纳的模型。
- 已提供单独 `verl_reward.py` 的最终答案评分接口；**完整 verl 多轮 AgentLoop 尚未接入**。待奖励与环境稳定后迁移，避免同时调试算法和框架。不能把参考训练器描述成 verl 训练结果。
- 当前没有证明稳定效果提升；已验证教程中的 3090 六卡短程配置，不代表任意长序列、长训练配置或收敛。

## 四组对照

| 组别 | 记忆 | 搜索 |
|---|---|---|
| A | 普通摘要 | 固定检索计划 |
| B | 显式记忆操作策略（阶段一 checkpoint） | 固定检索计划 |
| C | 普通摘要 | 阶段二控制器 |
| D | 显式记忆操作策略（阶段一 checkpoint） | 阶段二控制器 |

没有加载已训练 checkpoint 时，B/C/D 只是提示词／模型策略基线，不应称为 RL 结果。Matrix 的 A/C 默认使用原始 backend 生成摘要；B/D 使用 `memory_backend`；C/D 使用相同 `controller_backend`。可用 `baseline_memory_backend` 显式指定摘要模型。

C 是固定控制器下的交叉消融；如果要严格比较“分别训练的搜索策略”，应另训一个在普通摘要上训练的控制器并单独评测。两种结论不要混用。

当前自动指标：EM/F1、检索支持证据召回、记忆支持证据召回、引用与标注支持证据的匹配、搜索次数、无效动作率、已执行记忆操作计数、全部模型调用 token 成本。**来源 ID 合法不代表语义蕴含成立**，引用支持性／记忆事实错误率需要额外人工或独立评审，报告会明确 `entailment_evaluated=false`。

## 文件导航

- `src/memsearch/agent.py`：决策循环、预算和证据可见性。
- `memory.py` / `prompts.py`：构造记忆决策观察并调用策略。
- `memory_actions.py`：校验并原子执行 ADD / UPDATE / DELETE / NOOP，不自动替策略淘汰条目。
- `training.py`：小模型 SFT／GRPO 参考训练器。
- `runner.py` / `metrics.py`：评测、成本和实验记录。
- `tests/`：数据泄漏、预算、引用、训练 loss 和采样一致性测试。
- [记忆决策与 RL 链路](docs/memory-decisions.md) / [代码导读](docs/code-tour.md) / [研究计划](docs/research-plan.md) / [服务器准备](docs/server.md) / [验证记录](docs/validation.md)。

## 参考来源

本项目代码为本次新建实现；研究设计参考以下项目，尚不声称完整复现或新算法贡献：

- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)：多轮 RL 与任务内记忆。
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)：搜索工具与强化学习。
- [MemFactory](https://github.com/MemTensor/MemFactory)：记忆训练实验。
- [verl Agentic RL](https://verl.readthedocs.io/en/latest/start/agentic_rl.html)：后续训练底座。
- [HotpotQA](https://github.com/hotpotqa/hotpot)：多跳问答数据与评测。

## uv 使用约定

提交 `pyproject.toml`、`uv.lock` 和 `.python-version`；`.venv/` 不提交。普通命令使用 `uv run --locked`，依赖模型库的命令使用 `uv run --locked --extra train`。训练 shell 脚本内部已使用 uv，不依赖当前 shell 的 Python/Accelerate 安装。

uv 官方说明：[依赖锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)。

依赖源在 pyproject.toml 中显式固定为本次解析使用的清华 PyPI 镜像，避免迁移到服务器后受全局 uv 配置差异影响。若要换源，同时更新配置与 uv.lock。项目默认 Python 3.13，包本身仍声明兼容 Python 3.11+。

## 研究实验

端到端验收之后，按 [研究流程](docs/research-workflow.md) 运行独立数据划分、严格预算基线、多种子学习曲线、机制与冷启动对照、记忆驱动搜索。具体已完成结果见验证记录；新增入口不代表已经验证算法优势。
