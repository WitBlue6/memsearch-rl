# MemSearch RL 研究流程

本流程接续 `training-guide.md` 的端到端验收。先回答是否有效，再研究机制、冷启动和搜索；所有未完成实验都不得写成结果。

## 1. 环境与数据

先按原教程启动冻结 reader，设置自己的本地配置。下文不包含个人用户名、主机或绝对路径。

```bash
source configs/experiment.local.env
uv sync --locked --extra train --extra structured
uv run --locked --extra train --with pyarrow==25.0.1 python scripts/research_data.py --test 200
```

要求已有 `data/raw/hotpot-hf` 三个 Parquet 文件。脚本排除旧前 200 道官方 dev 题，按固定种子随机选 80 道 tune、200 道 test；训练集独立抽取 2000 道有效题。按原始 ID 和规范化问题去重，不宣称已解决语义近重复。`splits.json` 记录全部 ID、数据指纹及原始 Parquet 指纹。标签只供奖励和离线评测，不能进入 Agent 状态。

## 2. 固定实验计划

```bash
uv run --locked --extra train --extra structured python scripts/research_pipeline.py setup --model "$POLICY_MODEL" --config "configs/generated/$EXPERIMENT_ID/before.toml"
```

默认输出 `outputs/research-v1`，三个种子 42/43/44，每个使用两张卡，12 次优化迭代，第 6、12 步保存。默认训练卡 0–5，reader 应在另一张卡上。这是探索性短程实验，不是已收敛训练。数据集大小与实际采样题数分开报告。

其他命令也必须传入相同 `--model`、`--config` 以及任何自定义的 `--root/--data/--steps/--seeds/--gpus`。计划漂移直接报错；不得覆盖历史产物。失败输出保留排查，重新执行应使用新的研究输出目录。

## 3. 基线与学习曲线

下文用 shell 函数缩短命令；此函数仅存在于当前终端。

```bash
research() {
  uv run --locked --extra train --extra structured python scripts/research_pipeline.py "$@" --model "$POLICY_MODEL" --config "configs/generated/$EXPERIMENT_ID/before.toml"
}
research baselines
research train
research curves
```

四组 API 基线：显式操作、普通摘要、摘录、最近证据。摘要与显式操作都拒绝超预算结果；规则基线依靠确定性选取满足预算。显式记忆 ID 也是表示开销，不应隐藏。检索、reader、证据预算和计数 tokenizer 相同。

`curves` 首先评测与 adapter 相同后端的未训练模型，再评测六个 adapter。默认采用本地 Transformers；大量评测可使用下述 vLLM LoRA 服务。训练采样为 T=1，更新时重放相同 grammar mask；评测为 greedy。`before-matched` 与训练后策略必须使用同一后端，不把解码实现差异归为 RL 收益。

在空闲 GPU 的独立终端运行研究服务（默认 GPU 7，不要与已有服务争用）：

```bash
bash scripts/serve_research.sh
```

评测终端设置：

```bash
export MEMSEARCH_RESEARCH_API=http://127.0.0.1:8002/v1
```

服务在 loopback 上运行，开启 LoRA 动态加载。adapter 名称由权重与配置的内容哈希生成，避免同名覆盖。原 reader 无需重启。相关接口见 [vLLM 官方说明](https://docs.vllm.ai/en/latest/features/lora/)。该变量应在所有 matched 基线和 checkpoint 评测中保持一致。

新增日志包括 valid_fraction、组内零方差、生成长度、梯度范数、概率比、clip_fraction、近似 KL。KL 统计针对旧 rollout 策略，不是基座 reference KL；仅统计参与更新的 token，不是全状态分布的 KL。

## 4. 机制验证

```bash
research mechanisms
```

在 tune 上改变记忆预算 256/1024，或把同一 BM25 top-k 总序列反转，比较 before/summary/after。固定使用预先指定的 seed 42 最终 checkpoint，避免挑最好种子。预算变化测试的是迁移，不是针对每个预算重训。反转测试不利用 gold support 选择顺序，不等同于人为保证“关键证据早到”。

每题 `metrics.jsonl` 中的 `memory_trace_metrics` 记录逐步支持来源召回、丢失/新增来源以及序列化记忆大小。来源保留不等于事实忠实，不能称为语义正确率或零幻觉。长程与额外噪声语料实验需另行构造，当前 Hotpot candidate-pool 不能代替 full-wiki 搜索。

## 5. 冷启动对照

```bash
research coldstart
```

从本次 GRPO 的训练轨迹筛选完整合法且答案 F1≥0.5 的 memory 调用，去重后做 12 步 SFT，再做 12 步 GRPO。仅允许 train task ID；标签只用于离线筛选，不加入 messages。这是自生成银标，既非专家示范，也未经语义蕴含验证。

分别评测 SFT、SFT+GRPO。此探索对照的训练 token 与原始 RL 不一定相等，不能直接作为等计算量优势证据；正式消融还要对齐生成数据与优化开销。若无合格数据直接失败，不偷偷放宽门槛。

## 6. 搜索策略

```bash
research controller
```

冻结预先指定 seed 42 最终 memory adapter 和 reader，先评测未训练 controller，再训练本地 controller 的 SEARCH/FINISH 决策。训练使用两张卡分别放可训练 controller 和冻结 memory。查询依据 question、memory、remaining_searches、previous_queries。FINISH 的答案仍由同一个冻结 reader 生成，避免把回答模型变化混入搜索策略收益。达到搜索上限时强制调用 reader，而不是继续消耗动作。

controller 使用 API/本地共享 JSON Schema，来源合法性仍由环境校验。评测同一 trained controller 搭配未训练/已训练 memory；与 `before-matched` 和 seed 42 最终固定搜索结果组成四组交叉对照。controller 只在已训练 memory 条件下优化，属于顺序训练，不宣称实现联合优化。

## 7. 锁定测试与报告

```bash
research test
research report
```

test 按预注册计划评测最终步的所有三个种子，不按 test 选择 checkpoint 或种子。仅在实现与调参冻结后运行一次。报告保留逐题结果与配对 bootstrap；bootstrap 反映题目抽样不确定性，训练种子波动应单独报告。不要反复查看 test 来改 prompt 或奖励。

所有输出位于忽略提交的 `outputs/`；公开 GitHub 时提交脚本、配置示例与去身份化结果摘要，不提交模型、数据、完整个人路径或 API 凭据。

## 代码入口

- `scripts/research_data.py`：互斥数据划分和指纹。
- `scripts/research_pipeline.py`：固定计划和按阶段执行。
- `scripts/research_eval.py`：匹配环境与本地 adapter 评测。
- `scripts/research_sft.py`：只从训练集提取银标动作。
- `scripts/research_report.py`：汇总和配对统计。
- `src/memsearch/research.py`：离线记忆机制指标，不参与 Agent 观察。
- `src/memsearch/training.py`：约束采样、GRPO/SFT 和训练诊断。
- `src/memsearch/agent.py`：固定检索与记忆驱动搜索、冻结 reader。

## 事实审核与绘图

```bash
uv run --locked --extra train python scripts/audit_memory.py --data data/processed/research-v1/tune --trajectories outputs/research-v1/before/trajectories.jsonl --out outputs/research-v1/grounding-audit.jsonl
uv run --locked --with matplotlib python scripts/plot_research.py --report outputs/research-v1/report.json --out outputs/research-v1/learning-curves.png
```

审核文件的 entailed 初始为 null，必须依据原句人工判断后才能统计。未经校准的 judge 分数不进入训练奖励。
