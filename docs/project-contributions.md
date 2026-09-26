# MemSearch RL：项目背景、个人贡献与实验总结

> 更新日期：2026-09-27。用途：项目复盘、GitHub 项目说明和面试准备。
>
> 证据来源：当前本地代码、已有验证记录，以及运行者在对话中回传的 research-v1 tune/test 结果。本次整理没有重新连接服务器审计全部原始轨迹，也没有重新运行实验。本文中的“我”指项目实施者；面试时应按自己实际理解、参与和能够解释的部分陈述，不把使用现有模型、算法或工具描述为原创发明。
>
> 项目已经完成训练—评测闭环，但不宣称算法已收敛、达到 SOTA 或获得统计显著提升。测试集结果、开发集观察和未实施方案分别说明。

## 1. 背景

多步检索问答需要在连续获得的材料中保留有用信息。直接累积全部历史会增加上下文和模型调用开销；简单截断可能丢失较早出现但对后续推理必要的事实；反复摘要则可能遗漏实体关系、混淆事实或改变原意。

本项目将这个问题具体化为：**Agent 在有限的任务内工作记忆中，应如何处理新证据，并利用已保留的信息决定下一步检索和停止时机？**

选择 HotpotQA 的多跳问答场景，使用 Qwen2.5-7B-Instruct、BM25 检索和带来源的显式记忆，分别训练记忆操作策略与搜索策略。

### 1.1 当前范围

- 任务内记忆：每题开始清空，不跨问题共享用户画像或长期历史。
- HotpotQA candidate-pool 检索：在每题提供的候选文档中搜索，不是 full-wiki 或互联网开放域搜索。
- 最多 3 次搜索，每次最多 2 篇文档；这是短程多步环境，不能称为已经验证长程 Agent 能力。
- reader、memory 和 controller 是逻辑角色；基座可以相同，但训练阶段通过冻结和不同 adapter 区分可训练部分。
- 主线是决策训练与受控评测，不是分布式吞吐优化。

### 1.2 为什么引入预算约束

预算使记忆管理成为真正的信息取舍问题。如果历史可以无限追加，删除、更新、合并和忽略的必要性会减弱。

在有限容量下，策略需要判断：旧信息是否仍有价值、新信息是否值得写入、重复信息如何合并，以及哪些内容可以舍弃。预算也为方法比较提供共同资源条件，并允许研究质量与资源的取舍。

预算是任务约束，不是“必然提高准确率”的技术。相同记忆上限也不等于相同总调用成本；完整提示词、新证据和模型输出都需要另行统计。

## 2. 项目目标／需要达成什么

### 2.1 功能目标

1. 将问答任务转化为可执行、可记录、可训练的 RL 环境。
2. 让模型显式生成 ADD、UPDATE、DELETE、NOOP，而不是由程序替它选择记忆。
3. 在固定 reader 和检索设置下，验证 memory RL 的作用。
4. 冻结一个已训练 memory，训练 SEARCH(query)／FINISH，验证搜索决策的作用。
5. 建立数据准备、训练、checkpoint 评测、部署、消融和结果汇总流程。

### 2.2 实验目标

- 比较未训练操作、摘要、摘录、最近证据、直接 RL、SFT 和 SFT＋RL。
- 区分“模型自主搜索的收益”和“controller RL 的额外收益”。
- 观察不同预算和证据顺序下的行为变化。
- 同时报告任务质量、错误率、搜索次数与全部模型调用 token。
- 用独立 test 检查 tune 上的观察能否复现，保留不支持原假设的结果。

### 2.3 当前完成状态

训练、checkpoint 保存、模型服务、评测和主要研究流程均已跑通。已有三种子的短程 memory RL 和独立 200 题 test；controller、SFT 路径仍为单种子短程实验。配对置信区间尚未提供核验结果，不能将方向一致等同于统计显著。

## 3. 项目难点

| 难点 | 实际遇到的问题 | 对应处理 |
|---|---|---|
| 语言输出不能直接执行 | YAML、尾随代码围栏、非法 JSON、缺失字段 | API／本地共用 Schema，约束生成合法结构 |
| 格式合法不等于动作合法 | 空记忆时删除、把 source ID 当 target ID、引用不可见来源 | 稳定记忆 ID、原子操作、运行时语义校验 |
| RL 没有有效信号 | 所有 rollout 都因格式错误得到相同负奖励 | 保存原始轨迹定位问题，修复格式后检查非零优势与实际参数变化 |
| 约束采样可能与损失不一致 | 生成时屏蔽非法 token，更新时若按原始分布计算概率会失配 | 保存 Schema 和精确 token IDs，更新时重放同一 grammar mask |
| 多步轨迹长度不同 | 每题有不同数量的模型调用和生成 token | 只训练动作 token，按轨迹动作 token 数归一化 |
| 多卡进程信号不一致 | 部分 rank 零优势，另一些 rank 有梯度 | 所有 rank 进入相同梯度归约，依据全局信号决定是否更新 |
| 难以归因效果 | 模型、服务、预算、回答方式同时变化 | 固定角色、匹配后端基线、分阶段训练与交叉对照 |
| 结果容易被高估 | 小规模 tune 提升不一定泛化 | 独立数据划分、多种子、锁定 test、披露负面结果 |

## 4. 怎么实现需求

### 4.1 系统结构

```text
问题 + 当前记忆 + 剩余搜索次数 + 查询历史
                    │
             固定计划 / controller
             ┌──────┴──────┐
        SEARCH(query)     FINISH
             │               │
          BM25 检索       冻结 reader
             │               │
       单轮证据观察上限     答案及引用
             │               │
        memory 策略       答案评分 / 成本统计
             │
    ADD / UPDATE / DELETE / NOOP
             │
      语义校验 + 预算校验
             │
         新的任务内记忆
```

主要代码链路：

- [runner.make_agent](../src/memsearch/runner.py)：装配检索器、memory 和模型后端，支持角色覆盖。
- [Agent.run](../src/memsearch/agent.py)：任务交互循环和终止处理。
- [Memory.update](../src/memsearch/memory.py)：构造可见状态，调用模型并解析操作。
- [apply_operations](../src/memsearch/memory_actions.py)：执行确定性的记忆状态转移。
- [metrics.py](../src/memsearch/metrics.py)：奖励与评测。

reader 不读取完整原始检索历史；答案依据保留的记忆生成，引用必须来自当前记忆。准确地说，当前代码传给 reader 的 observation 还包含查询历史和剩余搜索次数，因此不能描述为输入中严格只有问题与记忆两个字段。

### 4.2 预算如何落地

| 资源 | 主实验设置 | 实现方式 |
|---|---|---|
| 持久记忆 | 512 tokens；机制实验 256/1024 | 指定 tokenizer 计算完整序列化记忆，包括来源、记忆 ID 和 unresolved |
| 单轮新证据 | 1536 tokens | 按句累加文本、ID、标题的计数，超出容量的句子不加入；不是精确的整个 prompt 上限 |
| 检索 | 最多 3 次，每次最多 2 篇文档 | Agent 交互循环和检索配置控制 |
| 训练 prompt／生成长度 | 4096／512 tokens | 参考训练器显式检查 prompt，不静默截断；生成达到上限可能形成无效动作 |

核心不变量：`tokens(serialize(next_memory)) <= memory_budget`。

操作策略超预算时整批拒绝，旧状态不变，轨迹记为 invalid。研究版本摘要也采用 reject；历史默认摘要模式支持 trim，不能混用结果。规则基线通过确定性挑选条目满足预算。

512-token 记忆上限不是整条轨迹的总 token 上限，也不是模型总上下文窗口。完整提示词等开销单独统计。

### 4.3 RL 环境与奖励

- 状态：问题、当前记忆、新证据、可用 ID 和资源信息；controller 另见查询历史与剩余搜索次数。
- 动作：memory 输出一批记忆操作；controller 输出 SEARCH(query) 或 FINISH。
- 转移：检索器返回证据，环境验证并原子执行记忆操作。
- 终止：FINISH、到达搜索上限后调用 reader，或无效动作导致轨迹失败。
- 标签：答案和支持来源仅用于训练奖励或离线指标，不进入 Agent 的可见状态。

当前奖励：

```text
R = I(无轨迹错误) × answer_F1
    − search_cost × 搜索次数
    − 0.1 × I(有轨迹错误)
```

本轮 `search_cost=0`。合法错误答案通常得 0，非法轨迹得 -0.1；没有额外的引用命中奖励、语义蕴含奖励或逐步 credit assignment。最终奖励分配给该轨迹内的可训练角色动作。

因此，搜索次数下降是观测结果，不能声称由显式搜索成本惩罚驱动。来源 ID 命中也不能证明记忆文本真实。

### 4.4 GRPO 训练实现

[training.py](../src/memsearch/training.py) 提供可检查的小模型参考训练器，使用轨迹归一化 clipped GRPO，`beta=0`，没有 critic 和 reference-model KL 正则。

1. `collect_group` 对同一问题采样 4 条完整轨迹，不是只生成 4 次单轮回复。
2. `groups[i]` 保存第 i 条轨迹中可训练角色的 `Sample` 列表；`episodes[i]` 保存结果摘要；`rewards[i]` 保存终局奖励。
3. `Sample` 保存精确 prompt IDs、completion IDs、旧 token log-probs 和采样 Schema。
4. 用组内均值、总体标准差和 epsilon 计算优势：`A_i=(R_i-mean(R))/(std(R)+epsilon)`。
5. teacher forcing 重算生成 token 概率，应用 clipped objective；每条轨迹按其动作 token 总数归一化，再按 group size 平均。
6. 反向传播、跨进程平均梯度、梯度范数裁剪，再执行 AdamW 更新。

`TrainBackend` 就是注入 Agent 的 actor。调用关系为：

```text
collect_group → Agent.run → Memory.update
             → memory.backend.complete → TrainBackend.complete
```

controller 训练时由 `Agent.run` 调用对应的 `complete`。环境 observation 只作为条件，不参与动作 token 的监督损失。

LoRA 配置为 rank 16、alpha 32、all-linear、dropout 0，默认学习率 1e-5；基座采用 BF16，启用 gradient checkpointing。每张训练卡放一个完整副本，通过显式梯度归约同步；不是 FSDP，也不是已经集成 verl 的 rollout 系统。

默认每批 rollout 只更新一次，第一次重算时新旧概率比接近 1，因此不能把 clipping 说成本轮训练效果的主要原因。零优势组可能跳过本地反向传播，但仍参与多卡同步。

## 5. 整体工作 SOP

### 5.1 先建立可运行基线

1. 使用 uv、锁文件与固定模型路径配置环境。
2. 下载 Parquet 数据，检查行数，转换数据，排除支持句索引不合法的样本。
3. 验证 Task 与 Labels 隔离、候选文档及支持来源 ID 完整。
4. 启动冻结 reader，以少量 smoke 题检查协议、引用和服务。
5. 先解决格式失败与零优势问题，再核验梯度和 checkpoint；不能把保存了 adapter 当成已发生有效训练。

### 5.2 建立研究版本

1. 排除曾反复调试的官方 dev 前 200 题。
2. 固定随机划分：train 2000、tune 80、test 200；按 ID 和规范化问题去重，不宣称已完成语义近重复检测。
3. 保存 plan、数据指纹、种子、配置及 checkpoint。
4. 运行操作、摘要、摘录、最近证据基线，以及同后端的 `before-matched`。
5. 三个种子分别训练 12 次迭代，第 6／12 步评测学习曲线。
6. 在 tune 上做预算、顺序、银标 SFT 和 controller 对照。
7. 固定方案后运行独立 test；不按 test 选择最好种子或改动奖励。
8. 汇总逐题指标、失败轨迹、成本与统计不确定性，公开去身份化的结果和复现入口。

本轮 memory RL 每个种子使用两张卡，共三个并行实验；每个种子 24 个训练题次、96 条轨迹，总计 288 条。2000 是训练池大小，不是已完整训练的题数。12 次迭代不等于 12 次有效参数更新。

冷启动：12 步 SFT → 12 步单进程 GRPO。搜索策略：冻结预先指定的直接 RL seed 42 memory，第 12 步 checkpoint，训练 controller 4 步。最终 controller 系统没有使用 SFT＋GRPO memory。

### 5.3 可复现入口及运行边界

| 文件 | 用途 |
|---|---|
| [research_data.py](../scripts/research_data.py) | 独立数据划分、排除旧 dev、记录指纹 |
| [research_pipeline.py](../scripts/research_pipeline.py) | setup、baselines、train、curves、mechanisms、coldstart、controller、test、report |
| [research_eval.py](../scripts/research_eval.py) | 统一预算和解码条件，支持本地 adapter 或 LoRA API |
| [research_sft.py](../scripts/research_sft.py) | 只从训练轨迹提取银标动作 |
| [research_report.py](../scripts/research_report.py) | 结果、错误、种子波动与配对统计汇总 |
| [plot_research.py](../scripts/plot_research.py) | 生成学习曲线图 |
| [audit_memory.py](../scripts/audit_memory.py) | 导出陈述与原始来源供事实审核 |
| [serve_research.sh](../scripts/serve_research.sh) | 启动支持多个 LoRA 的本地评测服务 |

详细步骤见 [研究流程](research-workflow.md) 和 [训练教程](training-guide.md)。主 `test` 入口当前只覆盖原始基线与三种子 memory RL，SFT 和 controller 的 test 是额外显式运行的，不能声称一条主命令已自动覆盖所有扩展。

输出默认禁止覆盖。当前不是完善的自动断点恢复流水线：服务中断后应保留失败日志，单独恢复未完成组。研究服务与 reader 都需要存活，环境变量在新终端要重新设置。LoRA API 模式缺少变量时可能转入本地加载路径。

## 6. 我做了什么：具体改进与代码落点

### 6.1 从“整段摘要”改成模型自主记忆操作

**问题：** 只有摘要重写，无法直接观察模型对新信息的写入、覆盖、删除和忽略决策。

**实现：** 引入四种显式动作；用稳定的 `m0/m1/...` 标识记忆，来源 ID 与记忆 ID 分开；UPDATE＋DELETE 可以表达合并。环境复制状态后执行，最后检查预算，通过后才提交。NOOP 必须独立，不能混用或悄悄修改 unresolved。

**代码：** `Memory.update`、`apply_operations`、[types.py](../src/memsearch/types.py)。

**贡献边界：** 这是项目的动作建模与环境实现，不是发明了新的 RL 算法。接口能够表达这些行为，不等于已证明模型学会了最优淘汰策略。

### 6.2 修复“无效格式 → 全组同罚 → 零优势”的训练阻塞

**定位过程：** 记录 role、原始回复、finish_reason、token IDs 和解码模式。真实故障中发现回复在长度上限之前以 EOS 结束，但包含 YAML、额外文字或未闭合 JSON，因此不能简单归因于 max_new 太小。

**实现：** API 和本地训练共享 JSON Schema；本地使用 XGrammar 的合法 token mask。结构字段受约束，但不会替模型纠正来源、选择 target 或自动删除条目；语义错误继续交给环境处理。

**代码：** [structured.py](../src/memsearch/structured.py) 的 `memory_schema`、`Grammar.processor`，以及 [backends.py](../src/memsearch/backends.py) 的结构化请求。

**证据：** 历史两步失败试验四条轨迹都得 -0.1，没有更新；修复后在相同两题上出现奖励 [1,0] 的组并发生真实参数更新。它证明训练阻塞被解除，不是独立任务准确率提升实验。

### 6.3 保证约束采样与策略梯度的概率一致

**问题：** 如果只在生成时约束格式，却在训练时使用未约束概率，就不是在优化实际采样的策略分布。

**实现：** `Sample` 保留采样时的精确 token IDs 和 Schema；`Grammar.masks` 按 token 重放约束，`token_logprobs` 在旧策略记录与新策略重算中采用相同合法集合并重新归一化。用 PyTorch 非原地 mask 保持梯度可计算，而不是在 backward 中用外部 CUDA kernel 改写 logits。

**代码：** `TrainBackend.complete`、`token_logprobs`、`Grammar.masks`、`mask_logits`。

**验证：** 测试在线 mask 与重放（包括 EOS）一致、非法 token 梯度为零、masked log-prob 数值一致，以及真实微模型反向传播。

### 6.4 增加训练可观测性，验证参数真的更新

**实现：** 保存每 rank 的奖励、优势、zero_variance、updated、动作明细、valid_fraction、生成长度、梯度范数、概率比、clip_fraction 与近似 KL；按需保存完整 rollout。

这里的近似 KL 是新策略相对旧 rollout 策略的采样统计，主要覆盖参与更新的 token，不是基座 reference KL 正则。

**代码：** `collect_group` 和 `training.main` 的 diagnostics；`average_gradients` 处理零梯度 rank 的同步。

**实测：** 三种子中期到最终 checkpoint 均有 392/392 个 LoRA 张量变化且全部有限；有效更新次数分别为 10、11、8。对应合法轨迹分别为 85/96、85/96、79/96。

### 6.5 修正实验公平性与配置耦合

**问题：** 摘要自动裁剪、操作策略超限失败会混入环境处理差异；不同服务路径也会产生不同输出。

**实现：** 研究版统一 LLM 记忆的 reject 策略，固定预算 tokenizer；补充 `before-matched`；配置化模型路径、服务地址、实验 ID；检测 plan 漂移并避免覆盖历史产物。

**代码：** `Memory(..., overflow="reject")`、`runner.make_counter`、训练侧使用配置中的 counter、[experiment.py](../scripts/experiment.py)、`research_pipeline.py`。

**观察：** test 上 `before` 的 F1 为 35.35%，`before-matched` 为 36.75%。因此直接 RL 的主要对照使用后者。部分机制实验的 before/after 仍有服务路径差异，尚不能宣称所有对照都已完全消除后端混杂。

### 6.6 从只看最终分数，改为记录记忆演化与覆盖

**实现：** 每步保存 memory_before、memory_after、操作提案、是否提交、检索文档和证据 ID；离线计算来源召回、丢失和新增的支持来源、实际记忆大小。

**代码：** `Agent.run`、[research.mechanism](../src/memsearch/research.py)、`runner.evaluate`。

**作用：** 可以区分“未观察到来源”“来源未保留”“存在协议错误”等现象，为逐条定位失败提供依据。但来源 ID 覆盖只是代理指标，不能自动判定事实是否被忠实表达。

### 6.7 构建可追溯的 SFT 冷启动对照

**实现：** 只允许训练任务 ID，从完整合法且答案 F1≥0.5 的训练轨迹提取 memory 调用；去重、检查长度，训练 SFT，再从该 adapter 继续 GRPO。

**代码：** `research_sft.py`、`training.main` 的 SFT 分支与 `--adapter`。

**边界：** 这是经过结果筛选的自生成银标，未逐条验证蕴含，也不是专家示范；各训练路径计算量不完全相同。结果显示 SFT＋GRPO 优于 SFT，但在 test 上没有超过直接 RL，因而不能把冷启动描述为已确认的最佳方案。

### 6.8 从固定检索扩展到记忆驱动的搜索决策

**原流程：** 对问题做一次 BM25 排序，分窗口依次取文档，后续顺序与 memory 无关。

**改进：** controller 根据记忆、查询历史和剩余次数输出 SEARCH(query) 或 FINISH；查询可随新信息变化，达到上限时由 reader 回答。FINISH 不再让 controller 自己生成最终答案，减少回答模型变化对搜索效果的干扰。

**代码：** `Agent.run` 的 model policy、[prompts.py](../src/memsearch/prompts.py) 的 `FROZEN_READER` 分支、`controller_schema`、训练参数 `--frozen-memory-adapter`。

**实验设计：** 冻结直接 RL seed 42 最终 memory，先测试未训练 controller，再训练 controller；tune 上另外测试已训练 controller＋未训练 memory。属于分阶段训练，不是联合优化。

**边界：** 查询改写与停止时机目前同时变化，尚未分别消融；不能说已经确定是哪一个因素贡献了全部收益。

### 6.9 做预算与顺序实验，保留不利结果

**实现：** 改变预算，并对同一 BM25 排名结果执行反转，不利用 gold 选择顺序；所有条件使用固定 checkpoint。

**代码：** `research_pipeline.py` 的 mechanisms、`Agent.run` 的 `evidence_order`。

**发现：** 512-token 反转时，操作式记忆比摘要下降更明显。但观察截取和提前失败也改变了实际证据覆盖，尚未分离纯顺序效应。

**当前没有做：** 随机顺序增强训练、长程噪声环境、顺序鲁棒性修复。因此能说“发现顺序敏感现象并设计了诊断”，不能说“已经解决顺序敏感性”。

### 6.10 降低复现成本并整理发布边界

**实现：** uv 环境、锁文件、脚本化下载/准备/训练/评测；LoRA 动态加载减少反复合并；adapter 名称由权重及配置内容哈希生成；日志、模型和本地配置不进入 Git。

**代码：** [lora_api.py](../src/memsearch/lora_api.py)、`serve_research.sh`、`research_eval.py`、`.gitignore`。

**当前缺口：** 评测元数据仍可能把 SFT adapter 统一标成 RL，或把未训练 controller＋已训练 memory 标成 untrained。结果需按实际角色与 adapter 解释。参考训练器会保存 optimizer，但 `--adapter` 是加载权重并重建 optimizer，不是精确断点续训。

本地还有有界并发评测与调用隔离测试；此前尚未同步到服务器，不把它描述为本轮实测的吞吐提升。历史记录为服务器 47 项测试通过，本地 48 项中 1 项因 Linux XGrammar 环境跳过；不是本次重新测试的结论。

## 7. 可证明的结果

### 7.1 独立 test：200 题完整结果

下面是运行者回传结果。EM/F1/invalid 均转换为百分比；搜索次数是每题均值。

| 目录／设置 | EM | F1 | invalid | 搜索次数 |
|---|---:|---:|---:|---:|
| before，原服务基线 | 28.50 | 35.35 | 9.50 | 2.840 |
| before-matched，同后端基线 | 30.50 | 36.75 | 8.50 | 2.835 |
| after-42，直接 memory RL | 34.00 | 40.47 | 5.00 | 2.905 |
| after-43，直接 memory RL | 32.00 | 39.59 | 6.50 | 2.870 |
| after-44，直接 memory RL | 32.00 | 39.17 | 7.00 | 2.865 |
| extractive，摘录 | 30.00 | 38.12 | 0.00 | 3.000 |
| recent，最近证据 | 7.00 | 9.82 | 0.00 | 3.000 |
| summary，普通摘要 | 35.50 | 43.77 | 0.50 | 2.990 |
| sft | 29.50 | 35.68 | 5.50 | 2.905 |
| sft-grpo | 31.00 | 37.32 | 5.00 | 2.905 |
| controller-untrained＋RL memory | 39.00 | 47.16 | 4.50 | 1.665 |
| controller-trained＋同一 RL memory | 40.00 | 49.13 | 4.50 | 1.610 |

可支持的比较：

1. **直接 memory RL：** 三种子 F1 均值 39.74%，相对同后端基线提高约 3.00 个百分点；三个种子方向一致，但仍低于摘要约 4.03 个百分点。
2. **冷启动后继续 RL：** SFT＋GRPO 相比 SFT 提高 1.64 个 F1 百分点、1.50 个 EM 百分点；不能证明冷启动优于直接 RL。
3. **动态搜索：** 在相同 seed 42 memory 下，未训练 controller 相比固定搜索提高 6.69 个 F1 百分点，平均搜索次数减少 42.69%。这是查询和停止机制共同变化的比较。
4. **controller RL：** 相同 memory 下，F1 从 47.16% 升至 49.13%，提高 1.97 个百分点；搜索次数从 1.665 降至 1.610，减少 3.30%。单种子结果尚无已核验配对置信区间。
5. **整体系统：** 最终组合比固定搜索＋摘要提高 5.36 个 F1 百分点，搜索次数减少 46.15%；同时 invalid 从 0.50% 升至 4.50%。这不是“RL 单独贡献 5.36 个百分点”的证据。

### 7.2 controller 的调用成本

| 条件 | 输入 token | 输出 token | 总 token |
|---|---:|---:|---:|
| 未训练 controller＋RL memory | 671,612 | 42,213 | 713,825 |
| RL controller＋同一 memory | 653,510 | 41,159 | 694,669 |

总 token 减少约 2.68%，来自全部模型调用统计。没有提供完整延迟、吞吐或 GPU 时长对照，不能换算成已证明的同等比例时延、费用或算力节省。

### 7.3 tune：机制与冷启动观察

| 条件 | 未训练操作 F1 | 摘要 F1 | seed 42 RL 操作 F1 |
|---|---:|---:|---:|
| 512，正常顺序 | 45.13（原服务）／41.29（matched） | 52.27 | 45.63 |
| 256，正常顺序 | 本文未收录回传值 | 本文未收录回传值 | 46.63 |
| 1024，正常顺序 | 47.96 | 53.52 | 54.96 |
| 512，反转顺序 | 34.29 | 47.87 | 32.11 |

- 1024 下的 RL 策略单次超过摘要 1.44 个 F1 百分点，但不能作为独立 test 或稳定优势结论。
- 同一个 RL checkpoint 在 512 下反转后 F1 降低约 13.52 个百分点；摘要降低约 4.40 个百分点。
- 反转时操作组检索支持召回为 76.25%，摘要为 81.04%，说明实际观察覆盖不完全相同，纯顺序机制仍需进一步控制。
- tune 上 SFT／SFT＋GRPO 的 F1 为 48.88%／50.20%，但 test 为 35.68%／37.32%。不同集合不能直接用差值判定过拟合；真正需要保留的结论是：它们在 test 上未超过直接 RL 的均值。
- tune 上 controller RL 的 F1 从 50.46% 降至 49.83%，test 则从 47.16% 升至 49.13%，提示存在抽样和训练波动，不能只选择有利集合报告。

### 7.4 统计与证据边界

`paired_bootstrap` 已实现，但当前对话没有提供针对关键对照的置信区间。特别是 controller-trained 应与 controller-untrained 配对，不能用报告默认的 before 比较替代。

建议核验的配对：before-matched 对各 after；sft 对 sft-grpo；after-42 对 controller-untrained；controller-untrained 对 controller-trained；summary 对最终组合。

不支持的表述包括：统计显著、稳定超过摘要、已解决长期记忆或顺序鲁棒性、引用即真实、SFT 路径最好、模型已充分收敛、已完成大规模 RL 训练。

## 8. 最终效果

### 8.1 已完成的工程交付

形成了可运行的预算约束 Agent 环境、显式 memory 操作协议、与约束采样一致的参考 GRPO 训练器，以及多种子、冷启动和搜索策略的研究流程。在 8×3090 资源下实际完成训练和独立评测，保留原始轨迹、checkpoint、配置与错误信息。

### 8.2 已得到的算法结论

- 格式约束和概率一致性修复让原本无有效梯度的链路能够产生有效更新。
- 直接 memory RL 在三个种子的 test 上均优于匹配的未训练操作策略，但尚不如摘要。
- 当前最明显的改善来自记忆驱动的动态搜索；controller RL 在 test 上有额外的小幅收益。
- 已识别顺序敏感、语义非法动作、冷启动优势未泛化等边界；这些问题尚不能被描述为已解决。

### 8.3 面试中可以使用的项目概述

> 我实现了一个预算约束下的多步检索问答 Agent，希望研究：在只能保留有限信息的情况下，模型如何管理证据，并决定下一步搜索什么、什么时候停止。
>
> 系统分成三个逻辑角色：**controller 是搜索决策模型**，根据问题、当前记忆、历史查询和剩余搜索次数，输出 SEARCH(query) 或 FINISH；实际检索由 BM25 执行。**memory 是记忆管理模型**，在新证据到来后，通过 ADD、UPDATE、DELETE、NOOP 更新带来源的记忆，主实验要求完整记忆不超过 512 tokens。**reader 是最终回答模型**，在搜索结束后，根据保留的记忆生成答案及来源引用，不负责搜索或修改记忆。
>
> 三个角色使用同一系列的 7B 指令模型，通过不同提示词、调用入口和 LoRA adapter 区分，并不是三个都从零训练的模型。训练采用分阶段方式：先固定检索计划和 reader，用 GRPO 训练 memory；再冻结已训练的 memory 和 reader，训练 controller 的查询与停止决策。reader 的模型参数始终冻结，以减少回答模型变化对实验归因的干扰。
>
> 实施过程中，我定位了非法结构化输出导致奖励退化的问题，通过共享 Schema、约束解码与概率掩码重放，保证采样和训练目标一致，并用真实参数变化验证更新。随后通过同后端基线、三种子 memory RL、预算和顺序实验、冷启动对照及独立测试分析收益。200 题 test 上，memory RL 相对匹配基线平均提升约 3 个 F1 百分点；同一 memory 下，controller RL 带来约 1.97 个 F1 百分点改善和约 3.30% 的搜索次数下降。整体系统仍有协议错误和顺序敏感性，统计显著性与更长任务泛化需要继续验证。

**角色与训练状态速查：**

| 模块 | 核心职责 | memory 训练阶段 | controller 训练阶段 |
|---|---|---|---|
| controller | 决定查什么、何时停止；不直接生成最终答案 | 不启用，由固定检索计划替代 | 更新 controller LoRA |
| memory | 决定写入、更新、删除或忽略什么信息 | 更新 memory LoRA | 冻结指定 memory checkpoint |
| reader | 根据保留的记忆生成最终答案和引用 | 冻结 | 冻结 |
| BM25 | 执行查询并返回候选文档 | 固定算法 | 固定算法，接收 controller 生成的查询 |

上述 controller 职责对应本轮 `controller_reader=True` 的实验协议。代码还保留其他模式，不能把旧模式中 controller 可输出答案的接口混入这轮实验说明。reader 的实际 observation 还包含查询历史和剩余次数，详见第 4.1 节；“依据记忆回答”不意味着输入严格只有记忆字段。

### 8.4 接下来应做、但不能计入当前贡献的工作

1. 完成逐题配对统计、元数据修正和原始产物审计。
2. 增加 controller 随机种子与更充分的训练曲线。
3. 在实际观察集合受控的条件下，验证随机顺序训练能否改善鲁棒性。
4. 分离 query 改写与停止决策的收益。
5. 增加完整上下文参考和更长证据序列，测量预算带来的质量—成本关系。
6. 校准事实验证器后，再考虑语义忠实性奖励；目前没有这项已验证贡献。

现有 200 题 test 已被查看，下一轮新方法研究应另设锁定测试集。已有结果保留为历史回归，不围绕它反复调整后再声称独立验证。
