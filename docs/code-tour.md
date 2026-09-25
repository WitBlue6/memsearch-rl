# 代码导读

这个项目研究两件事：在有限记忆预算内保留有用证据，以及根据已有证据决定下一次搜索或结束回答。当前是可运行的研究原型：环境、评测与参考训练器已经实现，正式数据上的算法效果尚未验证。

## 一次评测如何运行

```text
uv run --locked memsearch eval ...
  cli.main()
    runner.load_config()       读取 TOML、检查预算
    runner.evaluate()
      data.load_dataset()     分开读取公开任务和私有标签
      runner.make_agent()     组装检索器、记忆、模型后端
      agent.Agent.run(task)   只传任务，不传答案标签
        controller            输出 SEARCH 或 FINISH
        retrieval.BM25.search()
        memory.Memory.update()
          decision: memory_actions.apply_operations()
          旧摘要基线: budget.enforce_budget()
        重复直到回答或到达预算
      metrics.score()         此时才使用标签评分
      写出轨迹、逐题结果、汇总、配置和数据指纹
```

## 数据和状态

`src/memsearch/types.py` 是接口定义：

- `Document`：文档标题、句子列表，通过 `evidence()` 生成带 ID 的证据。
- `Evidence`：`文档ID:句号索引`、原文和标题；标题只作观察的一部分。
- `Task`：问题及候选文档 ID，不包含标准答案。
- `Labels`：标准答案与支持证据，只有奖励和评测读取。
- `MemoryItem` / `MemoryState`：事实、来源 ID、稳定 memory_id、未解决问题和分配 ID 的 next_id。
- `Action`：SEARCH 查询，或 FINISH 答案及引用。
- `Call` / `Episode`：模型调用与一整道题的执行记录。

`data.py` 负责 JSONL 读写、数据校验、数据指纹和 HotpotQA 转换。处理后的数据文件为 `corpus.jsonl`、`tasks.jsonl`、`labels.jsonl`，来源元数据另存 `dataset.json`。

## Agent 控制循环

`agent.py` 的 `Agent.run()` 是理解系统最好的起点。控制器可见问题、当前记忆、剩余搜索次数和历史查询；没有自动重放全部历史检索文本。每题开始时记忆清空。

`policy="fixed"` 时，用原始问题得到固定排序，每轮取下一个窗口；这一计划不依赖记忆内容。达到固定轮数后调用 reader 回答。

`policy="model"` 时，controller 自己生成查询，或者直接输出最终答案。因此这个分支的回答也是 controller 生成；并没有在 FINISH 后额外调用同一个独立 reader。训练阶段二实际同时优化搜索、停止与回答生成，这一点需要在实验解释中说明。

`parse_action()` 校验协议，`validate_citations()` 拒绝不在当前记忆中的引用。每次 SEARCH 都扣预算，包括重复搜索和之后发生记忆格式错误的情况。格式错误终止该轨迹并记录原因；网络服务故障会向外抛出，避免悄悄被当成模型表现。

## 记忆更新

`memory.py` 的 `Memory.update(question, old, evidence)` 接收问题、旧记忆和新证据，返回新的记忆状态。

| mode | 行为 | 用途 |
|---|---|---|
| recent | 新证据在前，保存原句并去重 | 简单近期记忆基线 |
| extractive | 按问题词重叠度排序原句 | 不调用 LLM 的摘录基线 |
| summary | 请求模型生成普通摘要，仍保留来源字段 | 摘要对照组 |
| structured | 请求模型整体重写事实、来源与待解决子问题 | 旧版结构化摘要对照 |
| decision | 策略输出 ADD / UPDATE / DELETE / NOOP 及参数 | 默认 RL 记忆决策模块 |

summary 和 structured 共用输出 schema 和来源检查，但提示词不同。结构化模式本身不意味着经过 RL，是否训练取决于它调用的 checkpoint。

`decision` 模式由 `memory.py` 构造观察，模型生成操作列表，`memory_actions.py` 的 `apply_operations()` 校验来源、目标 ID 和预算，全部合法才一次提交。ADD 创建稳定 ID；UPDATE 保持 ID；DELETE 删除指定条目；NOOP 保持状态。合并由 UPDATE + DELETE 表达。超预算拒绝整批操作并终止该轨迹，没有程序自动淘汰。

`budget.py` 对完整记忆 JSON 计数，包含 schema、来源 ID、memory_id、next_id 和未解决问题。只有旧版 recent/extractive/summary/structured 基线继续自动裁剪末尾条目。当前每次检索后提供一次决策机会；策略可 NOOP，尚未学习何时调用记忆模块。

`agent.py` 将更新前后状态、模型提案及 applied 状态写入步骤记录。失败保留旧状态，原始模型输出仍在 calls 中。详见 [记忆决策链路](memory-decisions.md)。

来源存在只证明模型引用了看过的 ID，不证明生成事实被原文支持；语义蕴含仍需要独立评测。

## 检索、提示词和模型调用

`retrieval.py` 实现小规模内存 BM25，以词频和逆文档频率为文档排序。默认只在每题候选文档中搜索；corpus 模式搜索全部导入文档。它不是向量检索，也不是搜索互联网。

`prompts.py` 分别定义 memory_decision、memory、summary、controller 和 reader 的系统提示词与 JSON 输出要求。想研究“未解决子问题”或不同记忆表示时，从这里和 memory.py 一起修改。

`backends.py` 的 `ChatBackend.complete()` 请求 OpenAI-compatible HTTP 服务。记录角色、输入、输出以及 token 开销；密钥只读环境变量。没有服务 usage 时使用显式标记的估计计数。

`demo.py` 的 `DemoBackend` 是针对虚构事实的规则程序，保证没有 GPU 和模型 API 时也能运行协议测试。它不是大模型，也不能代表算法性能。

## 评测与四组消融

`runner.py` 负责配置、后端、Agent 组装与逐题执行。memory/controller/reader 可指向不同服务或 checkpoint，因此可以冻结一个模块、替换另一个模块。

`cli.py` 提供：

- `eval`：单配置评测。
- `matrix`：A 普通摘要＋固定搜索；B 显式记忆决策＋固定搜索；C 普通摘要＋模型搜索；D 显式记忆决策＋模型搜索。
- `prepare-hotpot` / `validate-data`：转换与校验数据。
- `export-sft`：筛选表现合格的轨迹，导出某角色的 prompt/completion。

Demo 中 A/C 用 recent 替代摘要以免依赖模型。真实 matrix 中，A/C 默认回到原始模型生成摘要，避免误用已训练 memory checkpoint。C/D 使用同一个控制器，所以 C 属于交叉消融；若要分别训练搜索基线，需要另一次训练与评测。

`metrics.py` 实现答案 EM/F1、证据与引用的标注匹配、搜索次数、无效输出、按计量单位分别汇总的模型成本。token 统计包括记忆模型，不能只挑 controller 的开销。

奖励函数 `reward()` 默认是合法轨迹的最终答案 F1，减去可配置搜索成本与无效输出惩罚；无效轨迹的正确性项置零。`group_advantages()` 在同一道题的多条轨迹之间标准化奖励，组内奖励完全相同时优势为零。

## 训练实现

`training.py` 可以通过 `uv run --locked --extra train memsearch-train` 启动。

- `load_policy()`：加载基础模型与 LoRA；可加载已有适配器继续优化，优化器状态从头开始。
- `TrainBackend.complete()`：用本地可训练模型生成输出，保存准确的 prompt IDs、completion IDs 和旧策略 log-prob。它替换普通 API 后端，使同一个 Agent 环境可用于训练。
- `collect_group()`：在同一问题上收集多条操作轨迹、最终奖励和各次生成的原始 token，记录操作提案及是否执行。
- `token_logprobs()`：只计算模型生成部分的 token 概率，观察／检索文本仅作为输入。
- `clipped_objective()`：计算 clipped policy objective。
- `average_gradients()`：显式跨进程平均梯度，允许不同 rank 有不同数量的有效决策步骤。
- `save_checkpoint()`：保存 LoRA、tokenizer 和 optimizer 文件；CLI 尚未提供精确断点恢复。

SFT 路径逐条训练导出的成功轨迹。GRPO 路径对每题生成一组完整 episode，以最终回答奖励计算组内优势，再将该 episode 的优势分配到所训练角色的各次生成上。每条轨迹的损失按该角色生成的 token 数归一化；当前没有步骤级奖励或细粒度信用分配。

默认 `--memory-mode decision`，memory 角色生成的操作名、目标条目、文本、来源和可选 unresolved 都参与策略 loss；环境产生的新状态不参与 loss。训练 memory 时自动使用固定检索与冻结 reader；训练 controller 时记忆模型从配置的 API 服务读取并保持冻结。参考实现采用 T=1、无 top-k/top-p 截断、KL 系数 0；每个进程加载一个完整基础模型和 LoRA，未做 FSDP 或高吞吐 rollout。

`verl_reward.py` 仅为最终答案评分的可选接口，完整 verl AgentLoop 尚未接入。

## 配置与启动脚本

- `configs/demo.toml`：虚构数据和规则后端。
- `configs/api.toml`：真实模型端点、模型名、tokenizer、证据／记忆预算和搜索次数。
- `configs/accelerate_8x3090.yaml`：单机多进程 bf16 基础配置。
- `scripts/train_memory.sh`：uv 启动阶段一；默认六卡训练，留两卡运行固定模型服务。
- `scripts/train_controller.sh`：uv 启动阶段二，必须给出冻结记忆模型配置。
- `scripts/merge_adapter.py`：合并 LoRA 权重，以便单独部署服务。
- `scripts/make_tiny_training_fixture.py`：生成随机微模型，仅用于 CPU 集成测试。
- `scripts/check_distributed.py`：验证不同本地 backward 次数下的同步。

## 环境与测试

`.python-version` 固定 Python 3.13；`pyproject.toml` 声明包、CLI 和可选 train 依赖；`uv.lock` 锁定解析版本；`.venv` 由 uv 创建，不进 Git。

`tests/test_core.py` 检查标签隔离、证据引用、预算、检索公平性、数据转换与产物；`tests/test_api.py` 检查接口 usage 与凭据处理；`tests/test_training.py` 检查生成 token loss、clipping、梯度更新、采样 log-prob 和 LoRA 保存。

建议阅读顺序：types.py → agent.py → memory.py / prompts.py → metrics.py → training.py → runner.py / cli.py。先理解一条 episode 的数据流，再看它如何变成训练样本。
