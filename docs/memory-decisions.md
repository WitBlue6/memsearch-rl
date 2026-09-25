# RL 训练什么记忆决策

旧版 `structured` 让模型整体重写记忆，再由代码自动裁剪，不能等价描述为显式记忆管理。现在默认 `decision`：模型生成操作，环境执行操作，最终任务奖励优化操作策略。

## 一轮决策

```text
问题 + 旧记忆 + 新检索证据 + 预算/当前大小
  → memory policy 采样操作 JSON
  → 环境校验并原子执行
  → 新记忆（下一轮不重放已丢弃证据）
  → 继续检索/记忆决策 → 冻结 reader 回答
  → 答案 F1 / 无效动作惩罚
  → 同题多条轨迹的相对优势 → 更新 memory policy
```

比如 m0 保存“Ada 出生于 Luma”，新证据 c1:0 说明“Luma 位于 Norvia”。策略可以更新旧条目，把两跳关系合并成一条记忆：

```json
{"operations":[{"op":"UPDATE","target_id":"m0","text":"Ada was born in Luma, which is in Norvia.","source_ids":["p1:0","c1:0"]}]}
```

也可以 ADD 新条目；若证据无关则 NOOP；空间不足时选择 DELETE 哪个旧条目，或通过 UPDATE 压缩。合并两个旧条目用 UPDATE 保留一个 ID，再 DELETE 另一个。操作列表中任一操作无效或最终状态超预算，整批不提交，轨迹记录错误并终止。环境不会自动删除条目来使动作成功。

## 按文件读

1. `types.py`：MemoryItem 的 memory_id 区分记忆条目，source_ids 指向证据；next_id 防止删除后的 ID 被复用。
2. `prompts.py`：memory_decision 约定观察和动作 JSON。`memory.py` 只负责构造观察、调用策略、保存提案，默认模式 decision。
3. `memory_actions.py`：apply_operations 是纯环境转换函数，不调用模型，不决定哪些事实重要。
4. `agent.py`：检索后给策略一次操作机会；保存 memory_before、memory、memory_decision。固定检索阶段，reader 只能看到保留下来的记忆，没有原始历史检索回放。
5. `training.py`：TrainBackend 用可训练模型采样操作，并保留原始 completion IDs 和旧 log-prob。collect_group 收集同题多条轨迹；group_advantages 由最终奖励计算优势；token_logprobs / clipped_objective 对操作 JSON 的全部生成 token 计算损失。执行环境返回的状态和证据只作为输入。
6. `metrics.py`：答案奖励、无效轨迹惩罚、已执行操作计数。ADD 本身没有人为设定的正奖励，NOOP 也不必然受罚；有用与否由最终任务表现决定。
7. `tests/test_memory_decisions.py`：覆盖操作、预算拒绝、原子回滚、证据遗忘，以及同一证据下 ADD/NOOP 影响奖励的链路。

## 怎样验证确实在学习决策

先固定检索顺序和 reader，仅训练 memory；同一道题采样多种操作轨迹，以最终答案奖励区分。无效输出不参与正确性得分并受罚。若所有轨迹奖励相同，组内优势为零，不会产生该组的策略梯度。

检查训练日志中的 rewards、advantages、zero_variance、updated，以及 episodes 内的 memory_decisions 和 memory_operations。评测轨迹另外包含操作前后状态和所有模型调用。重点观察预算压力下 UPDATE/DELETE/NOOP 的使用，而不能只看 JSON 是否能生成。

冷启动教师轨迹必须输出当前动作协议。默认训练入口为：

```bash
TRAIN_DATA=data/processed/train RUN_OUT=outputs/memory-decision-grpo \
  bash scripts/train_memory.sh --memory-mode decision --steps 100 --group-size 4
```

需要真实数据与冻结模型服务；不能把规则 demo 直接当 RL 结果。旧的 summary / structured 模式仍可通过 --memory-mode 做消融。

当前还不是统一动作空间的联合 Agent RL：记忆决策机会固定在每次检索之后，第一阶段只训练 memory，第二阶段冻结 memory 训练搜索/停止/回答。没有步骤级信用分配，所有记忆操作共享整条轨迹的相对优势；任务内记忆在每题开始清空。尚未完成正式 GPU RL 训练和效果评测。
