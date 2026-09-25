# 8×3090 服务器准备

本地开发机为 Apple Silicon；GPU 实验在用户的服务器进行，目前尚未连接该服务器。

## 首次接入需要确认

- 已配置的 SSH 别名或服务器上的工作目录，不发送密码和私钥。
- `nvidia-smi` 的驱动版本、空闲显存和 GPU 占用。
- `nvidia-smi topo -m`、系统内存与可用磁盘；不要假设八张卡有全互联。
- uv（至少 0.7.9）、Python 3.13、网络／模型缓存位置，以及允许使用的卡。

## uv 环境

在项目根目录执行 `uv sync --locked --extra train`。uv 按 `.python-version` 选择 Python 3.13，必要时下载 managed Python，并创建项目 `.venv`。无需 `source activate`。先执行 `uv run --locked --extra train python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`。

训练脚本已经调用 `uv run --locked --extra train accelerate`。锁文件发生变化时先审查依赖差异；驱动兼容性确认后再启动训练。

## 资源安排

默认脚本把 GPU 0–5 用于六个策略副本，GPU 6–7 留给冻结 reader／memory 服务。实际配置按可用卡调整。若固定模型在远程服务，训练可使用八卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRAIN_PROCESSES=8 \
  TRAIN_DATA=data/processed/train RUN_OUT=outputs/memory-8gpu \
  bash scripts/train_memory.sh --steps 10
```

起步 1.5B、LoRA rank 16、记忆 512、观察证据 1536、每题 4 条 rollout；这些是待 profiling 的配置，不是 24GB 显存保证。单卡 smoke 只需：

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_PROCESSES=1 \
  TRAIN_DATA=data/processed/train RUN_OUT=outputs/memory-smoke \
  bash scripts/train_memory.sh --steps 2 --group-size 2 --max-new 256
```

每个 rank 一题的多个 rollout 顺序生成，梯度按轨迹生成 token 数归一化再跨 rank 平均。长轨迹需要更久；先看显存、有效输出率和奖励方差，不要直接开启长时间训练。

阶段一默认训练 `decision` 操作协议。阶段二的冻结 memory 服务必须部署相同协议的 checkpoint；旧版只生成 `items` 的摘要 checkpoint 不兼容。SFT 数据也须使用操作格式重新生成。

## 冻结与复现

服务器环境通过 smoke 后保存 `uv.lock`、`uv pip freeze --python .venv/bin/python`、GPU/驱动信息、数据哈希、模型 revision、代码 commit 和随机种子。训练与服务依赖分环境安装。配置中的 API key 只读取环境变量。

当前 checkpoint 包含 adapter、tokenizer 和 optimizer 文件；CLI 只支持 adapter warm start，不支持恢复 RNG／数据进度的精确续训。

多 rank 中任一服务故障应终止并检查日志；当前未实现分布式故障恢复。完整 verl AgentLoop 迁移列为后续工作，先固定环境协议和奖励。
