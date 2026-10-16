# 联合实验复现准备

代码不包含原始 HDF5、分支轨迹和 checkpoint。YAML 中的历史运行路径用于对应实验报告，需替换为实际输出；缺少这些文件时不能直接运行联合训练。

## 1. 配置仿真与采集

安装 `scripts/requirements.txt`；UR5-WSG 模型随仓库提供，GPU/Vulkan 必须可用。气垫球资产及专家协议见 [资产说明](air_hockey_assets.md) 和 [专家流水线](air_hockey_expert_pipeline.md)。

复制 `env_cfg/task_config/air_hockey_clean.yml` 为新的任务配置，设置目标 episode 数、`save_freq: 20`、`video_fps: 20`，保留物理配置及相机标定。学习模块要求 0.05 秒的状态/RGB对齐，不可直接使用默认 30 Hz 数据。

```bash
bash collect_data.sh air_hockey <new_task_config_name> 0
```

正式导出目录由任务配置与 embodiment 决定。将模型 YAML 的 `data_root` 指向实际 HDF5 目录。实验使用 100 个不同种子，按 70/15/15 划分；同种子的所有窗口和分支必须留在同一划分。

## 2. 前置模型依赖

```text
原生 HDF5 → RGB V1 → RGB V4
                         ↓
                        H1 → 训练/验证/测试分支 → H3 → P2 BC
                                                         ↓
                                                混合动作分支 → J1
```

| 阶段 | 命令/配置 | 执行前要更新 |
| --- | --- | --- |
| RGB V1 | `train_world_model_rgb.py --config env_cfg/world_model_rgb.yml` | `data_root` |
| RGB V4 | `train_world_model_rgb.py --config env_cfg/world_model_rgb_v4.yml` | `data_root`、`initial_checkpoint` 指向 V1 |
| H1 | `train_world_model_hit.py --config env_cfg/world_model_hit.yml` | `backbone_checkpoint` 指向 V4、`data_root` |
| H3 | `train_execution_model.py --config env_cfg/world_model_hit_v3.yml` | V4 checkpoint 与三种分支目录 |
| P2 | `train_behavior_policy.py --world-checkpoint <H3_checkpoint>` | 实际 H3 checkpoint |
| J1 | `train_joint_policy.py --config env_cfg/joint_air_hockey.yml` | `source_checkpoint` 指向 H3、补采分支目录 |

以上脚本均位于 `scripts/`，训练命令加 `python scripts/` 前缀及 `--device cuda`。H3 提供数据划分和配置来源；J1 固定视觉前端实际加载的是其中绑定的 V4 checkpoint，并非继续训练 H3 的头。

H1 同状态分支分别采集，不能混用划分：

```bash
python scripts/validate_hit_counterfactual.py --checkpoint <H1_checkpoint> --split train --episodes 15
python scripts/validate_hit_counterfactual.py --checkpoint <H1_checkpoint> --split val --episodes 5
python scripts/validate_hit_counterfactual.py --checkpoint <H1_checkpoint> --split test --episodes 15
```

将三次输出填入 H3 YAML 的 `train_branch_root`、`val_branch_root`、`test_branch_root`。H1 性能报告不作为训练 J1 的验收前提。

## 3. 补采 J1 分支

```bash
python scripts/collect_joint_branches.py \
  --policy-checkpoint <P2_checkpoint> --episodes 20 --actions mixed
```

每个训练状态请求 8 个动作，包含专家、策略及联合扰动；保存执行前意图、相同快照下的机器人状态、决策后至少 1 秒的球轨迹、击打接触及后续碰撞。不可行请求单独记录，不伪装成已执行动作。

将输出目录填入 `joint_air_hockey.yml` 的 `extra_train_branch_root`。新采集器使用 v2 schema，完成后直接可读；`finalize_joint_branches.py` 只用于历史 v1 pilot 格式。J1 还读取 H3 配置继承的旧分支目录，需保持这些前置数据可用。

## 4. 训练、选择与测试

按照根目录 README 的命令训练。正式脚本训练 action_only/joint/stop_gradient 三组，每组种子 42、43、44；只用验证集动作损失选择各组 checkpoint。

先在验证场景比较真实接触与拒绝原因，再锁定配置运行测试。`validate_joint_policy.py` 的接触对照只测试左臂；`validate_hit_pipeline.py` 才运行双臂与右臂接触后 1 秒检查。两者均依赖专家准备状态，不能把结果表述为全自主策略。

新生成的数据与旧 checkpoint 的 SHA256 不一致时，应重新训练后续阶段；不能直接替换 checkpoint 内的数据指纹。独立新场景泛化需要额外采集和锁定测试场景，旧 15 个场景仅作既有实验回归。
