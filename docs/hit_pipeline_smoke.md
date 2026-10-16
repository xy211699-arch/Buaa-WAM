# 世界模型与模仿学习策略：流程实验 P1

2026-10-11。用户将目标改为先验证流程接通，暂不等待 H3 的误差和失败识别率达标。本实验因此允许使用未通过性能验收的 H3；没有修改其原有失败结论或放宽物理成功条件。

## 当前实际链路

```text
专家回放生成场景及击球前准备状态（实验初始化）
  → 9 帧顶视 RGB + 当前左臂关节位置/速度/驱动目标
  → V4 冻结 RGB 网络：过去位置特征及未来 1 秒自由轨迹
  → 小型模仿学习策略：接触延迟、目标 Y、期望球速、方向修正、速度倍率、控制提前量
  → 策略附近 5 个意图（原动作、方向 ±8°、速度倍率 ×0.9/1.1）
  → 原生专家执行器的 plan_only：几何、时序与 IK 检查，不推进物理
  → H3 预测各候选的击球后轨迹，按 0.5 秒的 +X 进展与中心 Y 排序
  → 只执行排名第一的候选，不用真实执行结果挑选或重试
  → 左臂真实接触后，原有右臂专家从当前状态规划并实际回击
  → 右臂接触后继续 1 秒，保存 PNG、JSONL、元数据、报告及预览
```

左臂模型接口 `HitPipeline.propose` 只接收 RGB 历史与当前机器人读数，不接收 seed、专家计划、球真值或未来图像。执行器的左臂参考轨迹已替换为 RGB 网络预测；速度来自预测位置的数值差分，旋转速度假设为零。候选准备仅调用当前机械臂几何和 IK，不做物理搜索。H3 接触分数只记录，因识别失败已知，不用于否决动作。

这是**专家准备状态中的接管实验**，尚不是从机器人初始位姿完全自主运行。观察截止时间取自初始化用专家计划，关节状态也包含专家准备动作的信息；策略可能依赖这些信息。右臂仍使用读取仿真真值的专家搜索。不能把本结果表述为双臂纯 RGB 自主策略或实时闭环控制。当前仅做一次左臂决策，没有每 0.05 秒重规划。接下来可按需移除专家准备、接入在线连续决策并学习右臂。

## 策略训练

新增 `world_model/policy.py`、`world_model/pipeline.py`、`scripts/train_behavior_policy.py`、`scripts/validate_hit_pipeline.py` 和 `tests/test_behavior_policy.py`。原生执行器只增加可选 `plan_only=False` 参数，默认执行流程保持原义。

策略为两层 96 隐藏单元 MLP，输入 58 维 RGB 世界模型特征与 18 维关节特征，输出 6 个连续意图量和一个直接/近轨反弹类别分数。只训练策略头，RGB 网络与 H3 固定；使用专家计划作模仿标签，不做强化学习、不加入 critic。数据仍按种子 70/15/15 划分，共 280/60/60 个专家准备中的窗口；输入/输出归一化仅在训练集拟合。检查源数据 SHA256；策略权重绑定世界模型 SHA256，推理拒绝混用模型。

训练 300 轮，验证集选择第 58 轮。测试意图误差：接触延迟 MAE 17.0 ms、目标 Y 6.87 cm、期望球速 0.307 m/s、方向修正 2.31°。这些是模仿标签误差，不能等同实际击球误差。

权重：

- 策略：`outputs/behavior_policy/run_20261011_170429_148931/best.pt`
- 世界模型：`outputs/world_model_hit_v3/run_20261011_163504_427151/robot_state_best.pt`

## 实际结果

运行：`outputs/hit_pipeline/run_20261011_171219/`。

| seed | 左臂模型链路到动作执行 | 左臂真实接触 | 右臂真实接触 + 1 秒尾段 |
| --- | --- | --- | --- |
| 68 | 完成 | 否，NO_CONTACT | 未触发 |
| 36 | 完成 | 是 | 是 |
| 30 | 完成 | 否，NO_CONTACT | 未触发 |

**已有一次完整双臂执行成功**，但另两次未击中。本次 3 个种子为原有留出集合中的 smoke test，不是新的盲测，也不能把 1/3 作为可靠成功率估计。流程完成和任务成功在报告中分别记录；只有左臂击中才触发右臂控制。

seed 36 的原生元数据记录左臂接触 0.325 秒、右臂接触 0.877 秒、录制结束 1.877 秒；39 帧、640×360、20 FPS，终止间隔可短于采样周期。右臂搜索的 323348 个物理步在演示时间线之外。所有候选的左臂准备阶段物理步数为 0。三个执行的 JSONL 时间戳严格递增，数量与元数据一致；成功视频实际解码到 39 帧。

成功视频：`seed_36/policy_execution/preview.mp4`。各 seed 的 `expert_warmup/` 是初始化回放，不能作为策略执行结果；`policy_execution/` 才是接管后的实测结果。`decision_inputs.npz` 保存实际输入和 RGB 预测，`report.json` 保存提议、候选排名与实际执行结果；所有生成物留在忽略的 outputs 目录。

验证：完整 unittest 51 项，45 通过、6 GPU/viewer 条件跳过；新增测试检查训练标签不读取执行结果、策略梯度、RGB 预测参考的时钟与速度，以及 plan_only 不执行运动、不读取球真值。测试在 GPU 可见环境运行。独立审计确认双侧物理接触及右臂 1 秒尾段。

## 复现

仓库根目录，已配置 CUDA/Vulkan 的 RoboTwin Python 环境：

```bash
python scripts/train_behavior_policy.py \
  --world-checkpoint outputs/world_model_hit_v3/run_20261011_163504_427151/robot_state_best.pt \
  --device cuda

# 将策略路径替换为本次训练生成的 best.pt；下列为已完成实验的路径
python scripts/validate_hit_pipeline.py \
  --policy-checkpoint outputs/behavior_policy/run_20261011_170429_148931/best.pt \
  --episodes 3 --selection world-model --device cuda

python scripts/make_episode_preview.py \
  outputs/hit_pipeline/run_20261011_171219/seed_36/policy_execution

PYTHONPATH=. python -m unittest discover -s tests -v
```

每次验证写入新的运行目录。右臂专家搜索可能耗时，当前实验不能据此宣称满足实时控制延迟。
