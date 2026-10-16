# J0/J1 联合训练首轮结果

日期：2026-10-13。结论：当前状态共享表示、直接 Flow Matching 策略、未来辅助预测及动作条件动力学已实现，完成正式训练与真实仿真接触对照。联合训练在两个初始化下增加一个击中，但第三个初始化退步，总体没有稳定提升；保留 P2 默认模型，不自动替换。

## 已实现内容

- `world_model/model.py` 抽出 `encode_observation`，只编码过去/当前 RGB；原 V4 预测接口和权重保持兼容。
- `world_model/joint_model.py`：两层、宽度 128 的当前上下文 Transformer，机器人状态 token，六维高层意图 Flow Matching 头，专家未来轨迹头 F、动作条件动力学头 D 与接触头。
- `world_model/joint_data.py`：决策起点后 0.05–1.00 秒的 XY 监督；原始样本在右臂接触前掩码，插值不能跨接触。失败样本保留真实未来，非专家分支不作策略模仿标签。
- `world_model/joint_inference.py`：策略只读当前上下文，固定采样噪声用于配对比较；既有执行器所需的 V4 来球预测另外计算和计时，不声称整个控制系统已删除未来预测。
- 新增补采、标签发布、训练、配对评测脚本及测试，`validate_hit_pipeline.py` 增加可选 `--joint-checkpoint`，旧 P2 参数仍支持。

首版只研究轨迹辅助监督，未训练未来 RGB 视频、未引入大型视频 DiT、语言模型、critic、RL 或未来门控。部署只生成一套高层击球意图，仍由原生 IK/运动执行器控制机械臂。

## 数据与关键纠正

原生 70/15/15 种子划分保持不变，280/60/60 个专家决策窗口。旧分支训练/验证/测试 71/23/74 条，均只监督动作条件动力学，使用固定视觉前端的逐帧特征缓存；共享上下文在每一步重新计算并训练。

首批补采 20 个训练状态、160 个专家附近扰动，仅 1 条未发生观测期碰撞，失败覆盖不足，保留作预采集。第二批增加当前策略与时机/方向/速度联合扰动，同样 20 个状态、160 条请求：155 条可执行，5 条被几何/时序/IK 检查拒绝。最终 155 条中击打阶段真实接触 110 条、击打失败 45 条，每条至少保存决策后 1 秒真实轨迹。

发现 45 条击打失败中，44 条在指令结束后的持续运动又发生了碰撞。不能把这种后续碰撞当作正式击打成功：公开 `contact` 标签定义为击打阶段的物理接触，`observed_contact_s` 保存后续实测碰撞以避免未来插值跨事件；实际失败原因保留。原始预采集数据不覆盖，发布新的完成目录并保留 SHA256。当前试点数据的成功接触时间采用 1 ms 采样的首个实测接触，失败原因经过一致性校验；新采集代码已直接保存完整执行返回值和击打阶段接触时间。

正式补采目录：`outputs/joint_branch_data/finalized_20261013_175025/`，含 `audit.json`。正式训练共 506 个训练窗口：280 个专家窗口、71 个旧分支、155 个新分支。重复状态/窗口不作为独立场景计数。

## 正式训练设计

配置 `env_cfg/joint_air_hockey.yml`，正式结果 `outputs/joint_air_hockey/run_20261013_175111_085087/`。三个训练随机种子 42、43、44，三种方式，各训练 100 轮：

| 方式 | 动作监督 | 未来/动力学监督 | 未来损失回传共享当前表示 |
| --- | --- | --- | --- |
| action_only | 是 | 否 | 否 |
| joint | 是 | 是 | 是 |
| stop_gradient | 是 | 是 | 否 |

各模型初始化、训练数据、步数、采样噪声和动作头一致，checkpoint 仅根据验证集动作 Flow Matching 损失选择，不按测试结果挑轮次。冻结视觉前端，未来不会输入动作头。采用 FP32（实测已明显低于显存预算），batch=8、4 步 Euler 动作采样、AdamW、学习率 0.0003。未来与动力学权重各 0.3，接触权重 0.03。

早先预实验 `run_20261013_164256_220553/` 使用全局梯度裁剪，使 stop_gradient 的辅助头间接改变动作梯度缩放；仅作预实验，不用于模型推广，已有 methodology_notice.json。正式版分别裁剪共享/动作参数与辅助参数，三对 action_only/stop_gradient 的策略与当前表示权重差均精确为 0，见 `ablation_audit.json`。这验证了停止未来梯度的控制确实隔离了辅助任务影响。

可训练参数 444557；本轮 PyTorch 已分配峰值约 32.6 MiB、保留显存峰值约 222 MiB，不包含 CUDA 驱动、桌面、渲染和整机 GPU 峰值。训练模型能放入当前 8 GB 设备，未做完整系统峰值监测。

测试专家窗口的采样动作时机 MAE，三训练种子平均：纯动作约 59.3 ms、联合约 54.2 ms。动作条件轨迹 ADE：联合约 14.8 cm；未来辅助轨迹 ADE 约 16.3 cm。其时间基准是决策后 1 秒，与 H3 接触后 ADE 不可直接横比。纯动作模型的未来头未训练，因此不拿其随机未来误差当作有效预测基线。当前接触分类在原测试的三条失败上仍不能可靠识别，未将分数用作执行门控。

## 真实控制对照

先在前 5 个验证场景配对评测，联合三个种子接触数为 4/5、3/5、3/5，纯动作为 4/5、3/5、2/5，P2 为 3/5。联合 seed42 在验证场景最好，选它做完整双臂 smoke test；不是看测试结果选出来的。

锁定设置后，在全部 15 个原有测试场景，同一初始快照、相同 RGB、机器人读数、采样噪声、V4 来球参考与原生执行器下，逐个执行，不用真实结果挑候选或重试。

| 训练随机种子 | 纯动作真实接触 | 联合训练真实接触 | 停止未来梯度 |
| --- | --- | --- | --- |
| 42 | 11/15 | 12/15 | 11/15 |
| 43 | 11/15 | 12/15 | 11/15 |
| 44 | 9/15 | 7/15 | 9/15 |

原 P2 同快照为 11/15。三个初始化累计均为 31/45 次接触，实际是 15 个场景被三个模型重复评测，并非 45 个独立测试场景。已有测试集多次使用，不能声称新的泛化或统计显著改善。

联合 seed44 的八个失败全是 TOO_LATE：生成的击球时机不足以完成准备，被执行器正确拒绝。问题更具体地落在生成动作的时序/可行性稳定性，不能把退步简单归因于“世界模型没有作用”。seed42/43 各一个 HIT_IK，其余为真实未接触。后续优先验证时机约束与 4/8 步采样稳定性，先在验证集选设置，不改成功条件。

配对验证目录：`outputs/joint_evaluation/run_20261013_180507/`；测试目录：`outputs/joint_evaluation/run_20261013_181244/`，含逐场景报告、summary 和 diagnostic_summary。联合 seed42 的直接策略推理中位约 3.06 ms、95 分位约 3.65 ms；加 V4 执行桥接后提议耗时 95 分位约 5.29 ms，不含相机渲染、解码、完整 IK 检查或右臂专家搜索，不是端到端闭环延迟。

## 双臂流程与边界

验证集选出的联合 seed42 完整流程目录：`outputs/hit_pipeline/run_20261013_181439/`。seed 68、36 完成双侧真实击球及右臂接触后 1 秒，seed 30 在动作准备/可行性检查阶段被拒绝；完整双臂成功 2/3。没有改失败标签，未执行也不记作流程执行完成。各 seed 的 report 区分 pipeline_completed、物理接触与 task_success。

成功视频保存在相应 `seed_<seed>/policy_execution/preview.mp4`，audit 核对实测接触、1 秒尾段、严格递增时间戳与可读帧数。仍从专家准备状态接管，观测截止时间来自专家计划，右臂仍是专家；仍是一次高层决策，尚未实现持续重规划或新独立场景评测。

原始 RGB 与机器人记录通过 direct_policy 接口，并通过 mock 禁止调用 V4 未来头和联合未来头，动作输出仍有限；见训练目录 inference_audit.json。单元测试也验证未来损失回传当前编码器、停止梯度隔离、改变未来头不改变固定噪声策略输出、失败后碰撞不覆盖击打失败、跨接触掩码与原 V4 兼容性。完整测试 60 项，54 通过、6 条件跳过，日志 `outputs/joint_logs/tests_final.log`。

## 复现命令

以下在仓库根目录的已配置 RoboTwin CUDA/Vulkan 环境运行；不要自动升级依赖。

```bash
# 新补采；保存原始记录及执行前意图
python scripts/collect_joint_branches.py \
  --policy-checkpoint outputs/behavior_policy/run_20261011_170429_148931/best.pt \
  --episodes 20 --actions mixed

# 仅早期 pilot 格式需要 finalize；新的采集器已区分接触阶段
# 当前正式训练 YAML 指向上述已完成的 finalized 数据目录
python scripts/train_joint_policy.py --config env_cfg/joint_air_hockey.yml --device cuda

python scripts/validate_joint_policy.py \
  --training-run outputs/joint_air_hockey/run_20261013_175111_085087 \
  --split test --episodes 15 \
  --reference-policy-checkpoint outputs/behavior_policy/run_20261011_170429_148931/best.pt

python scripts/validate_hit_pipeline.py \
  --joint-checkpoint outputs/joint_air_hockey/run_20261013_175111_085087/joint_seed42_best.pt \
  --episodes 3 --selection bc --device cuda

PYTHONPATH=. python -m unittest discover -s tests -v
```

所有模型、数据、日志与视频保留在忽略的 outputs；未提交生成产物，保留用户原有改动。当前交付的是已运行的 J0/J1 研究原型，联合训练收益仍未稳定确认，不切换默认 P2 模型，也不启动未来门控或 RL。
