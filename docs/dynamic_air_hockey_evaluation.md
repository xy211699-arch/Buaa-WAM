# 动态冰球策略评测代码

## 评测范围

本入口比较 FastWAM、Diffusion Policy 和 ACT 的动态回击能力，包含反应时间、挡板反弹、突发变速三组实验。它不是原专家采集脚本的成功率统计，也不使用专家替模型选择动作。受测机械臂完成一次回击，另一机械臂始终保持初始驱动目标；左右角色可通过统一协议镜像切换。

本次交付只进行了代码与合成数据测试，没有加载模型、启动模型服务或运行真实模型评测。SAPIEN 后端应在已安装 RoboTwin/SAPIEN、UR5 资产和 Vulkan 的仿真环境使用。本机缺少 SAPIEN，因此不能将纯逻辑测试通过理解为真实仿真或模型性能已验证。

## 文件与入口

统一入口为 `scripts/evaluate_dynamic_air_hockey.py`，也可使用 `python -m air_hockey_eval`。`air_hockey_eval/schema.py` 定义协议，`suite.py` 生成冻结清单，`reference.py` 做无策略预检，`sapien_backend.py` 负责真实物理和 RGB，`policies.py` 接入现有策略，`control.py` 与 `runner.py` 负责延迟与动作队列，`storage.py` 保存可恢复日志，`report.py` 生成统计和图表。

默认协议为 `env_cfg/eval/air_hockey_dynamic.json`，策略连接示例在 `env_cfg/eval/dynamic_policies/`。这些策略配置中的检查点标识和训练数据标识是明确的待替换值，不代表已经存在或训练了相应权重；运行前填写真实版本、训练种子和服务地址。

## 生成候选测试清单

在仓库根目录执行以下命令。这一步只读取配置并生成候选，不需要 SAPIEN 或模型；默认三档速度、三档 TTC、每格二十个重复，共九百个回合。每格包含一条反应时间案例、一对直接/反弹案例、一对无扰动/变速案例。

```bash
python scripts/evaluate_dynamic_air_hockey.py generate \
  --config env_cfg/eval/air_hockey_dynamic.json \
  --assets env_cfg/air_hockey_assets.yml \
  --output outputs/dynamic_eval/proposals.json

python scripts/evaluate_dynamic_air_hockey.py check \
  --suite outputs/dynamic_eval/proposals.json
```

速度和 TTC 是预设的测试网格，不是已证明可处理的能力范围。候选利用展开的镜面轨迹构造，考虑真实摩擦和恢复系数后仍需下一步预检。几何不可行时整次生成报错，不会悄悄删掉困难案例。相同配置和种子产生同一清单，已存在的清单不会被覆盖。

## 冻结真实 PhysX 参考轨迹

在已配置的仿真环境执行以下命令。它只运行桌面和球的物理，不加载策略，也不搜索专家动作。保持发球速度不变、调整发球 X，使实测球到达固定截击线的时间接近指定 TTC，并记录校准前位置、实测 TTC、截击位置、速度和反弹次数。

```bash
python scripts/evaluate_dynamic_air_hockey.py prepare \
  --suite outputs/dynamic_eval/proposals.json \
  --output outputs/dynamic_eval/suite_physx.json
```

直接来球要求参考轨迹没有侧边反弹，反弹来球要求一次反弹，扰动案例还会验证扰动后的自由轨迹。任一案例未通过时，整个预检失败；程序不会按专家或策略是否成功筛选案例。改变配置后应重新生成新的清单，而不是手改清单或对不同方法使用不同案例。

正式 `run` 只接受预检后清单。协议、全部资产参数和案例均参与 SHA-256 指纹；检查点变化、模型配置变化、执行代码变化或协议变化都会阻止错误恢复。物理参数在本任务中固定为清单保存的资产参数，不额外调用专家的随机物理采样器。

## 策略接入与观测边界

`xpolicy_ws` 使用 XPolicyLab websocket 的通用方法调用接口。FastWAM、DP 可使用原有 `reset`、`update_obs`、`get_action` 服务；ACT 必须通过下述父仓库入口启动完整动作段包装服务。原生 ACT 每次 `get_action` 只返回一个时间步，并按调用次数推进内部索引，不能直接当作 30 Hz 动作段接入延迟队列；控制端会在运行前检查能力声明并拒绝旧 ACT 服务。同一 GPU 上应避免其他模型同时执行推理，以减少资源竞争。

复制各方法原有的 `XPolicyLab/policy/<name>/deploy.yml` 到自己的部署配置，填写真实权重、归一化统计和训练结构。共同设置 `env_cfg_type: air_hockey_eval_ur5`、`action_type: joint`，训练输出维度必须对应十六维双臂 UR5-WSG；ACT 还须明确 `temporal_agg: false`。以下命令只作启动示例，本次没有运行服务或加载模型。

```bash
python -m air_hockey_eval.server --config outputs/my_act_deploy.yml --port 9003
```

ACT 包装层复用原生观测编码、归一化和网络推理，每次重新预测完整 `chunk_size` 段，再反归一化并交给共同队列截取执行前缀。ACT 的部署时序集成在本协议中关闭；这是明确的共同动作段对照设置，不代表已经评估其原生时序集成版本。该入口也可启动 FastWAM 或 DP，方法由部署文件的 `policy_name` 决定，所有模型均需已有冰球任务检查点。

父仓库加载入口会检查动作类型和真实机器人维度，并给原生 ACT 同时设置 `action_dim` 与构造期间的 `ACT_ACTION_DIM=16`，无需另外手工配置环境变量。已有十四维 ALOHA 检查点不因改这个数字而变成十六维 UR5 检查点；网络结构、归一化统计和训练动作都必须匹配。

`xpolicy_local` 可在同一 Python 环境中导入已有 `XPolicyLab.policy.<name>.model.Model`，模型初始化参数通过 `model_config` 传入。`python` 可使用自定义 `module:factory`，工厂返回具有 `reset()` 和 `predict(packet)` 的对象；只有明确执行 `run` 时才会导入工厂或连接服务。自定义策略的返回值必须为已完成计算的 CPU NumPy 数组。

所有策略得到同一 RGB 数据流、机器人本体状态和指令。球真值、发球种子、反弹标签、未来参考轨迹、扰动计划和案例编号不进入策略观测。默认 `history_frames: 3`，每次提供最近三帧 30 Hz 观测，不足时显式重复首帧；已有 DP 恰好需要三帧，因此不会将 10 Hz 推理请求误当成 30 Hz 观测采样。模型内部是否保留更早状态仍应在训练/部署配置中记录，不能将同一输入流误写成所有模型使用相同的内部记忆结构。

实际只有一路俯视相机。默认 JSON 显式将同一 cam_head RGB 复制到旧三相机适配器要求的两个别名，所有方法使用相同映射；这两个别名不是真实腕部相机，也不增加信息。如果训练使用单相机，应将协议映射改为仅 cam_head，并使用支持单相机的适配器；不能把单相机数据直接配上要求真实三视角的检查点并宣称兼容。

机器人为两个六关节 UR5，各有两个物理手指。评测端提供 `env_cfg/air_hockey_eval_ur5.yml` 和父仓库运行时的 `ur5_wsg_dual` 维度定义，总维度为十六。该配置用于已有适配器的评测运行时；XPolicyLab 内部训练脚本使用另一份独立维度注册表，当前不修改该子模块，也不声称已经完成其训练侧接入。

策略输出必须是绝对关节驱动目标，受测侧提取六个机械臂关节，手指和另一侧机械臂保持原目标。原始采集 HDF5 的 `action` 是下一实测关节值，不等于实际指令；训练与部署应使用 `control/*_joint_drive_target` 的相应臂关节，并核对保存的关节名顺序。配置明确要求 `action_semantics: joint_drive_target`，不自动将两种语义混用。

## 运行与恢复

确认三种策略使用相同训练数据版本、共同观测/动作协议和相同运行硬件。不同训练种子的实际权重需要独立配置和输出目录；修改 `training_seed` 标签本身不能代替重新训练。以下命令是使用方法，本次没有执行。

```bash
python scripts/evaluate_dynamic_air_hockey.py check \
  --suite outputs/dynamic_eval/suite_physx.json \
  --policy-config env_cfg/eval/dynamic_policies/fastwam.json

python scripts/evaluate_dynamic_air_hockey.py run \
  --suite outputs/dynamic_eval/suite_physx.json \
  --policy-config env_cfg/eval/dynamic_policies/fastwam.json \
  --output outputs/dynamic_eval/FastWAM/seed_0 \
  --hardware-label node17-RTX5090-isolated \
  --mode latency --resume
```

`run.json` 固定运行身份，`episodes.jsonl` 每回合刷新并同步到磁盘，`traces/` 保存物理真值、接触事件和实际驱动目标，不重复存 RGB。相同身份恢复时跳过已完成的 case_id；只有文件末尾未写完的 JSON 才会修复，中间损坏或重复结果会报错。操作系统文件锁防止两个进程同时写入同一结果目录，异常退出后锁自动释放。

模型报错、非法动作和超时会保存为失败，仍计入分母；场景初始化或渲染等基础设施错误会中止运行并留下 `interruption.json`，不会冒充某个策略的任务分数。长时间无响应的策略停止整次运行，避免旧 RPC 在后续回合继续影响模型状态。完成全部案例后才生成 `complete.json`。

## 延迟与成功判定

`latency` 模式将策略放在后台线程，等待期间物理继续执行已有动作。逻辑时间门控避免 CPU 仿真提前跑过尚未结束的真实推理；相机处理、观测传递、推理和额外延迟参与动作到达时间。若仿真本身运行慢于实时，它并不承诺墙钟一倍速，但不会把提前计算出的动作应用到过去。

动作按观测时间锚定，最多执行共同的 `execution_steps` 前缀；到达时已过期的动作步直接丢弃，不把旧轨迹整体向后平移。新段替换旧段的未来动作，空缓冲继续保持旧命令并统计饥饿比例。模型请求、到达和首次应用时刻均保存；端到端延迟统计与每段动作吞吐分开。

`zero_latency` 将计算开销从仿真时钟中移除，只作诊断，不能与主测试混合汇总。不同额外延迟、分辨率、观测窗口或动作频率形成不同清单指纹，也不能混合比较。P95 应用延迟仅针对实际应用过的动作段，需结合缓冲饥饿、过期动作和任务失败阅读。

成功要求先有真实 PhysX 接触冲量，球随后沿回击方向越过固定对侧判定线，并满足有效宽度、最小纵向速度及时间限制；跨线时刻和横向位置通过相邻物理帧插值。关节位置/速度、命令位置/速度、击球器工作区与高度及机器人碰桌违规均会使成功无效。速度限值为当前基准配置值，场景资产仍标注 provisional，不能将其当作实机安全认证。

## 统计与三张结果图

每种方法建议至少三次独立训练，所有方法运行同一冻结案例清单。统计按训练运行和场景两层重采样；同一网格重复对应的五个相关条件作为一个场景组共同抽样，方法间保持场景配对、独立重采样训练种子，条件间按 pair_id 配对。缺少案例、同方法重复检查点身份或训练种子、混用评测代码、延迟模式、训练数据或硬件标签时拒绝汇总，不把剩余成功样本当成完整结果。检查点标识必须真实反映训练来源；单靠更改标识无法证明独立训练。

```bash
python scripts/evaluate_dynamic_air_hockey.py report \
  --suite outputs/dynamic_eval/suite_physx.json \
  --runs outputs/dynamic_eval/FastWAM/seed_0 \
         outputs/dynamic_eval/FastWAM/seed_1 \
         outputs/dynamic_eval/FastWAM/seed_2 \
         outputs/dynamic_eval/DP/seed_0 \
         outputs/dynamic_eval/DP/seed_1 \
         outputs/dynamic_eval/DP/seed_2 \
         outputs/dynamic_eval/ACT/seed_0 \
         outputs/dynamic_eval/ACT/seed_1 \
         outputs/dynamic_eval/ACT/seed_2 \
  --output outputs/dynamic_eval/report
```

输出 `summary.json`、汇总与分组 CSV、条件配对差值、方法间配对差值和反应时间阈值；反应时间曲线、反弹对照柱状图及扰动对照柱状图同时输出 PNG/SVG。方法差值分别覆盖各 TTC；`method_condition_effects.csv` 进一步比较两方法从直接到反弹、从无扰动到变速的成功率变化，正值表示 A 的退化较小或增益更大，避免仅凭困难条件的绝对成功率混淆一般能力与动态适应能力。所有 CSV 中的成功率和差值均为比例，绘图转为百分数；延迟单位为秒。反应时间阈值只报告已测试网格内、95% 下界达到目标且所有更宽裕档位也达标的最短 TTC，不外推未测极限。

经验 bootstrap 在全成功、全失败等情况下会出现零宽区间，代码以 `ci_degenerate` 标记，不能据此宣称总体没有不确定性。`observed_min_ttc_s` 仅描述样本成功率达标的最短已测档位；`min_tested_ttc_s` 还要求至少三个独立训练运行及所需区间不退化，否则返回空值和证据不足状态。这些区间是经验近似区间，并非实机能力认证；组间差中差用于预先指定的对比，不应从大量分组中挑选显著结果后声称已验证普遍优势。

## 仅代码验证

新增测试使用有明确期望结果的合成场景、轻量策略替身和暂存目录，不加载真实模型。可在 CPU 环境安装 `requirements-dynamic-eval.txt` 后运行以下命令；旧测试依赖 SAPIEN，不应通过修改旧测试或伪造包来制造全套通过。

```bash
python -m unittest discover -s tests -p 'test_dynamic_hockey_*.py' -v
python -m compileall -q air_hockey_eval scripts/evaluate_dynamic_air_hockey.py
```
