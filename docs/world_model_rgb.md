# 第一阶段：RGB 来球轨迹预测

目标：输入顶视 RGB 历史，预测机械臂第一次击球之前的球轨迹。允许墙面反弹；不预测机器人击球后的结果，也不选择或执行动作。

## 输入、标签和时间

- 默认输入过去 0.4 秒的 9 帧 RGB，320×180，每帧间隔 0.05 秒。
- 输出未来 1 秒的 20 个世界坐标 `(x,y)`，间隔 0.05 秒。
- RGB 取 HDF5 的 `vision/cam_head/colors`，复用 `data/decode_image_bit.py` 的 RGB 解码。它们与 20 Hz `task_state/timestamps` 对齐；不能按帧号配对 50 FPS MP4。
- 标签取 `task_state/puck_pose` 的前两维。监督标签和诊断用速度不传给模型；`forward` 只接受 RGB 和预测时间。
- 首次机械臂物理接触前 0.02 秒截断。较短未来仍可训练，但至少有 0.1 秒；接触之后和不存在的标签由 mask 屏蔽。
- 所有输入帧也位于首次接触之前。这里学习实际发生的自由运动，不把专家击球后的未来错误地当成无动作预测标签。
- 每条 seed 的所有 episode 放在同一集合；默认 70%/15%/15% 训练/验证/测试，先划分 episode 再切窗口。

网络采用小 CNN 学习球的空间热图，再用 GRU 编码运动历史。训练时将球真值通过记录的相机位姿和内参投影到图像，为小球提供像素级辅助监督；模型预测的热图通过 softmax 加权得到位置，真实位置仅用于损失。当前固定顶视相机的平面标定保存在 checkpoint，推理不需要球真值，也不使用颜色阈值检测。预测头学习对视觉匀速预测的修正，不生成未来 RGB。

默认 `spatial` 模型要求相机标定一致、球中心位于配置的 `puck_plane_z_m` 平面；变化相机或明显离桌运动需要扩展模型。早期 `cnn_gru` 检查点仍可按其保存配置评估，不能与默认模型混用。

## 使用

从仓库根目录使用已经安装依赖的 RoboTwin 环境，不需要安装新包或启动 SAPIEN：

```bash
# 验证数据、前向、反向、保存、评测与绘图全链路
PYTHONDONTWRITEBYTECODE=1 python \
  scripts/train_world_model_rgb.py --device cpu --epochs 2 --max-batches 2

# 完整训练，默认 30 个 epoch；CUDA 不可用时 auto 回退 CPU
PYTHONDONTWRITEBYTECODE=1 python \
  scripts/train_world_model_rgb.py --device auto

# GPU 8 GB 建议先 batch 16；检查 metrics.json 中的峰值分配显存
PYTHONDONTWRITEBYTECODE=1 python \
  scripts/train_world_model_rgb.py --device cuda --batch-size 16

# 固定 checkpoint 和原始测试划分重评估；数据 SHA256 变化时拒绝评估
PYTHONDONTWRITEBYTECODE=1 python \
  scripts/train_world_model_rgb.py --evaluate outputs/world_model_rgb/run_<timestamp>/best.pt

# 回归测试
PYTHONDONTWRITEBYTECODE=1 python \
  -m unittest tests.test_world_model_rgb -v
```

配置：`env_cfg/world_model_rgb.yml`。历史帧数、预测步数与 `dt_s` 一起定义时间范围；例如预测 0.5 秒改为 `horizon_steps: 10`。数据采样间隔必须真实匹配配置，代码不会将 20 Hz 数据插值为 50 Hz。更高频预测应补采对应数据。

每次生成独立 `outputs/world_model_rgb/run_*`：

| 文件 | 含义 |
| --- | --- |
| `config.json` / `splits.json` / `dataset_sha256.json` | 配置、完整 episode 划分与原始数据校验 |
| `data_report.json` | 各集合窗口数及完整预测范围窗口数 |
| `initial_validation.json` | 未训练模型的验证结果 |
| `history.jsonl` | 逐 epoch 损失、验证误差和耗时 |
| `best.pt` | 验证集平均位移误差最低的模型及配置/划分 |
| `metrics.json` | 最佳模型测试结果、各预测时刻样本数与误差 |
| `test_predictions.png` | 未见 episode 的最后一帧 RGB 与轨迹对比 |

较长预测时刻可用样本更少，须同时看每个 horizon 的样本数与误差。`ade_m` 是所有有效未来点的平均欧氏距离，不是成功率。`rgb_constant_velocity_ade_m` 使用同一视觉模块估计的最近两个位置作匀速外推，帮助判断时间模型是否提供增益。`privileged_constant_velocity_ade_m` 使用真实当前位置和速度，只是特权诊断基线；不能声称它和 RGB 模型的输入相同。`--max-batches` 的结果标记为 limited run，不能当成完整测试结论。

8 GB 内默认 batch 16 的低维输出设计留有训练空间，但实际 CUDA/Vulkan 总占用需要设备实测。本流程不启动渲染；解码后的 uint8 图像缓存在 CPU，DataLoader 使用单进程避免复制缓存。训练和 RGB 仿真采集宜错峰。只加载可信的本脚本 checkpoint。

## 用训练好的模型预测

```python
from world_model.inference import RGBTrajectoryPredictor

predictor = RGBTrajectoryPredictor("outputs/world_model_rgb/run_<timestamp>/best.pt")
# rgb_history: 按时间排列的 9 帧 RGB uint8，形状 (9,H,W,3)，间隔 0.05 秒。
# 原始 HDF5 图像需先用 decode_image_bit 解码；传入相机图像则必须保证 RGB 顺序。
forecast = predictor.predict(rgb_history)
print(forecast["times_s"])  # 相对最后一帧的未来时间，0.05 ... 1.00 秒
print(forecast["xy_m"])     # 20×2，世界坐标，米
```

不能传 BGR、未来图像、改变裁剪后的画面或不同位姿/FOV 的相机。调用者须维护正确的帧时间间隔；本接口不使用球真值，也不执行机械臂动作。

## 下一阶段的边界

逐版改动、指标和采用结论见 [世界模型版本记录](world_model_versions.md)。当前采用 V4：固定视觉定位后微调时间模型，整体测试误差 2.51 cm，1 秒误差 5.32 cm。GPU 训练及原始 RGB 回放已验证，在线仿真击球尚未验证。

当前只有 100 条成功专家 episode，短预测和初步视觉学习可以验证，但不代表能够泛化到任意来球、相机或物理参数。检查长预测和反弹片段误差，再增加无机械臂击球的独立仿真数据及 camera/physics 验证集。后续击球结果预测需要动作条件和完整控制记录，不能直接用本模型代替。
