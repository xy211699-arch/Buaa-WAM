<h1 align="center">
  <a href="https://robotwin-benchmark.github.io"><b>RoboTwin</b> Bimanual Robotic Manipulation Platform<br></a>
</h1>
<h2 align="center">Lastest Version: RoboTwin 2.0<br>🤲 <a href="https://robotwin-platform.github.io/">Webpage</a> | <a href="https://robotwin-platform.github.io/doc/">Document</a> | <a href="https://arxiv.org/abs/2506.18088">Paper</a> | <a href="https://robotwin-platform.github.io/doc/community/index.html">Community</a> | <a href="https://robotwin-platform.github.io/leaderboard">Leaderboard</a></h2>

https://private-user-images.githubusercontent.com/88101805/463126988-e3ba1575-4411-4a36-ad65-f0b2f49890c3.mp4

**RoboTwin 2.0** (*ICML 2026*) — [Webpage](https://robotwin-platform.github.io/) · [Doc](https://robotwin-platform.github.io/doc) · [Paper](https://arxiv.org/abs/2506.18088) · [Talk](https://www.bilibili.com/video/BV18p3izYE63/?spm_id_from=333.337.search-card.all.click) · [机器之心](https://mp.weixin.qq.com/s/SwORezmol2Qd9YdrGYchEA) · [Leaderboard](https://robotwin-platform.github.io/leaderboard)

<details>
<summary>Earlier papers & challenge report</summary>

- **1.0 / Early** — RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins · *CVPR 2025 (Highlight)* [PDF](https://arxiv.org/pdf/2504.13059) / [arXiv](https://arxiv.org/abs/2504.13059) · *ECCV Workshop 2024 (Best Paper)* [PDF](https://arxiv.org/pdf/2409.02920) / [arXiv](https://arxiv.org/abs/2409.02920)
- **CVPR'25 MEIS Challenge Report** — [PDF](https://arxiv.org/pdf/2506.23351) / [arXiv](https://arxiv.org/abs/2506.23351) · [量子位](https://mp.weixin.qq.com/s/qxqs9vvvHsAJ-0hoYANYzQ)

</details>

# BUAA WAM air hockey scene

This fork includes a two-UR5 air hockey preview, the table/puck/striker definitions,
interaction controls, tests, and the UR5-WSG model needed to run the preview.
See [the scene guide](docs/air_hockey_assets.md) for controls and parameters.

```bash
python scripts/preview_air_hockey_assets.py
```

The other RoboTwin assets and datasets are not stored in this repository. Follow
the upstream asset download instructions below to run the remaining benchmark tasks.
Clone with `--recurse-submodules` to obtain XPolicyLab.
Before using the full RoboTwin CuRobo planner, run
`python scripts/update_embodiment_config_path.py` from the repository root to
generate machine-specific embodiment paths from the included templates.

# 📚 Overview

> Prefer the [RoboTwin Document](https://robotwin-platform.github.io/doc/) for full guides — this README is a quick start.

RoboTwin 2.0 and [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) share deployment via [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab): one policy-serving / eval stack across both benchmarks (single-task, multi-GPU, and remote server + local sim).

**Default branch:** [`main`](https://github.com/RoboTwin-Platform/RoboTwin/tree/main) (RoboTwin 2.0).

<details>
<summary>Other branches (legacy / special-purpose)</summary>

- [IsaacLab-Arena](https://github.com/RoboTwin-Platform/RoboTwin/tree/IsaacLab-Arena) · [RLinf_support](https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support) · [WBCD-2026](https://github.com/RoboTwin-Platform/RoboTwin/tree/WBCD-2026)
- [RoboTwin-1.0](https://github.com/RoboTwin-Platform/RoboTwin/tree/RoboTwin-1.0) / [early_version](https://github.com/RoboTwin-Platform/RoboTwin/tree/early_version) · [gpt](https://github.com/RoboTwin-Platform/RoboTwin/tree/gpt)
- [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) · [CVPR-Challenge-2025-Round1](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round1) / [Round2](https://github.com/RoboTwin-Platform/RoboTwin/tree/CVPR-Challenge-2025-Round2)

</details>

# 🐣 Update
* **2026/08/03**, We add [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)-based policy evaluation with single-task evaluation, multi-task multi-GPU scheduling, and remote policy-server/local-simulator deployment.
* **2026/03/03**, We release [RMBench](https://github.com/RoboTwin-Platform/RMBench), which is a memory-dependent manipulation benchmark built upon RoboTwin 2.0.
* **2026/02/20**, Usage supported in <a href="https://github.com/starVLA/starVLA">StarVLA</a>, which is a user-friendly codebase for VLA development.
* **2026/01/23**, We update IsaacLab-Arena and <a href="https://github.com/RLinf/RLinf">RLinf</a> support (contributed by RLinf team).
* **2025/08/28**, We update the RoboTwin 2.0 Paper [PDF](https://arxiv.org/pdf/2506.18088).
* **2025/08/25**, We fix ACT deployment code and update the [leaderboard](https://robotwin-platform.github.io/leaderboard).
* **2025/08/06**, We release RoboTwin 2.0 Leaderboard: [leaderboard website](https://robotwin-platform.github.io/leaderboard).
* **2025/07/23**, RoboTwin 2.0 received Outstanding Poster at ChinaSI 2025 (Ranking 1st).
* **2025/07/19**, We Fix DP3 evaluation code error. We will update RoboTwin 2.0 paper next week.
* **2025/07/09**, We update endpose control mode, please see [[RoboTwin Doc - Usage - Control Robot](https://robotwin-platform.github.io/doc/usage/control-robot.html)] for more details.
* **2025/07/08**, We upload [Challenge-Cup-2025](https://github.com/RoboTwin-Platform/RoboTwin/tree/Challenge-Cup-2025) Branch (第十九届挑战杯分支).
* **2025/07/02**, Fix Piper Wrist Bug [[issue](https://github.com/RoboTwin-Platform/RoboTwin/issues/104)]. Please redownload the embodiment asset.
* **2025/07/01**, We release Technical Report of RoboTwin Dual-Arm Collaboration Challenge @ CVPR 2025 MEIS Workshop [[arXiv](https://arxiv.org/abs/2506.23351)] !
* **2025/06/21**, We release RoboTwin 2.0 [[Webpage](https://robotwin-platform.github.io/)] !
* **2025/04/11**, RoboTwin is seclected as <i>CVPR Highlight paper</i>!
* **2025/02/27**, RoboTwin is accepted to <i>CVPR 2025</i> ! 
* **2024/09/30**, RoboTwin (Early Version) received <i>the Best Paper Award  at the ECCV Workshop</i>!
* **2024/09/20**, Officially released RoboTwin.

# 🛠️ Installation

See [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) for installation instructions. It takes about 20 minutes for installation.

XPolicyLab is embedded as a Git submodule. For a fresh checkout, clone RoboTwin recursively:

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
```

For an existing checkout, initialize the version pinned by RoboTwin:

```bash
git submodule update --init --recursive XPolicyLab
```

To pull the latest XPolicyLab commit on its configured `main` branch and refresh RoboTwin's submodule pin:

```bash
bash scripts/update_xpolicylab.sh
# optional: stage the pin / reinstall the editable package
bash scripts/update_xpolicylab.sh --stage --install
```

# 🤷‍♂️ Tasks Informations
See [RoboTwin 2.0 Tasks Doc](https://robotwin-platform.github.io/doc/tasks/index.html) for more details.

<p align="center">
  <img src="./assets/files/50_tasks.gif" width="100%">
</p>

# 🧑🏻‍💻 Usage 

## Document

> Full usage details live in the [RoboTwin Document](https://robotwin-platform.github.io/doc/) — start from [Usage](https://robotwin-platform.github.io/doc/usage/index.html). Prefer the Doc over this README when anything conflicts.

## Getting Data
We provide over 100,000 pre-collected trajectories as part of the open-source release [RoboTwin Dataset](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset). **We recommend downloading the pre-collected data (step 1) as the default path** — it is ready to train on immediately. Collect data yourself (step 2) only when you need custom task configs, domain randomization, or embodiment setups.

> **Always decode through `decode_image_bit`, and always encode through `encode_image_bit`.** Downloaded and self-collected episodes store cameras as encoded image bits. Decoding them yourself is unsupported, because they come in two byte formats and a hand-rolled decoder is right on one and reverses the channels on the other. A local copy of the pair is in [`data/decode_image_bit.py`](data/decode_image_bit.py) — it is the same functions as `XPolicyLab.utils.process_data`. Prefer the XPolicyLab import when the package is available:
>
> ```python
> from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit
> rgb = decode_image_bit(image_bits)  # RGB for every data version
> ```
>
> Stored image bits come in two formats, and both decode to RGB:
>
> | Format | How it was written | What a standard decoder sees |
> | --- | --- | --- |
> | **legacy** | an RGB array handed straight to `cv2.imencode`, which reads its input as BGR | red and blue swapped — the bytes are channel-reversed against the JPEG standard, and `cv2.imdecode` reverses them back |
> | **standard** | `encode_image_bit`, which converts to BGR first and stamps a JPEG `COM` segment with the payload `XPL-RGB1` | correct colors |
>
> Legacy data is never migrated — JPEG cannot swap channels losslessly — so the two formats coexist indefinitely and may appear in the same training run. Self-collected episodes now write the standard format. `decode_image_bit` tells the formats apart and returns RGB, so its output never needs a channel swap. Do **not** add `cv2.cvtColor(..., COLOR_BGR2RGB)` after it — the two formats are indistinguishable from any single sample, and that swap is the bug. Official LeRobot converters already call it; any custom training dataloader that reads these HDF5 files must do the same. Collection must not replace `encode_image_bit` with a bare `cv2.imencode`, which omits the marker and writes a buffer that reads back reversed. At eval time the policy server hands over decoded RGB, so `model.py` must not decode again. Details: [XPolicyLab — Decode only through `decode_image_bit`](https://github.com/XPolicyLab/XPolicyLab#decode-only-through-decode_image_bit).

<img src="./assets/files/domain_randomization.png" alt="description" style="display: block; margin: auto; width: 100%;">

## 1. Download XPolicyLab-Format Data (Recommended)
Download and extract all available XPolicyLab-format trajectories from Hugging Face:

```bash
bash scripts/download_xpolicylab_data.sh
```

To download only selected tasks, pass their names:

```bash
bash scripts/download_xpolicylab_data.sh adjust_bottle beat_block_hammer
```

Downloads and extractions both run in parallel (defaults: 8 workers each). As soon as a task ZIP finishes downloading, extraction starts without waiting for the rest. Tune concurrency with:

```bash
# Parallel download / extract workers (extract defaults to HF_MAX_WORKERS)
HF_MAX_WORKERS=8 HF_EXTRACT_WORKERS=16 bash scripts/download_xpolicylab_data.sh
```

Downloads land under `data/demo_clean/<task_name>/aloha_agilex/data/` (note: self-collected data lands under `data/<task_config>/...` instead).

## 2. Task Running and Data Collection (Optional)
For custom task configs, domain randomization, or embodiment setups, collect data yourself. The following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh beat_block_hammer demo_randomized 0
```

Collected demonstrations are saved directly in the XPolicyLab trajectory format — no extra conversion step is needed:

```text
data/<task_config>/<task_name>/<embodiment>/data/episode_0000000.hdf5
```

`<embodiment>` follows the `embodiment` field of the task config (`aloha_agilex` for the default `aloha-agilex` setup).

## 3. Convert to LeRobot (Optional)

Many XPolicyLab policies train on LeRobot datasets. After you have XPolicyLab-format HDF5 under `data/<task_config>/<task>/<embodiment>/data/` (from download or collection), convert with the shared scripts in `XPolicyLab/scripts/`. Those scripts already decode through `decode_image_bit` — do not replace that with `cv2.imdecode` / PIL if you fork them. Stored bits come in two byte formats; only `decode_image_bit` tells them apart.

Patterns are `<task_config>.<task>.<embodiment>` and may use `*` wildcards. They resolve against `data/` next to the RoboTwin root (for example `demo_clean.*.aloha_agilex`). Keep `--data_type` as the default `RoboDojo` — RoboTwin XPolicyLab trajectories share that HDF5 layout.

Run in an environment that already has the matching LeRobot package (v2.1 vs v3.0). Conversion writes under `HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`); point it at a large disk if needed:

```bash
export HF_LEROBOT_HOME=/path/with/enough/space/lerobot

# LeRobot v2.1 — all demo_clean tasks
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.*.aloha_agilex" \
  --repo_id robotwin_demo_clean_aloha_agilex \
  --max_episode 50

# LeRobot v3.0 — same selection
python XPolicyLab/scripts/transform_lerobot_v30_format.py \
  "demo_clean.*.aloha_agilex" \
  --repo_id robotwin_demo_clean_aloha_agilex_v30 \
  --max_episode 50

# Single task
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.beat_block_hammer.aloha_agilex" \
  --repo_id beat_block_hammer_demo_clean
```

Useful flags: `--repo_id` (output dataset name), `--max_episode` (cap episodes per task/embodiment), `--resolution HxW` or `--image_height` / `--image_width` (default: auto-detect; RoboTwin is often `240x320`). Output lands at `${HF_LEROBOT_HOME}/<repo_id>`.

## 4. Modify Task Config
☝️ See [RoboTwin 2.0 Tasks Configurations Doc](https://robotwin-platform.github.io/doc/usage/configurations.html) for more details.

Task settings such as `demo_clean` and `demo_randomized` are stored in `env_cfg/task_config/`.

## 5. Evaluate Policies via XPolicyLab

All evaluation goes through `scripts/eval_policy.sh`. The policy adapter must exist under `XPolicyLab/policy/<policy_name>/` (see the [XPolicyLab policy catalog](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy)).

`--env-cfg-type` selects the XPolicyLab action profile (validated against `XPolicyLab/utils/robot/_robot_info.json`; `arx_x5` matches RoboTwin's default aloha-agilex layout), while the simulator embodiment stays controlled by `--task-config`.

**Local evaluation (multi-task, multi-GPU).** The scheduler starts a policy server and simulator per task on your GPU pool. Task lists and GPU settings live in `env_cfg/eval/all_tasks.yml` (trim `tasks` to a single entry for single-task evaluation):

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name <policy_name> \
  --ckpt-name <checkpoint> \
  --env-cfg-type arx_x5 \
  --policy-conda-env <policy_env> \
  --eval-env-conda-env <robotwin_env> \
  --action-type <action_type>
```

Add `--dry-run` to validate the schedule without launching anything. Results are written to `eval_result/multitask/` by default.

**Split deployment (remote policy server + local simulator).** Start the server pool on the policy host, then point the local scheduler at it:

```bash
# On the policy-server host (fill in the placeholders first):
bash scripts/eval_policy.sh serve --config env_cfg/eval/remote_server.yml

# On the simulator host:
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name <policy_name> \
  --env-cfg-type arx_x5 \
  --eval-env-conda-env <robotwin_env> \
  --enable-remote \
  --policy-server-ip <server_ip> --policy-server-port <port>
```

`--policy-server-ip/--policy-server-port` can be repeated to use a server pool, or configured once via `enable_remote` / `policy_server_ip` / `policy_server_port` in the eval config.

# 🏄‍♂️ Experiment & LeaderBoard

> We recommend that the RoboTwin Platform can be used to explore the following topics: 
> 1. single - task fine - tuning capability
> 2. visual robustness
> 3. language diversity robustness (language condition)
> 4. multi-tasks capability
> 5. cross-embodiment performance

The full leaderboard and setting can be found in: [https://robotwin-platform.github.io/leaderboard](https://robotwin-platform.github.io/leaderboard).

# 💽 Pre-collected Large-scale Dataset

Please refer to [RoboTwin 2.0 Dataset - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset).

# 🎯 Official Benchmark Checkpoints

Please refer to [RoboTwin 2.0 Checkpoints - Huggingface](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/cotrain_ckpt).

# 👍 Citations
If you find our work useful, please consider citing:

<b>RoboTwin 2.0</b>: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
```
@article{chen2025robotwin,
  title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}
```

<b>RoboTwin</b>: Dual-Arm Robot Benchmark with Generative Digital Twins, accepted to <i style="color: red; display: inline;"><b>CVPR 2025 (Highlight)</b></i>
```
@InProceedings{Mu_2025_CVPR,
    author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
    title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {27649-27660}
}
```

Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop
```
@article{chen2025benchmarking,
  title={Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop},
  author={Chen, Tianxing and Wang, Kaixuan and Yang, Zhaohui and Zhang, Yuhao and Chen, Zanxin and Chen, Baijun and Dong, Wanxi and Liu, Ziyuan and Chen, Dong and Yang, Tianshuo and others},
  journal={arXiv preprint arXiv:2506.23351},
  year={2025}
}
```

<b>RoboTwin</b>: Dual-Arm Robot Benchmark with Generative Digital Twins (early version), accepted to <i style="color: red; display: inline;"><b>ECCV Workshop 2024 (Best Paper Award)</b></i>
```
@article{mu2024robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins (early version)},
  author={Mu, Yao and Chen, Tianxing and Peng, Shijia and Chen, Zanxin and Gao, Zeyu and Zou, Yude and Lin, Lunkai and Xie, Zhiqiang and Luo, Ping},
  journal={arXiv preprint arXiv:2409.02920},
  year={2024}
}
```

# 😺 Acknowledgement

**Software Support**: D-Robotics, **Hardware Support**: AgileX Robotics, **AIGC Support**: Deemos.

Contact [Tianxing Chen](https://tianxingchen.github.io) if you have any questions or suggestions.

# 🏷️ License
This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

---

## Air Hockey V2: DEFEND + HIT + RGB Data Collection

### Overview

The current air-hockey branch validates a complete minimum pipeline from an
incoming puck to a recorded expert demonstration.

Current pipeline:

    automatic random inbound puck
    -> DEFEND prediction
    -> Cartesian goalie motion
    -> physical puck/striker contact
    -> read actual post-contact puck state
    -> HIT preparation
    -> candidate forward simulation
    -> best candidate selection
    -> restore search seed
    -> execute selected HIT
    -> puck flight
    -> synchronized RGB/state recording
    -> episode termination

The current milestone is intended to validate the algorithm and data-collection
architecture before large-scale WAM training or real-robot deployment.

### Current validated status

The following components have been validated:

- two-UR5 air-hockey simulation scene
- SAPIEN/PhysX puck state access
- 2D DEFEND intercept prediction
- Cartesian goalie motion on the defensive line
- physical striker/puck contact
- actual post-contact puck-state observation
- DEFEND -> HIT_READY phase transition
- stable HIT preparation state
- simulator snapshot / restore
- 20-candidate HIT forward search
- best-candidate selection
- search-to-live replay consistency
- end-effector-following kinematic striker
- standalone RGB camera capture
- continuous episode RGB recording
- synchronized robot/state recording
- planner timeline isolation
- automatic random inbound episodes

Detailed validation notes:

    docs/2026-09-20_air_hockey_validation_summary.md

### Recommended entry point

The current end-to-end validation entry point is:

    scripts/collect_episode_rgb_auto_v1.py

Run one deterministic automatic episode:

    cd ~/Buaa-WAM
    conda activate RoboTwin
    python scripts/collect_episode_rgb_auto_v1.py --duration 0 --seed 1

Another random seed can be selected with:

    python scripts/collect_episode_rgb_auto_v1.py --duration 0 --seed 4

No manual puck impulse is required.

### Episode execution

The automatic collector performs:

    random puck initialization
    -> DEFEND
    -> CONTACT_HOLD
    -> HIT_READY
    -> HIT_PREP
    -> planner candidate search
    -> restore planner seed
    -> HIT_EXECUTE
    -> FLIGHT
    -> terminal event

The planner currently evaluates 20 HIT candidates generated from multiple
angle offsets and striker speeds.

### RGB and synchronized state recording

RGB is currently recorded at:

    resolution: 640 x 360
    rate:       30 FPS

Each recorded frame can include:

- RGB image
- simulation episode time
- control phase
- left robot q / dq
- right robot q / dq
- left TCP pose
- left joint drive target
- puck ground-truth pose
- puck ground-truth velocity
- left striker pose

Episode output is written under:

    outputs/episode_rgb_v1/

Typical layout:

    outputs/episode_rgb_v1/episode_YYYYMMDD_HHMMSS/
        rgb/
            000000.png
            000001.png
            ...
        frames.jsonl
        meta.json

Generated datasets are intentionally excluded from Git.

### Planner timeline isolation

Candidate forward simulation is not part of the continuous expert
demonstration.

The recorded demonstration timeline contains:

    DEFEND
    CONTACT_HOLD
    HIT_PREP
    HIT_EXECUTE
    FLIGHT

Internal planner operations are excluded:

    snapshot validation
    candidate 01
    candidate 02
    ...
    candidate 20

During candidate search:

- RGB recording is paused.
- demonstration episode time is paused.
- internal simulator physics steps are counted separately.
- simulator state may be repeatedly restored.

After planning:

- the stable search seed is restored.
- the recorder resumes.
- only the selected BEST execution is written into the expert episode.

This prevents planner branch resets from appearing as visual teleportation in
the training trajectory.

### Kinematic striker follower

The first physical implementation relied on friction between the WSG gripper
and a dynamic striker.

That approach was unstable during repeated forward simulations.

The current validation version therefore uses:

    UR5 IK / joint drive
        -> end-effector pose
        -> fixed EE-to-striker transform
        -> kinematic striker target
        -> physical collision with dynamic puck

The robot still performs real simulated joint motion.

The striker is kinematically synchronized with the end effector so that
gripper-slip artifacts do not invalidate HIT planning experiments.

### Validation scripts

The files under:

    scripts/validate_*.py

are incremental validation programs used during development.

They cover:

- puck state
- DEFEND prediction
- Cartesian DEFEND control
- DEFEND phase transitions
- snapshot / restore
- HIT candidate search
- best-candidate execution
- goal-event logic
- kinematic striker following
- RGB camera capture
- episode RGB collection

These scripts are intended for debugging and regression validation.

For normal end-to-end testing, use:

    scripts/collect_episode_rgb_auto_v1.py

### Scene-only preview

To inspect the original two-UR5 air-hockey simulation without running the
DEFEND/HIT algorithm:

    python scripts/preview_air_hockey_assets.py


## Air Hockey Technical Roadmap

The project is intentionally developed in stages.

The current V2 milestone proves the minimum closed simulation pipeline.
The next milestones focus first on producing reliable expert data, then on
learning from that data, and finally on transferring the learned controller to
a real robot.

### Stage 0 - Simulation and algorithm baseline

Status: validated.

Goals:

- build the two-robot air-hockey scene
- expose puck state from SAPIEN/PhysX
- implement DEFEND
- detect real puck/striker contact
- use actual post-contact puck state
- implement HIT candidate forward simulation
- restore simulator state between candidates
- select and replay the best HIT
- record RGB and robot state
- separate planner simulation from demonstration time

This stage corresponds to the current V2 implementation.


### Stage 1 - Production-quality episode collector

The current collector validates the architecture, but it is not yet the final
large-scale data-production system.

Next work:

- remove validation-only perturbation tests from the production collector
- implement deterministic episode reset
- support many episodes in one process
- automatically recover from failed episodes
- save episode-level metadata and random seeds
- finalize terminal conditions
- finalize camera placement and calibration conventions
- add TCP velocity
- define the final action representation
- add dataset integrity checks
- verify RGB/state/action synchronization
- verify that no planner frames enter the demonstration timeline

A production episode should have a clean structure:

    reset
    -> randomized puck state
    -> DEFEND
    -> contact
    -> HIT_PREP
    -> internal planner search
    -> BEST execution
    -> flight
    -> terminal event
    -> save metadata
    -> reset

Exit criterion:

The collector should be able to generate a large set of episodes
automatically without manual interaction or timeline corruption.


### Stage 2 - Finalize simulation physics and task geometry

The current milestone intentionally focused on the basic control and
data-collection flow.

Before creating the final training dataset, simulation physics should be
cleaned up.

Important work:

- finalize side-wall restitution
- finalize rail friction
- validate oblique puck-wall collision
- validate multi-bounce trajectories
- finalize physical goal opening geometry
- define precise goal / miss / timeout events
- verify puck radius and legal goal-center region
- verify striker/puck collision parameters
- verify table coordinate conventions

The task distribution should then expand gradually from simple direct incoming
shots to wall-bounce trajectories.

Recommended curriculum:

    direct inbound trajectories
    -> shallow-angle trajectories
    -> single-wall bounce
    -> larger angle distribution
    -> multi-bounce / harder states


### Stage 3 - Expert demonstration dataset

The simulator planner acts as a privileged expert.

The planner may use simulator ground-truth state internally:

- puck position
- puck velocity
- collision state
- forward simulation
- candidate scores

However, this privileged information should not be required by the deployed
learned policy.

The main learning trajectory should therefore be organized around:

    observation:
        RGB history
        + robot proprioception

    target:
        future robot action chunk

Puck ground-truth state should be treated as:

    privileged training/debug information

rather than as a required deployment input.

Candidate planner branches should remain outside the continuous expert
trajectory.

Conceptually:

    privileged simulator planner
        -> candidate forward search
        -> selected BEST behavior
        -> continuous expert demonstration
        -> learned WAM / policy

Dataset work should include:

- many randomized puck initial states
- reproducible random seeds
- train / validation / test splits
- episode success metadata
- phase metadata
- camera calibration metadata
- robot configuration metadata
- action timestamps
- optional privileged puck labels for analysis


### Stage 4 - Final action representation

The current recorder stores joint drive targets for validation.

Before training the final model, the action definition should be chosen to
match the desired deployment controller.

Candidate action spaces include:

- joint position targets
- joint velocity targets
- TCP position deltas
- TCP velocity commands
- future Cartesian waypoint chunks

For the planned WAM-style controller, the preferred direction is:

    image history
    + robot proprioception
        ->
    future action chunk

The final representation should be kept identical between:

- dataset generation
- policy training
- simulation inference
- real-robot deployment

This prevents an unnecessary action-space conversion layer during sim-to-real.


### Stage 5 - WAM / visuomotor policy training

The learning system should consume only information that can be available at
deployment time.

Primary model input:

    RGB image history
    + robot proprioception

Possible proprioception:

- q
- dq
- TCP pose
- TCP velocity

Primary model output:

    future robot action chunk

Puck ground truth remains optional privileged information for:

- diagnostics
- auxiliary losses
- offline analysis
- planner supervision

but should not be required for normal inference.

The first training objective is not to outperform the simulator planner.

The initial objective is to imitate the selected expert behavior reliably.


### Stage 6 - Closed-loop simulation evaluation

After offline training, the learned controller must be returned to the
simulator and evaluated without using planner ground truth for control.

Closed-loop evaluation should measure at least:

- successful DEFEND rate
- contact rate
- HIT completion rate
- goal rate
- miss distribution
- action smoothness
- striker tracking
- robot joint-limit violations
- collisions
- inference latency
- episode stability

The learned policy should be evaluated on states not used during training.

This stage determines whether the model actually learned a closed-loop visual
control policy rather than merely fitting recorded trajectories.


### Stage 7 - Robustness and domain randomization

After the base learned controller works in simulation, increase variability.

Possible randomization:

- puck initial state
- table friction
- puck friction
- restitution
- striker collision parameters
- lighting
- camera exposure
- camera pose perturbation
- visual textures
- small robot calibration offsets
- control latency
- observation latency

Randomization should be introduced progressively.

The purpose is to reduce sensitivity to simulator-specific details before
real-robot deployment.


### Stage 8 - AUBO sim-to-real preparation

Real deployment should begin only after the learned controller is stable in
closed-loop simulation.

Required real-system work:

- AUBO kinematic model
- AUBO joint limits
- AUBO controller interface
- table-to-robot coordinate calibration
- camera intrinsic calibration
- camera extrinsic calibration
- end-effector-to-striker transform calibration
- real puck / striker geometry measurements
- controller frequency validation
- observation latency measurement
- action latency measurement
- emergency stop
- workspace limits
- velocity / acceleration limits

The same logical controller interface should be preserved:

    RGB history
    + robot proprioception
        ->
    future action chunk


### Stage 9 - Progressive real-robot deployment

Real deployment should be incremental.

Recommended sequence:

    1. robot moves without puck
    2. replay low-speed simulated trajectories
    3. verify coordinate calibration
    4. static puck positioning
    5. slow incoming puck
    6. DEFEND only
    7. controlled HIT
    8. full DEFEND + HIT
    9. faster and more varied puck trajectories

At every stage, safety limits remain outside the learned policy.


### Overall architecture

The intended final architecture is:

    SAPIEN / RoboTwin simulator
            |
            v
    privileged expert planner
            |
            v
    continuous RGB + proprio + action dataset
            |
            v
    WAM / visuomotor policy
            |
            v
    closed-loop simulation evaluation
            |
            v
    robustness / domain randomization
            |
            v
    AUBO real-robot deployment


### Current project boundary

The current repository state proves:

    DEFEND + HIT algorithm path works
    planner forward search works
    search seed restore works
    RGB data collection works
    planner timeline isolation works
    automatic single episodes work

The next engineering milestone is therefore not another algorithm prototype.

The next milestone is:

    production-quality multi-episode expert data collection

followed by:

    WAM training and closed-loop simulation evaluation

and only after that:

    AUBO sim-to-real deployment

