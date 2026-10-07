# BUAA-WAM Air Hockey Expert Pipeline

## Canonical entry points

```text
envs/air_hockey.py                  # the only registered task
envs/_air_hockey/expert.py          # both arms' final PhysX/IK expert
envs/_air_hockey/runtime.py         # scene, tool following, synchronized recorder
envs/_air_hockey/assets.py          # procedural table/puck/strikers
env_cfg/task_config/air_hockey_clean.yml
env_cfg/air_hockey_assets.yml       # unchanged geometry/physics/URDF settings
```

The current planner is a privileged simulator expert, not an RL policy.

## Official collection (recommended)

On jd / node17:

```bash
cd /mnt/data/users/haoran/project/egoav/Buaa-WAM
conda activate RoboTwin
bash collect_data.sh air_hockey air_hockey_clean 0
```

The last argument is the GPU index. `episode_num: 1` is the initial smoke-test
default; change it to the desired TOTAL count to continue collecting. Keep
`use_seed: false` to search new seeds and resume from `seed.txt`. `use_seed: true`
requires the existing `_traj_data` as well as the seed list. Use a new config name
and output directory after changing task physics; cached plans check the asset
configuration hash and reject incompatible replay.

Phase 1 searches both arms, executes the selected episode, checks success and
saves both selected candidates in `_traj_data`. Phase 2 builds a fresh scene with
the same seed and replays these candidates with no candidate-grid search. The
exact free-puck reference and Cartesian IK are still computed for timed replay.
Search time is excluded from the demonstration. Both physical contacts and the
1-second post-right-contact tail are required before export.

The task uses Base_Task's synchronized frame cache and the repository's native
HDF5/video converter. It does not instantiate Base_Task's generic table or robot
planner; these would change this task's dynamics. This integration is for expert
data collection, not yet a generic policy-evaluation `take_action` interface.

```text
data/air_hockey_clean/air_hockey/ur5_wsg_ur5_wsg/
  seed.txt
  _traj_data/episode0.pkl
  _expert/episode0_seedN_plan/{summary.json,meta.json,frames.jsonl,planner_label.json}
  _expert/episode0_seedN_replay/{summary.json,meta.json,frames.jsonl,planner_label.json}
  data/episode_0000000.hdf5
  video/episode_0000000.mp4
  instruction/episode_0000000.json
  scene_info.json
```

Only `vision/cam_head` is present: it is the custom vertical overhead camera,
640×360 at 30 Hz, not a wrist camera or the generic D435 pose. Native `state` and
`action` follow this repository's converter: state[t] and next measured state[t+1].
Each arm has 6 revolute joints; each gripper retains its two physical finger
positions in meters. Pose quaternions use wxyz. Do not treat the native `action`
as the expert's drive target. Actual sampled 8-DOF commands are also saved in
`control/{left,right}_joint_drive_target`, aligned with `state`.

`task_state` contains timestamps, next timestamps, phases and privileged puck
ground truth. `episode/timestamps` preserves all N frames (including the final
terminal frame); native state/action/vision contain N−1 transitions. The final
interval can be shorter than 1/30 s so recording ends exactly at right contact
+1.000 s; use recorded timestamps for dynamics training. `episode/metadata_json`
and `episode/expert_summary_json` preserve contacts and physics/serve labels.
Native mode writes images to the frame cache/HDF5/video, not duplicate PNGs.

Old planner and recorder module paths are compatibility shims into `envs`.
The old recorder script now runs the final dual-arm planner CLI, not its superseded
interactive one-arm loop. The legacy commands below still provide PNG/JSONL
debugging and parallel seed experimentation; they are not the official HDF5 path.

### Integration validation (2026-09-28, node17)

The official command above completed both phases: seed 0 was rejected for no
valid coarse candidate; seed 1 produced a successful episode. Recorded left
contact: 2.095 s; right contact: 2.707 s; end: 3.707 s. Output has 113 video frames,
112 native transitions, one cam_head camera, and a roughly 3.0 MB HDF5. Both replay
candidate-search counts are zero. All per-transition dataset lengths agree and
timestamps are strictly increasing. A terminal-capture guard prevents duplicate
timestamps when termination coincides with a scheduled capture.

`python -m unittest discover -s tests -v`: 25 tests, 19 passed, 6 viewer/Vulkan tests
skipped. The skips do not replace the separate real-GPU official collection test.
This is integration validation, not a dataset-scale success-rate measurement.

## Dual-arm continuation (2026-09-28)

The initial serve is unchanged: set puck velocity directly, sample speed in
0.70–1.20 m/s and heading in ±65 degrees. There is no right-arm serving motion.
The left arm returns the puck, then the right arm uses the same PhysX rollout,
IK and goal-conditioned candidate evaluation in a 180-degree rotated planning
frame to hit it back. Both striker discs follow their respective end effectors
as kinematic collision bodies.

Right-arm candidate search pauses the recorder and restores the handoff state.
Only the selected execution advances the dataset clock. PhysX collision impulses
timestamp actual right-striker contact; the episode ends 1.000 s after that
contact (including strike follow-through), even if the puck crosses a goal sooner.
`actual_event: RIGHT_RETURN` and `episode_success: true` identify a successful
dual-arm episode. Failed returns remain labeled failures and must be filtered.

Data uses one straight-down overhead camera centered on the table, 2.0 m above
the tabletop, 55-degree vertical FOV, 640×360 at 30 FPS. World +X stays image-right.
Optional `episode_camera` YAML settings: `height_above_table_m`, `fovy_deg`,
`width`, `height`, `fps`.

`frames.jsonl` now also includes `right_tcp`, `right_striker_pose` and
`action.right_joint_drive_target`. `meta.json` includes camera pose/FOV and
both physical contact times. Right-expert candidate coordinates in the planner
label are in its explicitly marked rotated frame; recorded observations remain
in world coordinates. `selected_search_candidate` at the top level still refers
to the left-arm plan, and `actual_execution.right_return` contains the right plan
and actual return.

On node17:

```bash
cd /mnt/data/users/haoran/project/egoav/Buaa-WAM
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /mnt/data/public_tools/miniconda3/envs/RoboTwin/bin/python \
  scripts/plan_direct_counterattack_stage5_v1_5.py \
  --seed 6 --coarse-only --collect --headless
```

Choose an available GPU for your run. `--coarse-only` skips local fine search;
`--verify` enables repeated candidate verification for both arms. Neither changes
the original serve distribution.

Initial node17 smoke validation: seeds 6 and 7 completed both physical strikes. Left
contact was at 2.584 s, right contact at 3.398 s and recording ended at 4.398 s
(133 frames). The right arm's joint targets stayed constant during the initial
serve. Seed 7 also passed through the batch runner: right contact at 2.844 s,
recording end at 3.844 s (117 frames), `episode_success=true` in its summary/CSV.
These two runs are functional smoke checks, not a broad success-rate estimate.

## Expert flow

```text
episode seed
-> sample serve + episode physics
-> snapshot exact simulator state
-> exact PhysX puck future rollout
-> candidate future contact states
-> workspace / timing / IK feasibility
-> goal-conditioned offensive action
-> candidate PhysX forward simulation
-> score candidates
-> restore exact initial state
-> execute selected BEST only
-> record synchronized RGB + robot/puck state
```

## Seed semantics

A seed reproducibly controls serve Y, serve speed, serve angle, initial puck spin, table friction, puck friction, rail friction, and rail restitution. Different seeds produce different episodes. All candidates inside one seed share the same episode conditions.

## Wall-bounce planning

Near-rail states are classified as:

```text
meaningful motion toward rail -> WAIT_FOR_BOUNCE
meaningful rebound into table -> DIRECT
near-zero wall-normal velocity -> EDGE_GLIDE / BANK_NEAR_RAIL
```

The planner uses an exact PhysX reference trajectory.

## PhysX bounce-threshold correction

The low-speed bounce threshold must be set on the same SceneConfig used for scene creation:

```python
scene_config = sapien.SceneConfig()
scene_config.bounce_threshold = float(config["physics"]["bounce_threshold_m_s"])
scene = engine.create_scene(scene_config)
```

Current target: `0.1 m/s`.

## One visual run

```bash
cd ~/Buaa-WAM
conda activate hockey
PYTHONPATH=. python scripts/plan_direct_counterattack_stage5_v1_5.py \
  --seed 6 --coarse-only --hold 50
```

## Collect one expert episode

```bash
PYTHONPATH=. python scripts/plan_direct_counterattack_stage5_v1_5.py \
  --seed 6 --coarse-only --collect --hold 20
```

Only the selected BEST execution is recorded. Candidate-search rollouts are excluded from the expert timeline.

## Parallel collection

```bash
PYTHONPATH=. python scripts/run_stage5_v1_5_parallel.py \
  --seed-start 1 --num-seeds 8 --workers 4 --coarse-only --collect
```

`num-seeds` is total episodes. `workers` is only maximum simultaneous child processes. Parallel workers are headless.

Monitor:

```bash
RUN=$(ls -dt outputs/direct_planner_stage5_v1_5_parallel/run_* | head -1)
tail -f "$RUN"/logs/seed_*.log
cat "$RUN/results.csv"
```

## Dataset

```text
RGB: 640 x 360
FPS: 30

outputs/episode_rgb_v1/episode_YYYYMMDD_HHMMSS/
├── rgb/
├── frames.jsonl
├── meta.json
└── planner_label.json
```

## Exact recorded-data visualization

```bash
EP=$(ls -dt outputs/episode_rgb_v1/episode_* | head -1)
python scripts/make_episode_preview.py "$EP"
```

Output: `<episode>/preview.mp4`.

## Git workflow

Integration branch:

```text
feature/air-hockey-expert-data-pipeline
```

Before merge to main:

```bash
PYTHONPATH=. python scripts/run_stage5_v1_5_parallel.py \
  --seed-start 1 --num-seeds 20 --workers 4 --coarse-only
```

Then run an 8-seed collection smoke test. After v1.5 edge-glide behavior is validated, the first milestone tag can be `air-hockey-expert-v0.1.0`.

## Current validation boundary

Established: deterministic seed episodes, direct counterattack search, PhysX forward simulation, selected-BEST replay, corrected wall-bounce behavior, synchronized RGB/state recording, planner-search isolation, and process-level parallel execution.

Still requiring larger regression: v1.5 edge-glide/BANK fallback, multi-seed success statistics, final physics calibration, final action representation, dataset integrity audit, and long-run collection stability.
