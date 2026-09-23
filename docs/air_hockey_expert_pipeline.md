# BUAA-WAM Air Hockey Expert Pipeline

## Canonical entry points

```text
scripts/plan_direct_counterattack_stage5_v1_5.py
scripts/run_stage5_v1_5_parallel.py
scripts/collect_episode_rgb_auto_v1.py
scripts/make_episode_preview.py
```

The current planner is a privileged simulator expert, not an RL policy.

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
