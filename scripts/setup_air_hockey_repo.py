#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import py_compile
import shutil
import subprocess
from pathlib import Path

BRANCH = "feature/air-hockey-expert-data-pipeline"
CANONICAL = [
    "scripts/collect_episode_rgb_auto_v1.py",
    "scripts/plan_direct_counterattack_stage5_v1_5.py",
    "scripts/run_stage5_v1_5_parallel.py",
]

README_SECTION = r'''
## BUAA-WAM Air Hockey Research Pipeline

This fork includes a two-UR5 air-hockey simulator expert and synchronized RGB/state data-collection pipeline.

### Current entry points

```text
Expert planner:
scripts/plan_direct_counterattack_stage5_v1_5.py

Parallel collector:
scripts/run_stage5_v1_5_parallel.py

RGB/state recorder:
scripts/collect_episode_rgb_auto_v1.py

Collected-episode visualization:
scripts/make_episode_preview.py
```

### Current pipeline

```text
randomized incoming puck
-> large-angle / spin / physics randomization
-> exact PhysX future rollout
-> bounce-aware contact-state generation
-> goal-conditioned direct counterattack search
-> IK / workspace feasibility
-> candidate forward simulation
-> restore exact initial state
-> execute selected BEST only
-> synchronized RGB + state recording
```

The planner is a privileged simulator expert. The intended learned policy later uses RGB history and robot proprioception rather than requiring puck ground truth at deployment.

### Quick start

```bash
conda activate hockey

PYTHONPATH=. python scripts/plan_direct_counterattack_stage5_v1_5.py \
  --seed 6 --coarse-only --hold 50
```

Parallel expert-data collection:

```bash
PYTHONPATH=. python scripts/run_stage5_v1_5_parallel.py \
  --seed-start 1 --num-seeds 8 --workers 4 --coarse-only --collect
```

Visualize the exact collected RGB trajectory:

```bash
EP=$(ls -dt outputs/episode_rgb_v1/episode_* | head -1)
python scripts/make_episode_preview.py "$EP"
```

The recorder currently stores RGB at 640x360 and 30 FPS together with synchronized robot/puck state. Candidate-search rollouts are excluded from the expert demonstration timeline.

Detailed documentation:

```text
docs/air_hockey_expert_pipeline.md
```

Incremental dependencies:

```bash
pip install -r requirements-air-hockey.txt
```
'''.strip()

PIPELINE_DOC = r'''
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
'''.strip()

REQUIREMENTS = '''# Incremental BUAA-WAM air-hockey dependencies.\n# Install normal RoboTwin / XPolicyLab first.\n# Development environment: Python 3.10\n\nnumpy>=1.24\nPyYAML>=6.0\nimageio>=2.31\nimageio-ffmpeg>=0.4\nsapien>=3.0,<4\n\n# Linux/POSIX recommended because parallel collection uses fcntl.\n# Interactive SAPIEN Viewer requires a working Vulkan stack.\n'''

PREVIEW_SCRIPT = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio


def main():
    p = argparse.ArgumentParser(description="Build preview.mp4 from a collected air-hockey episode")
    p.add_argument("episode_dir", type=Path)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()

    episode = a.episode_dir.expanduser().resolve()
    rgb = episode / "rgb"
    if not rgb.is_dir():
        raise FileNotFoundError(rgb)

    frames = sorted(x for x in rgb.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not frames:
        raise RuntimeError(f"no frames under {rgb}")

    fps = a.fps
    meta = episode / "meta.json"
    if fps is None and meta.is_file():
        try:
            fps = float(json.loads(meta.read_text(encoding="utf-8")).get("fps", 30.0))
        except Exception:
            fps = 30.0
    fps = 30.0 if fps is None else fps
    if fps <= 0:
        raise ValueError("fps must be > 0")

    out = a.output.expanduser().resolve() if a.output else episode / "preview.mp4"
    with imageio.get_writer(str(out), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None) as w:
        for f in frames:
            w.append_data(imageio.imread(f))

    print("episode:", episode)
    print("frames :", len(frames))
    print("fps    :", fps)
    print("output :", out)


if __name__ == "__main__":
    main()
'''


def root_dir():
    cwd = Path.cwd().resolve()
    if (cwd / "scripts").is_dir():
        return cwd
    here = Path(__file__).resolve().parents[1]
    if (here / "scripts").is_dir():
        return here
    raise RuntimeError("Run from Buaa-WAM root or place this file in scripts/.")


def write(path, text, apply):
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        print("[WRITE]", path)
    else:
        print("[WOULD WRITE]", path)


def patch_readme(root, apply):
    path = root / "README.md"
    if not path.exists():
        write(path, "# BUAA-WAM\n\n" + README_SECTION, apply)
        return

    text = path.read_text(encoding="utf-8")
    markers = [
        "## Air Hockey V2: DEFEND + HIT + RGB Data Collection",
        "## BUAA-WAM Air Hockey Research Pipeline",
    ]
    hits = [text.find(m) for m in markers if text.find(m) >= 0]
    new = (text[:min(hits)].rstrip() + "\n\n" if hits else text.rstrip() + "\n\n") + README_SECTION + "\n"

    if apply:
        backup = root / "README.pre_air_hockey_setup.bak.md"
        if not backup.exists():
            shutil.copy2(path, backup)
            print("[BACKUP]", backup)
        path.write_text(new, encoding="utf-8")
        print("[UPDATE] README.md")
    else:
        print("[WOULD UPDATE] README.md")


def patch_gitignore(root, apply):
    path = root / ".gitignore"
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    entries = [
        "outputs/episode_rgb_v1/",
        "outputs/direct_planner_stage5_v1_5/",
        "outputs/direct_planner_stage5_v1_5_parallel/",
        "outputs/**/*.mp4",
        "*.log",
        "__pycache__/",
        "*.pyc",
    ]
    missing = [x for x in entries if x not in old]
    if not missing:
        print("[OK] .gitignore")
        return
    if apply:
        with path.open("a", encoding="utf-8") as f:
            if old and not old.endswith("\n"):
                f.write("\n")
            f.write("\n# BUAA-WAM air-hockey generated artifacts\n")
            for x in missing:
                f.write(x + "\n")
        print("[UPDATE] .gitignore")
    else:
        print(f"[WOULD UPDATE] .gitignore ({len(missing)} entries)")


def setup_branch(root, apply):
    if not (root / ".git").exists() or shutil.which("git") is None:
        print("[SKIP] Git unavailable")
        return
    r = subprocess.run(["git", "branch", "--list", BRANCH], cwd=root, capture_output=True, text=True, check=True)
    exists = bool(r.stdout.strip())
    if not apply:
        print(f"[WOULD {'SWITCH' if exists else 'CREATE'}] {BRANCH}")
        return
    cmd = ["git", "switch", BRANCH] if exists else ["git", "switch", "-c", BRANCH]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


def syntax_check(root):
    print("\n=== syntax check ===")
    failed = []
    for rel in CANONICAL + ["scripts/make_episode_preview.py"]:
        p = root / rel
        if not p.exists():
            print("[MISSING]", rel)
            failed.append(rel)
            continue
        try:
            py_compile.compile(str(p), doraise=True)
            print("[OK]", rel)
        except Exception as e:
            print("[FAIL]", rel, e)
            failed.append(rel)
    return failed


def main():
    ap = argparse.ArgumentParser(description="One-command BUAA-WAM air-hockey repository organizer")
    ap.add_argument("--apply", action="store_true", help="Actually modify files; default is dry-run")
    ap.add_argument("--branch", action="store_true", help="Create/switch recommended feature branch")
    args = ap.parse_args()

    root = root_dir()
    print("repository:", root)
    print("mode      :", "APPLY" if args.apply else "DRY RUN")

    print("\n=== canonical scripts ===")
    for rel in CANONICAL:
        print(f"[{'OK' if (root / rel).exists() else 'MISSING'}] {rel}")

    if args.branch:
        print("\n=== git branch ===")
        setup_branch(root, args.apply)

    print("\n=== repository setup ===")
    write(root / "requirements-air-hockey.txt", REQUIREMENTS, args.apply)
    write(root / "docs" / "air_hockey_expert_pipeline.md", PIPELINE_DOC, args.apply)
    write(root / "scripts" / "make_episode_preview.py", PREVIEW_SCRIPT, args.apply)
    patch_gitignore(root, args.apply)
    patch_readme(root, args.apply)

    failed = syntax_check(root) if args.apply else []

    print("\n==================================================")
    print("SETUP COMPLETE" if args.apply else "DRY RUN COMPLETE - NO FILES CHANGED")
    print("==================================================")

    if not args.apply:
        print("\nApply with:")
        print("python scripts/setup_air_hockey_repo.py --apply --branch")
        return

    print("\nNext:")
    print("git status")
    print("PYTHONPATH=. python scripts/run_stage5_v1_5_parallel.py --seed-start 1 --num-seeds 20 --workers 4 --coarse-only")
    print("PYTHONPATH=. python scripts/run_stage5_v1_5_parallel.py --seed-start 1 --num-seeds 8 --workers 4 --coarse-only --collect")
    print('EP=$(ls -dt outputs/episode_rgb_v1/episode_* | head -1)')
    print('python scripts/make_episode_preview.py "$EP"')

    if failed:
        print("\nWARNING: fix missing/syntax failures before commit.")
    else:
        print("\nCore entry points passed syntax checks.")


if __name__ == "__main__":
    main()
