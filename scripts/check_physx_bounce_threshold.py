#!/usr/bin/env python3
"""
Check whether BUAA-WAM actually applies the configured PhysX bounce threshold.

Run:
    PYTHONPATH=. python scripts/check_physx_bounce_threshold.py
"""

import sys
from pathlib import Path

import sapien.core as sapien

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey_assets import configure_air_hockey_physics, load_asset_config


def get_actual_threshold():
    try:
        return float(sapien.physx.get_scene_config().bounce_threshold)
    except Exception as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def main():
    cfg_path = ROOT / "env_cfg" / "air_hockey_assets.yml"
    cfg = load_asset_config(cfg_path)

    print("")
    print("============================================================")
    print("PHYSX BOUNCE THRESHOLD CHECK")
    print("============================================================")
    print(f"YAML requested threshold : {cfg['physics']['bounce_threshold_m_s']} m/s")
    print(f"actual BEFORE configure  : {get_actual_threshold()}")

    configure_air_hockey_physics(cfg)

    print(f"actual AFTER configure   : {get_actual_threshold()}")

    print("")
    print("Now force the SAPIEN-3 global PhysX scene config explicitly...")

    try:
        current = sapien.physx.get_scene_config()
        current.bounce_threshold = float(cfg["physics"]["bounce_threshold_m_s"])
        sapien.physx.set_scene_config(current)
        print(f"actual AFTER direct set  : {get_actual_threshold()}")
    except Exception as exc:
        print(
            "direct config-object set failed:",
            type(exc).__name__,
            str(exc),
        )
        try:
            sapien.physx.set_scene_config(
                bounce_threshold=float(
                    cfg["physics"]["bounce_threshold_m_s"]
                )
            )
            print(f"actual AFTER kwarg set   : {get_actual_threshold()}")
        except Exception as exc2:
            print(
                "direct kwarg set failed:",
                type(exc2).__name__,
                str(exc2),
            )

    print("============================================================")


if __name__ == "__main__":
    main()
