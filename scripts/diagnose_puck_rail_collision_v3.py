#!/usr/bin/env python3
"""
Isolated puck-vs-side-rail collision diagnostic v3 with threshold fixed in the actual scene config.

Purpose
-------
Remove robot/planner effects and directly measure whether the SAPIEN/PhysX
puck-rail contact rebounds or sticks.

Run from repository root:
    PYTHONPATH=. python scripts/diagnose_puck_rail_collision.py

Optional:
    PYTHONPATH=. python scripts/diagnose_puck_rail_collision.py --duration 3.0
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import sapien.core as sapien

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey_assets import (  # noqa: E402
    build_puck,
    build_table,
    configure_air_hockey_physics,
    load_asset_config,
)


def get_puck_body(puck):
    body = puck.find_component_by_type(
        sapien.physx.PhysxRigidDynamicComponent
    )
    if body is None:
        raise RuntimeError("puck has no PhysxRigidDynamicComponent")
    return body


def force_global_bounce_threshold(value):
    """Force SAPIEN-3 global PhysX scene config before scene creation."""
    value = float(value)
    try:
        cfg = sapien.physx.get_scene_config()
        cfg.bounce_threshold = value
        sapien.physx.set_scene_config(cfg)
        return float(sapien.physx.get_scene_config().bounce_threshold)
    except Exception:
        # Fallback for bindings that only expose the kwarg overload.
        sapien.physx.set_scene_config(
            bounce_threshold=value
        )
        return float(sapien.physx.get_scene_config().bounce_threshold)


def run_case(
    engine,
    base_cfg,
    *,
    name,
    surface_mu,
    puck_mu,
    rail_mu,
    rail_restitution,
    omega_z,
    puck_restitution=None,
    surface_restitution=None,
    vx=-0.30,
    vy=-0.90,
    duration=3.0,
):
    cfg = copy.deepcopy(base_cfg)

    # Keep static >= dynamic but do not exaggerate static friction.
    cfg["table"]["surface_dynamic_friction"] = float(surface_mu)
    cfg["table"]["surface_static_friction"] = float(max(surface_mu, 1.20 * surface_mu))

    cfg["puck"]["dynamic_friction"] = float(puck_mu)
    cfg["puck"]["static_friction"] = float(max(puck_mu, 1.20 * puck_mu))

    cfg["table"]["rail_dynamic_friction"] = float(rail_mu)
    cfg["table"]["rail_static_friction"] = float(max(rail_mu, 1.20 * rail_mu))
    cfg["table"]["rail_restitution"] = float(rail_restitution)

    if puck_restitution is not None:
        cfg["puck"]["restitution"] = float(puck_restitution)

    if surface_restitution is not None:
        cfg["table"]["surface_restitution"] = float(surface_restitution)

    configure_air_hockey_physics(cfg)

    requested_threshold = float(
        cfg["physics"]["bounce_threshold_m_s"]
    )

    # IMPORTANT: Engine.create_scene(config) re-applies the supplied scene
    # config. Use the desired threshold in that exact config, otherwise a new
    # default SceneConfig would silently reset it.
    scene_config = sapien.SceneConfig()
    scene_config.bounce_threshold = requested_threshold

    scene = engine.create_scene(scene_config)

    try:
        actual_threshold = float(
            sapien.physx.get_scene_config().bounce_threshold
        )
    except Exception:
        actual_threshold = float("nan")

    dt = float(cfg["physics"]["timestep_s"])
    scene.set_timestep(dt)

    build_table(scene, cfg, render=False)
    puck = build_puck(scene, cfg, render=False)
    body = get_puck_body(puck)

    # Let the puck settle vertically before the horizontal launch.
    for _ in range(max(1, int(round(0.20 / dt)))):
        scene.step()

    pose = puck.get_pose()
    puck.set_pose(
        sapien.Pose(
            [0.0, 0.0, float(pose.p[2])],
            np.asarray(pose.q, dtype=float),
        )
    )
    body.set_linear_velocity(
        np.asarray([vx, vy, 0.0], dtype=float)
    )
    body.set_angular_velocity(
        np.asarray([0.0, 0.0, omega_z], dtype=float)
    )

    table_width = float(cfg["table"]["width_m"])
    rail_width = float(cfg["table"]["rail_width_m"])
    puck_radius = float(cfg["puck"]["diameter_m"]) / 2.0
    y_limit = table_width / 2.0 - rail_width - puck_radius

    n_steps = int(round(float(duration) / dt))

    pre = None
    post = None
    min_y = +float("inf")
    max_abs_y = 0.0
    min_abs_normal_speed_near_wall = float("inf")
    near_wall_samples = 0

    history = []

    for i in range(n_steps + 1):
        if i > 0:
            scene.step()

        t = i * dt
        p = np.asarray(puck.get_pose().p, dtype=float)
        v = np.asarray(body.get_linear_velocity(), dtype=float)
        w = np.asarray(body.get_angular_velocity(), dtype=float)

        min_y = min(min_y, float(p[1]))
        max_abs_y = max(max_abs_y, abs(float(p[1])))

        # We're shooting toward the lower (-Y) side rail.
        near_wall = float(p[1]) <= (-y_limit + 0.012)

        if near_wall:
            near_wall_samples += 1
            min_abs_normal_speed_near_wall = min(
                min_abs_normal_speed_near_wall,
                abs(float(v[1])),
            )

        # Last inward-going sample close to the wall.
        if near_wall and float(v[1]) < -0.02:
            pre = {
                "t": t,
                "p": p.copy(),
                "v": v.copy(),
                "w": w.copy(),
            }

        # First clear outward-going sample after we have seen an inbound sample.
        if (
            pre is not None
            and post is None
            and float(v[1]) > +0.02
            and t > pre["t"]
        ):
            post = {
                "t": t,
                "p": p.copy(),
                "v": v.copy(),
                "w": w.copy(),
            }

        # Keep sparse history around the collision for manual diagnosis.
        if near_wall and (i % max(1, int(round(0.010 / dt))) == 0):
            history.append(
                (
                    t,
                    float(p[0]),
                    float(p[1]),
                    float(v[0]),
                    float(v[1]),
                    float(w[2]),
                )
            )

        # Once it has clearly rebounded away from the wall, enough data.
        if (
            post is not None
            and float(p[1]) > (-y_limit + 0.080)
        ):
            break

    print("")
    print("=" * 76)
    print(f"CASE: {name}")
    print("=" * 76)
    print(
        f"surface_mu={surface_mu:.4f}  puck_mu={puck_mu:.4f}  "
        f"rail_mu={rail_mu:.4f}  rail_e={rail_restitution:.4f}  "
        f"omega_z0={omega_z:+.3f} rad/s"
    )
    print(
        f"bounce threshold requested/actual = "
        f"{requested_threshold:.4f}/{actual_threshold:.4f} m/s"
    )
    print(
        f"surface_e={cfg['table']['surface_restitution']:.3f}  "
        f"puck_e={cfg['puck']['restitution']:.3f}"
    )
    print(f"launch v=({vx:+.3f},{vy:+.3f}) m/s")
    print(f"expected puck-center rail limit |y| ~= {y_limit:.5f} m")
    print(f"observed min y = {min_y:+.5f} m")

    if pre is not None:
        print(
            "PRE : "
            f"t={pre['t']:.4f}s "
            f"p=({pre['p'][0]:+.4f},{pre['p'][1]:+.4f}) "
            f"v=({pre['v'][0]:+.4f},{pre['v'][1]:+.4f}) "
            f"wz={pre['w'][2]:+.4f}"
        )
    else:
        print("PRE : no clear inbound near-wall sample detected")

    if post is not None:
        print(
            "POST: "
            f"t={post['t']:.4f}s "
            f"p=({post['p'][0]:+.4f},{post['p'][1]:+.4f}) "
            f"v=({post['v'][0]:+.4f},{post['v'][1]:+.4f}) "
            f"wz={post['w'][2]:+.4f}"
        )

        if pre is not None and abs(float(pre["v"][1])) > 1e-9:
            measured_e_n = (
                abs(float(post["v"][1]))
                / abs(float(pre["v"][1]))
            )
            print(f"measured normal rebound ratio |vy_out/vy_in| = {measured_e_n:.3f}")

        if pre is not None and abs(float(pre["v"][0])) > 1e-9:
            tangential_ratio = (
                float(post["v"][0])
                / float(pre["v"][0])
            )
            print(f"measured tangential ratio vx_out/vx_in = {tangential_ratio:.3f}")

        print("classification: REBOUND")
    else:
        # Distinguish "never reached wall" from "reached wall but no rebound".
        if near_wall_samples > 0:
            print("POST: no outward vy > 0.02 m/s detected after rail approach")
            print(
                "classification: POSSIBLE_STICK / NON-REBOUND "
                f"(near-wall samples={near_wall_samples})"
            )
        else:
            print("POST: puck never reached the side rail")
            print("classification: NO_IMPACT")

    print("")
    print("near-wall samples (10 ms spacing), first 20:")
    print("   t[s]       x        y       vx       vy       wz")
    for row in history[:20]:
        print(
            f"{row[0]:7.3f} "
            f"{row[1]:+8.4f} {row[2]:+8.4f} "
            f"{row[3]:+8.4f} {row[4]:+8.4f} "
            f"{row[5]:+8.3f}"
        )

    return {
        "name": name,
        "pre": pre,
        "post": post,
        "near_wall_samples": near_wall_samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "env_cfg" / "air_hockey_assets.yml",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
    )
    args = parser.parse_args()

    base_cfg = load_asset_config(args.config)

    engine = sapien.Engine()

    # A: ideal rail sanity check. If THIS does not rebound, the issue is not
    # planner logic and not the chosen random material values.
    cases = [
        dict(
            name="A_IDEAL_ELASTIC_ZERO_FRICTION",
            surface_mu=0.0,
            puck_mu=0.0,
            rail_mu=0.0,
            rail_restitution=1.0,
            omega_z=0.0,
            puck_restitution=1.0,
            surface_restitution=1.0,
        ),
        dict(
            name="B_BASELINE_YAML_ZERO_SPIN",
            surface_mu=0.02,
            puck_mu=0.02,
            rail_mu=0.10,
            rail_restitution=0.80,
            omega_z=0.0,
        ),
        dict(
            name="C_SEED6_PHYSICS_ZERO_SPIN",
            surface_mu=0.0266,
            puck_mu=0.0156,
            rail_mu=0.1302,
            rail_restitution=0.7346,
            omega_z=0.0,
        ),
        dict(
            name="D_SEED6_PHYSICS_WITH_SPIN",
            surface_mu=0.0266,
            puck_mu=0.0156,
            rail_mu=0.1302,
            rail_restitution=0.7346,
            omega_z=-3.099,
        ),
    ]

    print("")
    print("PUCK-RAIL COLLISION DIAGNOSTIC")
    print("Planner/robot are NOT involved.")
    print(f"config: {args.config}")
    print(f"dt    : {base_cfg['physics']['timestep_s']} s")
    print(f"bounce threshold: {base_cfg['physics']['bounce_threshold_m_s']} m/s")

    for case in cases:
        run_case(
            engine,
            base_cfg,
            duration=args.duration,
            **case,
        )


if __name__ == "__main__":
    main()
