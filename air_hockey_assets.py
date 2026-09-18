"""Configurable SAPIEN primitives for an air-hockey table, puck and striker.

The assets remain independent of a robot or task. The caller owns the scene,
its time step, and any later gripper contact and control.
"""

from pathlib import Path
from typing import Any, Mapping

import sapien.core as sapien
import yaml


# SAPIEN cylinders point along local X; rotate their axis onto table normal Z.
_UPRIGHT = [0.7071067811865476, 0.0, -0.7071067811865476, 0.0]


def load_asset_config(path: str | Path) -> dict[str, Any]:
    """Load SI-valued geometry and physics settings, rejecting invalid shapes."""
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("air hockey asset config must be a mapping")
    for group in ("table", "puck", "striker"):
        if not isinstance(config.get(group), dict):
            raise ValueError(f"{group} must be a mapping")
    for group, fields in {
        "table": ("length_m", "width_m", "top_height_m", "thickness_m", "rail_width_m", "rail_height_m", "goal_width_m", "leg_width_m"),
        "puck": ("diameter_m", "thickness_m", "mass_kg"),
        "striker": ("diameter_m", "thickness_m", "mass_kg"),
    }.items():
        for field in fields:
            value = config[group].get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{group}.{field} must be positive")
    table = config["table"]
    if table["goal_width_m"] >= table["width_m"] - 2 * table["rail_width_m"]:
        raise ValueError("goal opening must leave room for end rails")
    if 2 * table["rail_width_m"] >= min(table["length_m"], table["width_m"]):
        raise ValueError("rail width leaves no playing surface")
    inertia = config["puck"].get("inertia_kg_m2")
    if not isinstance(inertia, list) or len(inertia) != 3 or any(not isinstance(x, (int, float)) or x <= 0 for x in inertia):
        raise ValueError("puck.inertia_kg_m2 must contain three positive values")
    return config


def configure_air_hockey_physics(config: Mapping[str, Any]) -> None:
    """Set the PhysX bounce threshold before creating the air-hockey scene.

    PhysX keeps this as process-wide scene configuration. The caller should
    preserve and restore its previous value if other scenes need it.
    """
    threshold = config["physics"]["bounce_threshold_m_s"]
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError("physics.bounce_threshold_m_s must be nonnegative")
    scene_config = sapien.physx.get_scene_config()
    scene_config.bounce_threshold = threshold
    sapien.physx.set_scene_config(scene_config)


def _material(scene: sapien.Scene, values: Mapping[str, Any], prefix: str = ""):
    return scene.create_physical_material(
        values[f"{prefix}static_friction"],
        values[f"{prefix}dynamic_friction"],
        values[f"{prefix}restitution"],
    )


def build_table(scene: sapien.Scene, config: Mapping[str, Any], *, render: bool = True):
    """Create a static table whose short ends have centered goal openings."""
    t = config["table"]
    length, width, top = t["length_m"], t["width_m"], t["top_height_m"]
    rail_w, rail_h = t["rail_width_m"], t["rail_height_m"]
    surface_mat = _material(scene, t, "surface_")
    rail_mat = _material(scene, t, "rail_")
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")

    def box(name, center, half_size, material, color):
        pose = sapien.Pose(center)
        builder.add_box_collision(pose=pose, half_size=half_size, material=material)
        if render:
            builder.add_box_visual(pose=pose, half_size=half_size, material=color, name=name)

    box("surface", [0, 0, top - t["thickness_m"] / 2],
        [length / 2, width / 2, t["thickness_m"] / 2], surface_mat,
        [0.88, 0.90, 0.94])
    for side in (-1, 1):
        box(f"side_rail_{side}", [0, side * (width / 2 - rail_w / 2), top + rail_h / 2],
            [length / 2, rail_w / 2, rail_h / 2], rail_mat, [0.12, 0.16, 0.22])

    end_segment = (width - 2 * rail_w - t["goal_width_m"]) / 2
    for end in (-1, 1):
        for side in (-1, 1):
            box(f"end_rail_{end}_{side}",
                [end * (length / 2 - rail_w / 2),
                 side * (t["goal_width_m"] / 2 + end_segment / 2), top + rail_h / 2],
                [rail_w / 2, end_segment / 2, rail_h / 2], rail_mat, [0.12, 0.16, 0.22])

    leg_w = t["leg_width_m"]
    leg_h = (top - t["thickness_m"]) / 2
    if leg_h > 0:
        for x in (-1, 1):
            for y in (-1, 1):
                box(f"leg_{x}_{y}",
                    [x * (length / 2 - leg_w), y * (width / 2 - leg_w), leg_h],
                    [leg_w / 2, leg_w / 2, leg_h], surface_mat, [0.32, 0.36, 0.42])
    return builder.build(name="air_hockey_table")


def build_puck(scene: sapien.Scene, config: Mapping[str, Any], *, render: bool = True,
               x: float = 0.0, y: float = 0.0, static: bool = False):
    """Create a puck on the surface, dynamic unless used for a fixed preview."""
    p, t = config["puck"], config["table"]
    radius, half_height = p["diameter_m"] / 2, p["thickness_m"] / 2
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if static else "dynamic")
    builder.add_cylinder_collision(pose=sapien.Pose(q=_UPRIGHT), radius=radius, half_length=half_height,
                                   material=_material(scene, p))
    if render:
        builder.add_cylinder_visual(pose=sapien.Pose(q=_UPRIGHT), radius=radius, half_length=half_height,
                                    material=[0.85, 0.12, 0.10], name="puck_disc")
    if not static:
        builder.set_mass_and_inertia(p["mass_kg"], sapien.Pose(), p["inertia_kg_m2"])
    builder.set_initial_pose(sapien.Pose([x, y, t["top_height_m"] + half_height + p["start_clearance_m"]]))
    puck = builder.build(name="air_hockey_puck")
    if not static:
        body = puck.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        body.linear_damping = p["linear_damping"]
        body.angular_damping = p["angular_damping"]
    return puck


def build_striker(scene: sapien.Scene, config: Mapping[str, Any], *, render: bool = True,
                  x: float = 0.4, y: float = 0.2, name: str = "air_hockey_striker",
                  static: bool = False):
    """Create a single-disc striker for later gripper clamping."""
    s, t = config["striker"], config["table"]
    radius, height = s["diameter_m"] / 2, s["thickness_m"]
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if static else "dynamic")
    material = _material(scene, s)
    builder.add_cylinder_collision(pose=sapien.Pose([0, 0, height / 2], _UPRIGHT),
                                   radius=radius, half_length=height / 2, material=material)
    if render:
        builder.add_cylinder_visual(pose=sapien.Pose([0, 0, height / 2], _UPRIGHT),
                                    radius=radius, half_length=height / 2,
                                    material=[0.12, 0.13, 0.16], name="striker_disc")
    if not static:
        mass = s["mass_kg"]
        i_xy = mass * (3 * radius ** 2 + height ** 2) / 12
        i_z = mass * radius ** 2 / 2
        builder.set_mass_and_inertia(mass, sapien.Pose([0, 0, height / 2]), [i_xy, i_xy, i_z])
    builder.set_initial_pose(sapien.Pose([x, y, t["top_height_m"] + 0.0005]))
    return builder.build(name=name)
