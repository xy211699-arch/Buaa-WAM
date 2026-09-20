import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import sapien.core as sapien

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from air_hockey_assets import (
    configure_air_hockey_physics,
    load_asset_config,
)

from scripts.preview_air_hockey_assets import (
    populate_robot_preview,
)


def look_at_pose(
    eye,
    target,
    world_up=(0.0, 0.0, 1.0),
):
    eye = np.asarray(
        eye,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    up_ref = np.asarray(
        world_up,
        dtype=float,
    )

    forward = target - eye
    forward /= np.linalg.norm(forward)

    left = np.cross(
        up_ref,
        forward,
    )

    left_norm = np.linalg.norm(left)

    if left_norm < 1e-8:
        raise RuntimeError(
            "camera look-at singularity"
        )

    left /= left_norm

    up = np.cross(
        forward,
        left,
    )

    T = np.eye(
        4,
        dtype=float,
    )

    # SAPIEN camera local convention:
    # +X forward, +Y left, +Z up
    T[:3, :3] = np.stack(
        [
            forward,
            left,
            up,
        ],
        axis=1,
    )

    T[:3, 3] = eye

    return sapien.Pose(
        T
    )


def set_camera_pose(
    camera,
    pose,
):
    if hasattr(
        camera,
        "set_pose",
    ):
        camera.set_pose(
            pose
        )
        return

    if hasattr(
        camera,
        "entity",
    ):
        camera.entity.set_pose(
            pose
        )
        return

    raise RuntimeError(
        "不知道当前 SAPIEN camera "
        "应该如何 set pose"
    )


def main():
    print("")
    print(
        "=============================================="
    )
    print(
        "RGB CAMERA SMOKE TEST"
    )
    print(
        "=============================================="
    )

    cfg_path = (
        ROOT
        / "env_cfg"
        / "air_hockey_assets.yml"
    )

    config = load_asset_config(
        cfg_path
    )

    configure_air_hockey_physics(
        config
    )

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()

    engine.set_renderer(
        renderer
    )

    scene = engine.create_scene(
        sapien.SceneConfig()
    )

    scene.set_timestep(
        config["physics"]["timestep_s"]
    )

    scene.set_ambient_light(
        [0.5, 0.5, 0.5]
    )

    scene.add_directional_light(
        [0, 0.5, -1],
        [0.7, 0.7, 0.7],
        shadow=True,
    )

    assembly = populate_robot_preview(
        scene,
        config,
    )

    print(
        "[CAMERA API] "
        f"scene.add_camera={hasattr(scene, 'add_camera')}"
    )

    if not hasattr(
        scene,
        "add_camera",
    ):
        raise RuntimeError(
            "scene.add_camera 不存在"
        )

    width = 640
    height = 360
    fovy = np.deg2rad(
        55.0
    )

    camera = scene.add_camera(
        "rgb_front",
        width,
        height,
        fovy,
        0.05,
        10.0,
    )

    print(
        "[CAMERA] type:",
        type(camera),
    )

    print(
        "[CAMERA] has set_pose:",
        hasattr(
            camera,
            "set_pose",
        ),
    )

    print(
        "[CAMERA] has entity:",
        hasattr(
            camera,
            "entity",
        ),
    )

    print(
        "[CAMERA] has take_picture:",
        hasattr(
            camera,
            "take_picture",
        ),
    )

    print(
        "[CAMERA] has get_picture:",
        hasattr(
            camera,
            "get_picture",
        ),
    )

    # 斜上方看整个球桌。
    camera_pose = look_at_pose(
        eye=[
            -0.15,
            -1.75,
            2.35,
        ],
        target=[
            0.0,
            0.0,
            float(
                config["table"]["top_height_m"]
            ),
        ],
    )

    set_camera_pose(
        camera,
        camera_pose,
    )

    # 让场景先稳定。
    settle_steps = int(
        config["grasp"]["settle_steps"]
    )

    for _ in range(
        settle_steps
    ):
        scene.step()

    scene.update_render()

    camera.take_picture()

    color = np.asarray(
        camera.get_picture(
            "Color"
        )
    )

    print(
        "[RGB] raw shape:",
        color.shape,
    )

    print(
        "[RGB] raw dtype:",
        color.dtype,
    )

    print(
        "[RGB] min/max:",
        float(
            np.nanmin(color)
        ),
        float(
            np.nanmax(color)
        ),
    )

    if (
        color.ndim != 3
        or color.shape[2] < 3
    ):
        raise RuntimeError(
            f"Color shape 异常: {color.shape}"
        )

    rgb = np.clip(
        color[..., :3]
        * 255.0,
        0.0,
        255.0,
    ).astype(
        np.uint8
    )

    out_dir = (
        ROOT
        / "outputs"
        / "rgb_camera_smoke"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        out_dir
        / "frame_000000.png"
    )

    iio.imwrite(
        out_path,
        rgb,
    )

    print("")
    print(
        "[RGB] SAVED:",
        out_path
    )

    print(
        "[RGB] resolution:",
        f"{rgb.shape[1]}x{rgb.shape[0]}"
    )

    print(
        "[RGB] mean:",
        float(
            rgb.mean()
        ),
    )

    print("")
    print(
        "=============================================="
    )
    print(
        "RGB CAMERA SMOKE PASS"
    )
    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
