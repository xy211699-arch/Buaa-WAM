# Air hockey asset prototype

`air_hockey_assets.py` creates the table, puck and striker directly as
SAPIEN rigid bodies. These procedural assets need no STL or URDF file. Their
dimensions and physical parameters live in `env_cfg/air_hockey_assets.yml` in
metres, kilograms and seconds. Values marked `provisional: true` are starting
values for geometry and physics checks, not a calibrated real table.

The table length and width, puck diameter, puck thickness, puck mass and puck
inertia start from [Air Hockey Challenge's MuJoCo table definition](https://github.com/AirHockeyChallenge/air_hockey_challenge/blob/qualifying-2025/air_hockey_challenge/environments/data/table.xml).
The striker diameter, thickness and mass start from its
[iiwa striker definition](https://github.com/AirHockeyChallenge/air_hockey_challenge/blob/qualifying-2025/air_hockey_challenge/environments/data/iiwas/iiwa1.xml).
The striker here is a single cylindrical disc without a handle. Its mass is
provisional for this shape. MuJoCo friction/contact numbers are not copied into
PhysX: the material settings in the YAML are initial SAPIEN tuning values.

The table top is at `table.top_height_m` above the scene origin. The playing
length is the X axis; goal openings are centered at both X ends. The Y axis
crosses the table width; Z is upward. Puck and striker positions are world XY
coordinates. The striker's entity origin is at the bottom center of its disc;
the disc rises in +Z. The WSG fingers physically contact its sides. There is
no joint or weld between the striker and robot.

To open the **interactive UR5 physics preview** from the repository root, use:

```bash
conda activate RoboTwin
python scripts/preview_air_hockey_assets.py
```

The preview closes after three minutes by default. Pass `--duration 0` to
leave it open until you close the window.

The preview has one table, a dynamic puck at the center, and two dynamic disc
strikers held by the grippers near X = -0.85 m and +0.85 m. It loads two UR5-WSG arms with their
built-in grippers and fixed roots. Their bases sit on independent supports
behind the goal centers at X = -1.244 m and +1.244 m, Y = 0. Each support top
is level with the 0.74 m playing surface. The arms face inward.
Both `ee_link` frames start on the goal centerline at X = -0.85 m and +0.85 m,
0.16 m above the playing surface.

In the viewer, press `Q` or `E`, or click **Left UR5** or **Right UR5** in the
**UR5 Reachability** panel. Drag the colored axes at the selected gripper's
`ee_link` frame with the left mouse button. The arm follows each target that
SAPIEN's IK solver can reach within the URDF joint limits. The panel reports
**Reachable** or **Unreachable**; unreachable targets leave the last valid joint pose in
place. The six arm joints are active for IK; the two gripper joints use
force-limited drives to maintain contact with the disc.
The bases stay fixed, and the scene advances physics at a 1 ms step using
elapsed wall-clock time. Robot
joint drives move to the IK target against gravity and puck impacts. The
striker discs collide with the fingers, puck, table and rails. The fingers
start open and close during the preview's initial settling steps. This tool checks
kinematic reach and joint limits; it does not check collision-free paths or
robot speed. The model path,
base setback, support dimensions, and home joint pose are in
`env_cfg/air_hockey_assets.yml` under `robot_preview`.
The bundled UR5 URDF limits `shoulder_pan_joint` to ±180°, while the
[UR5 specification](https://www.universal-robots.com/media/50588/ur5_en.pdf)
lists ±360°. The preview widens this joint to ±360° at runtime, without
editing the shared URDF, to avoid a false lateral reach boundary near the
goal centerline. The override is in `robot_preview.shoulder_pan_limits_rad`.

To move the puck, click it once. It changes from red to bright yellow when
selected. Then press one of the **number-row** keys `7 8 9 / 4 6 / 1 2 3`.
These follow a keypad direction layout: `8` is +Y, `2` is -Y, `4` is -X,
`6` is +X, and diagonal keys combine the adjacent directions. Each keypress
applies one horizontal impulse at the puck center. Holding a key does not
repeat the impulse. The magnitude is `interaction.keyboard_impulse_Ns` in the
YAML; diagonal impulses have the same magnitude as straight ones. Clicking
elsewhere deselects the puck. `Q` and `E` select the left and right robot.
The force-limited grip can slip under sufficiently large disturbances; its
contact coefficients and drive limits are provisional values in `grasp`.
The preview camera starts with the full table in view. Both WSG grippers start
with their fingers separated along the table's Y axis.

The window requires the SAPIEN Vulkan renderer and a working graphical desktop.
To check the asset actors and planned robot mounts without a graphics device:

```bash
python scripts/preview_air_hockey_assets.py --check
```

Example use in a **new** scene (the physics setup must run before scene
creation):

```python
import sapien.core as sapien

from air_hockey_assets import (
    build_puck, build_striker, build_table,
    configure_air_hockey_physics, load_asset_config,
)

cfg = load_asset_config("env_cfg/air_hockey_assets.yml")
configure_air_hockey_physics(cfg)
scene = sapien.Scene()  # use RoboTwin's renderer setup in an actual task
scene.set_timestep(cfg["physics"]["timestep_s"])
table = build_table(scene, cfg)
puck = build_puck(scene, cfg)
left_striker = build_striker(scene, cfg, x=-0.4, y=0, name="left_striker")
right_striker = build_striker(scene, cfg, x=0.4, y=0, name="right_striker")
```

`configure_air_hockey_physics()` changes the process-wide PhysX bounce
threshold. Save and restore the prior scene config when creating unrelated
scenes in the same process. With PhysX's 2 m/s default threshold, a slow puck
stops at the rail rather than rebounding. The provided 0.1 m/s setting was
verified in a physics-only scene and remains subject to real-world tuning.

The current RoboTwin task base creates its own 1.2 × 0.7 m workbench. A
dedicated air-hockey task will need to use these builders in place of that
workbench. The preview uses the viewer camera for inspection; it does not yet
install a dedicated observation camera or record trajectories.

Run the asset, physical grasp, puck-control and UR5 checks with:

```bash
ROBOTWIN_GPU_TEST=1 python -m unittest tests.test_air_hockey_assets tests.test_air_hockey_interaction tests.test_ur5_reachability -v
```
