"""Mouse-driven joint-limited IK control for the UR5 preview viewer."""

import numpy as np
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin
from sapien.utils.viewer.transform_window import TransformWindow


def valid_ik_solution(success, qpos, limits):
    """Accept only converged, finite IK solutions within URDF joint limits."""
    qpos = np.asarray(qpos, dtype=float)
    limits = np.asarray(limits, dtype=float)
    return bool(success and qpos.ndim == 1 and limits.shape == (len(qpos), 2)
                and np.all(np.isfinite(qpos))
                and np.all(qpos >= limits[:, 0] - 1e-6)
                and np.all(qpos <= limits[:, 1] + 1e-6))


class UR5DragController(Plugin):
    """Select a UR5 tool frame and apply valid SAPIEN gizmo IK during dragging."""

    def __init__(self, assembly, config):
        self.robots = {side: assembly[f"{side}_robot"] for side in ("left", "right")}
        self.ee_entities = {
            side: next(link.entity for link in robot.get_links()
                       if link.get_name() == config["ee_link_name"])
            for side, robot in self.robots.items()
        }
        self.arm_joint_names = set(config["arm_joint_names"])
        self.selected_side = None
        self.status = "Select an arm, then drag its tool-frame axes"
        self._previous_keys = {"q": False, "e": False}
        self._last_target = None
        self._ui_window = None

    def init(self, viewer):
        super().init(viewer)
        self.transform = next(p for p in viewer.plugins if isinstance(p, TransformWindow))
        # Keep the selected tool frame while clicking a gizmo handle.
        viewer.control_window.register_click_handler(lambda *_: True)

    def select(self, side):
        if side not in self.robots:
            raise ValueError(f"unknown robot side: {side}")
        self.viewer.select_entity(self.ee_entities[side])
        self.selected_side = side
        self.transform.enabled = True
        self.transform.ik_enabled = True
        self.transform.move_group_selection = [
            name in self.arm_joint_names for name in self.transform.move_group_joints
        ]
        self._last_target = np.array(self.transform.gizmo_matrix, copy=True)
        self.status = f"Selected {side} UR5: drag the colored axes"

    def apply_gizmo_target(self):
        if (self.selected_side is None or self.transform.follow
                or self.viewer.selected_entity is not self.ee_entities[self.selected_side]):
            return
        target = np.array(self.transform.gizmo_matrix, copy=True)
        if self._last_target is not None and np.allclose(target, self._last_target,
                                                         atol=1e-7, rtol=0):
            return
        self._last_target = target
        robot = self.robots[self.selected_side]
        result = self.transform.ik_result
        if result is None or not valid_ik_solution(self.transform.ik_success, result,
                                                   robot.get_qlimits()):
            self.status = "Unreachable: IK failed or a joint limit was exceeded"
            return
        for joint, target_value in zip(robot.get_active_joints(), result):
            if joint.name in self.arm_joint_names:
                joint.set_drive_target(float(target_value))
        self.viewer.notify_render_update()
        self.status = f"Reachable: driving {self.selected_side} UR5 tool frame"

    def after_render(self):
        if self.viewer.closed:
            return
        for key, side in (("q", "left"), ("e", "right")):
            down = bool(self.viewer.window.key_down(key))
            if down and not self._previous_keys[key]:
                self.select(side)
            self._previous_keys[key] = down
        self.apply_gizmo_target()

    def get_ui_windows(self):
        if self._ui_window is None:
            self._status_text = R.UIDisplayText()
            self._ui_window = (
                R.UIWindow().Label("UR5 Reachability").Pos(820, 10).Size(360, 170)
                .append(
                    R.UIButton().Label("Left UR5 (Q)").Callback(lambda _: self.select("left")),
                    R.UIButton().Label("Right UR5 (E)").Callback(lambda _: self.select("right")),
                    R.UIDisplayText().Text("Drag the colored axes at the selected tool frame"),
                    self._status_text,
                )
            )
        self._status_text.Text(self.status)
        return [self._ui_window]


def install_drag_controller(viewer, assembly, config):
    """Install per-viewer controls without mutating SAPIEN's default plugin list."""
    controller = UR5DragController(assembly, config)
    controller.init(viewer)
    viewer.plugins = [*viewer.plugins, controller]
    return controller
