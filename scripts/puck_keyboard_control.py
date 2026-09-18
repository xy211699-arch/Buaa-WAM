"""Select the puck with the mouse and give it one impulse per number-key press."""

import numpy as np
import sapien.core as sapien
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin


_KEY_VECTORS = {
    "7": (-1, 1), "8": (0, 1), "9": (1, 1),
    "4": (-1, 0),                 "6": (1, 0),
    "1": (-1, -1), "2": (0, -1), "3": (1, -1),
}


def direction_for_key(key):
    """Use the numeric keypad layout: X right, Y up, diagonals normalized."""
    vector = np.asarray(_KEY_VECTORS[key], dtype=float)
    return vector / np.linalg.norm(vector)


def apply_directional_impulse(puck, key, *, impulse_ns):
    """Give a selected puck an instantaneous horizontal momentum change."""
    if not np.isfinite(impulse_ns) or impulse_ns <= 0:
        raise ValueError("impulse_ns must be positive")
    direction = direction_for_key(key)
    body = puck.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    if body is None:
        raise TypeError("the puck must be a dynamic rigid body")
    body.add_force_at_point([*(direction * impulse_ns), 0], puck.pose.p, mode="impulse")


class PuckKeyboardController(Plugin):
    """Give keyboard impulses only while the clicked puck is highlighted."""

    def __init__(self, puck, config, arm_controller=None):
        self.puck = puck
        self.config = config
        self.arm_controller = arm_controller
        self.selected = False
        self.status = "Click the puck to select"
        self._previous_keys = {key: False for key in _KEY_VECTORS}
        self._original_materials = []
        self._ui_window = None

    def init(self, viewer):
        super().init(viewer)
        # The arm controller preserves its gizmo selection by consuming clicks.
        # Puck picking must run before that handler.
        viewer.control_window.click_handlers.insert(0, self.on_click)
        render_body = self.puck.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_body is not None:
            for shape in render_body.render_shapes:
                material = shape.material
                self._original_materials.append(
                    (material, list(material.base_color), list(material.emission)))

    def _set_highlight(self, enabled):
        for material, base, emission in self._original_materials:
            material.base_color = [1.0, 0.92, 0.12, 1.0] if enabled else base
            material.emission = [0.45, 0.38, 0.02, 1.0] if enabled else emission

    def deselect(self):
        self.selected = False
        self._set_highlight(False)
        self.status = "Click the puck to select"
        if self.viewer.selected_entity is self.puck:
            self.viewer.select_entity(None)

    def on_click(self, viewer, x, y):
        pixel = viewer.window.get_picture_pixel("Segmentation", x, y)
        hit = (int(pixel[1]) == self.puck.per_scene_id
               and int(pixel[2]) == self.puck.scene.id)
        if hit:
            viewer.select_entity(self.puck)
            self.selected = True
            self._set_highlight(True)
            self.status = "Selected: press 1-4 or 6-9"
            if self.arm_controller is not None:
                self.arm_controller.transform.enabled = False
            return True
        if self.selected:
            self.deselect()
            if self.arm_controller is not None and self.arm_controller.selected_side is not None:
                self.arm_controller.select(self.arm_controller.selected_side)
            return True
        return False

    def after_render(self):
        if self.viewer.closed:
            return
        if self.selected and self.viewer.selected_entity is not self.puck:
            self.selected = False
            self._set_highlight(False)
            self.status = "Click the puck to select"
        for key in _KEY_VECTORS:
            down = bool(self.viewer.window.key_down(key))
            if self.selected and down and not self._previous_keys[key]:
                apply_directional_impulse(
                    self.puck, key,
                    impulse_ns=self.config["interaction"]["keyboard_impulse_Ns"])
                self.status = f"Impulse {key}: {self.config['interaction']['keyboard_impulse_Ns']:.3f} N s"
            self._previous_keys[key] = down

    def get_ui_windows(self):
        if self._ui_window is None:
            self._status_text = R.UIDisplayText()
            self._ui_window = (
                R.UIWindow().Label("Puck Control").Pos(820, 190).Size(360, 130)
                .append(R.UIDisplayText().Text("7 8 9 / 4 - 6 / 1 2 3"),
                        R.UIDisplayText().Text("8:+Y  2:-Y  4:-X  6:+X"),
                        self._status_text)
            )
        self._status_text.Text(self.status)
        return [self._ui_window]


def install_puck_keyboard_controller(viewer, puck, config, arm_controller=None):
    controller = PuckKeyboardController(puck, config, arm_controller)
    controller.init(viewer)
    viewer.plugins = [*viewer.plugins, controller]
    return controller
