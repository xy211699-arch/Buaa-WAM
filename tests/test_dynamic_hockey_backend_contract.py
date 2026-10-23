"""Exercise real backend guards with lightweight entities, without SAPIEN."""
from types import SimpleNamespace
import unittest

import numpy as np

from air_hockey_eval.sapien_backend import SapienBackend, _contact
from air_hockey_eval.schema import BenchmarkConfig, Geometry
from tests.test_dynamic_hockey_suite import ASSETS


class BackendContractTests(unittest.TestCase):
    def backend(self):
        backend = SapienBackend.__new__(SapienBackend)
        backend.config = BenchmarkConfig()
        backend.geometry = Geometry.from_assets(ASSETS)
        backend.limits = np.tile([-2., 2.], (6, 1))
        backend.indices = [0, 2, 3, 4, 6, 7]  # fingers need not be the final joints
        targets = [[] for _ in range(8)]
        joints = [SimpleNamespace(set_drive_target=lambda value, index=i: targets[index].append(value)) for i in range(8)]
        backend.robot = SimpleNamespace(get_active_joints=lambda: joints,
                                        get_qpos=lambda: np.zeros(8), get_qvel=lambda: np.zeros(8))
        backend.last_command = np.zeros(6)
        backend.last_command_time = -1 / backend.config.action_hz
        backend.t = 0.
        backend._violations = set()
        return backend, targets

    def test_valid_drive_targets_change_only_the_six_controlled_arm_joints(self):
        backend, targets = self.backend()
        self.assertEqual(backend.apply(np.full(6, .02)), ())
        self.assertEqual([i for i, history in enumerate(targets) if history], backend.indices)
        self.assertEqual(targets[1], [])
        self.assertEqual(targets[5], [])

    def test_nonfinite_out_of_limit_or_excessive_speed_never_reaches_a_drive(self):
        for command, reason in ((np.full(6, np.nan), "invalid_command"),
                                (np.full(6, 3.), "command_joint_position"),
                                (np.full(6, .5), "command_joint_velocity")):
            backend, targets = self.backend()
            with self.subTest(reason=reason):
                self.assertIn(reason, backend.apply(command))
                self.assertFalse(any(targets))

    def test_contact_requires_both_correct_entities_and_a_nonzero_impulse(self):
        def event(a, b, impulse):
            return SimpleNamespace(bodies=[SimpleNamespace(entity=SimpleNamespace(name=n)) for n in (a, b)],
                                   points=[SimpleNamespace(impulse=np.array(impulse))])
        puck = SimpleNamespace(get_name=lambda: "puck")
        striker = SimpleNamespace(get_name=lambda: "striker")
        for contacts, expected in (([event("puck", "striker", [0, 0, 0])], False),
                                   ([event("puck", "rail", [1, 0, 0])], False),
                                   ([event("striker", "puck", [.01, 0, 0])], True)):
            scene = SimpleNamespace(get_contacts=lambda: contacts)
            self.assertEqual(_contact(scene, puck, striker), expected)

    def test_robot_table_impulse_invalidates_an_otherwise_valid_frame(self):
        backend, _ = self.backend()
        backend._robot_entities = {"elbow"}
        backend.body = SimpleNamespace(linear_velocity=np.array([1., 0, 0]))
        backend.puck = SimpleNamespace(pose=SimpleNamespace(p=np.array([0., 0., .75])), get_name=lambda: "puck")
        backend.assembly = {side + "_striker": SimpleNamespace(
            pose=SimpleNamespace(p=np.array([x, 0, .7405])), get_name=lambda side=side: side + "_striker")
            for side, x in (("left", -.85), ("right", .85))}
        collision = SimpleNamespace(bodies=[SimpleNamespace(entity=SimpleNamespace(name=n))
                                             for n in ("air_hockey_table", "elbow")],
                                    points=[SimpleNamespace(impulse=np.array([.01, 0, 0]))])
        backend.scene = SimpleNamespace(get_contacts=lambda: [collision])
        frame = backend.frame()
        self.assertIn("robot_table_collision", frame.violations)
        self.assertFalse(frame.contact)
        self.assertFalse(frame.out_of_bounds)


if __name__ == "__main__":
    unittest.main()
