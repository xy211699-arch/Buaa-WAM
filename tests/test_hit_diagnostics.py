import unittest

import numpy as np

from world_model.diagnostics import decompose_xy_error


class HitDiagnosticsTests(unittest.TestCase):
    def test_along_cross_and_equivalent_time_have_defined_signs(self):
        result = decompose_xy_error([-.03, .02], [0., 0.], [-.6, 0.])
        self.assertAlmostEqual(result['along_m'], .03)
        self.assertAlmostEqual(result['cross_m'], -.02)
        self.assertAlmostEqual(result['equivalent_time_s'], .05)
        self.assertAlmostEqual(result['total_m'], np.hypot(.03, .02))

    def test_stationary_direction_is_not_fabricated(self):
        result = decompose_xy_error([.1, 0.], [0., 0.], [0., 0.])
        self.assertIsNone(result['along_m'])
        self.assertIsNone(result['equivalent_time_s'])
        self.assertAlmostEqual(result['total_m'], .1)
        with self.assertRaises(ValueError):
            decompose_xy_error([np.nan, 0.], [0., 0.], [1., 0.])
