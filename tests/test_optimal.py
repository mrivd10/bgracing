"""Deterministic model invariants; no claim of experimental vehicle validation."""
import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from optimal.optimal import (
    VehicleParameters, MotorParameters, SolverSettings, Track,
    static_vehicle_properties, aerodynamic_forces, normal_loads,
    tire_force_limits, friction_utilization, motor_torque, slip_angles,
    allocate_longitudinal_force, evaluate_vehicle_state, validate_configuration,
    load_track, generate_trajectory, calculate_speed_profile, validate_speed_profile,
    optimize_racing_line, has_self_intersection,
)


class PhysicsTests(unittest.TestCase):
    def test_static_equilibrium(self):
        for cg_y in (0., .08, -.08):
            with self.subTest(cg_y=cg_y):
                p = replace(VehicleParameters(), cg_x=-.65, cg_y=cg_y)
                static = static_vehicle_properties(p)
                loads = static["corner_loads"]
                self.assertAlmostEqual(loads.sum(), p.mass*p.gravity)
                np.testing.assert_allclose(
                    loads @ static["wheel_positions"], [0., 0.], atol=1e-10)
                self.assertAlmostEqual(static["axle_loads"][0], p.mass*p.gravity*(p.wheelbase+p.cg_x)/p.wheelbase)
                self.assertTrue(np.all(loads > 0))

    # More physics tests
