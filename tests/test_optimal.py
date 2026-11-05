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

    def test_aero_and_load_conservation(self):
        p = VehicleParameters()
        down, drag = aerodynamic_forces(np.array([0., 10., 20.]), p)
        np.testing.assert_allclose(down[2], 4*down[1])
        self.assertAlmostEqual(drag[2], 4*drag[1])
        loads = normal_loads(np.array([0., 10., 20.]), np.array([0., 3., -4.]),
                             np.array([0., 5., -6.]), p)
        np.testing.assert_allclose(loads.sum(axis=-1), p.mass*p.gravity+down.sum(axis=-1))
        self.assertAlmostEqual(down[1, :2].sum()/down[1].sum(), p.aero_front_fraction)

    def test_load_transfer_moments(self):
        p = replace(VehicleParameters(), drag_area=0.)
        static = static_vehicle_properties(p)["corner_loads"]
        forward = normal_loads(0., 4., 0., p)
        reverse = normal_loads(0., -4., 0., p)
        self.assertAlmostEqual(static[:2].sum()-forward[:2].sum(), p.mass*4*p.cg_z/p.wheelbase)
        np.testing.assert_allclose(forward+reverse, 2*static)
        left = normal_loads(0., 0., 6., p)
        right = normal_loads(0., 0., -6., p)
        np.testing.assert_allclose(left, right[[1, 0, 3, 2]])
        self.assertGreater(left[1], left[0])
        roll_moment = (left[1]-left[0])*p.front_track/2+(left[3]-left[2])*p.rear_track/2
        self.assertAlmostEqual(roll_moment, p.mass*6*p.cg_z)
        self.assertTrue(np.any(normal_loads(0., 0., 100., p) < 0))
        self.assertFalse(bool(evaluate_vehicle_state(20., .25, 0., p).feasible))

    def test_friction_ellipse_and_load_sensitivity(self):
        p = VehicleParameters()
        capacity = tire_force_limits(np.array([500., 1000.]), p)
        self.assertLess(capacity[1, 0], 2*capacity[0, 0])
        force = np.array([[.6, .8], [.8, .8]])*capacity
        utilization = friction_utilization(force, capacity)
        self.assertAlmostEqual(utilization[0], 1.)
        self.assertGreater(utilization[1], 1.)
        self.assertTrue(np.isinf(friction_utilization(np.array([1., 0.]), np.zeros(2))))
        self.assertEqual(float(friction_utilization(np.zeros(2), np.zeros(2))), 0.)

    def test_motor_limits(self):
        motor = MotorParameters()
        omega = np.array([0., 40., 100., 200., 241.])
        torque = motor_torque(omega, motor)
        self.assertTrue(np.all(torque <= motor.maximum_torque))
        self.assertTrue(np.all(torque*omega <= motor.maximum_power+1e-8))
        self.assertLess(torque[3], torque[1])
        self.assertEqual(torque[-1], 0.)
        self.assertTrue(np.all(motor_torque(omega, motor, True) <= motor.regenerative_torque))
        with self.assertRaises(ValueError):
            motor_torque(np.nan, motor)

    # More physics tests
