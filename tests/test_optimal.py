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

    def test_allocation_and_battery_limits(self):
        p = replace(VehicleParameters(), battery_power=1000., battery_regen_power=300.)
        speed = np.full((1, 4), 20.)
        capacity = np.full((1, 4, 2), 1e5)
        lateral = np.zeros((1, 4))
        drive = allocate_longitudinal_force(np.array([10000.]), speed, lateral, capacity, p)
        self.assertLessEqual(drive["battery_power"][0], p.battery_power+1e-8)
        self.assertAlmostEqual(drive["yaw_moment"][0], 0.)
        np.testing.assert_allclose(drive["actual"][0, [0, 2]], drive["actual"][0, [1, 3]])
        brake = allocate_longitudinal_force(np.array([-20000.]), speed, lateral, capacity, p)
        self.assertLessEqual(brake["regen_power"][0], p.battery_regen_power+1e-8)
        self.assertGreater(np.abs(brake["actual"]).sum(), 0.)
        stalled = allocate_longitudinal_force(np.array([-200.]), np.zeros((1, 4)), lateral, capacity, p)
        self.assertEqual(stalled["regen_power"][0], 0.)
        weak = replace(p, battery_power=40000., front_left=replace(p.front_left, maximum_torque=10.))
        limited = allocate_longitudinal_force(np.array([10000.]), speed, lateral, capacity, weak)
        self.assertLessEqual(limited["actual"][0, 0], 10./p.rolling_radius+1e-8)

    def test_invalid_configuration_and_slip(self):
        p = VehicleParameters()
        validate_configuration(p, SolverSettings())
        for invalid in (replace(p, mass=-1.), replace(p, cg_x=.2),
                        replace(p, tire_load_sensitivity=1.),
                        replace(p, front_left=replace(p.front_left, efficiency=0.))):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_configuration(invalid, SolverSettings())
        loads = normal_loads(np.array([10.]), 0., 0., p)
        steering, slip, beta = slip_angles(np.array([10.]), np.array([0.]), np.zeros((1, 4)), loads, p)
        np.testing.assert_allclose(steering, 0.)
        np.testing.assert_allclose(slip, 0.)
        np.testing.assert_allclose(beta, 0.)
        for values in ((np.nan, 0., 0.), (10., np.inf, 0.), (-1., 0., 0.)):
            with self.assertRaises(ValueError):
                evaluate_vehicle_state(*values, p)

    def test_closed_speed_profile_and_candidate_geometry(self):
        p = VehicleParameters()
        settings = replace(SolverSettings(), longitudinal_samples=48, geometry_samples=96,
                           curvature_resolution=240, interpolation_resolution=240, control_points=8)
        theta = np.linspace(0., 2*np.pi, 240, endpoint=False)
        radial = np.column_stack((np.cos(theta), np.sin(theta)))
        track = Track(36*radial, 44*radial, 40*radial, theta/(2*np.pi), True, 80*np.pi, ())
        line = generate_trajectory(track, np.zeros(8), p, settings)
        np.testing.assert_allclose(line.spline(0.), line.spline(1.), atol=1e-10)
        np.testing.assert_allclose(line.spline(0., 1), line.spline(1., 1), atol=1e-8)
        self.assertGreater(line.boundary_clearance, 0.)
        np.testing.assert_allclose(line.curvature, 1/40, atol=5e-5)
        shifted = generate_trajectory(track, np.full(8, .3), p, settings)
        self.assertGreater(shifted.length, line.length)
        self.assertFalse(np.allclose(shifted.curvature, line.curvature))
        result = calculate_speed_profile(line, p, settings)
        self.assertTrue(result.feasible, result.message)
        self.assertTrue(np.all(result.state.feasible))
        self.assertLessEqual(result.state.utilization.max(), 1.+1e-8)
        self.assertAlmostEqual(result.acceleration[-1], (result.speed[0]**2-result.speed[-1]**2)/(2*line.segment_length[-1]))
        exaggerated = validate_speed_profile(line, np.full(48, 45.), p, settings)[3]
        self.assertFalse(np.all(exaggerated.feasible))

    def test_open_track_speed_profile(self):
        p = VehicleParameters()
        settings = replace(SolverSettings(), longitudinal_samples=40, geometry_samples=40,
                           curvature_resolution=200, start_speed=0., finish_speed=0.)
        x = np.linspace(0., 200., 200)
        center = np.column_stack((x, np.zeros_like(x)))
        track = Track(center+[0., 4.], center-[0., 4.], center,
                      np.linspace(0., 1., len(x)), False, 200., ())
        line = generate_trajectory(track, np.zeros(8), p, settings)
        result = calculate_speed_profile(line, p, settings)
        self.assertTrue(result.feasible, result.message)
        self.assertEqual(result.speed[0], 0.)
        self.assertEqual(result.speed[-1], 0.)
        self.assertGreater(result.speed.max(), 10.)
        self.assertTrue(np.isfinite(result.lap_time))
        self.assertEqual(len(line.segment_length), len(line.distance)-1)

    # More physics tests
