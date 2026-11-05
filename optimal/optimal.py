"""Four-motor quasi-static lap-time estimation, using ASSUMED vehicle data.

SI units. X forward, Y left, Z up; configuration origin = front axle midpoint.
A CG behind the front axle therefore has negative cg_x. Wheel order: FL FR RL RR.
Positive curvature and yaw turn left. Flat-road model, not experimental validation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial import cKDTree

Array = NDArray[np.float64]
WHEELS = ("FL", "FR", "RL", "RR")
EPS = 1e-10  # Numerical zero, not physical tuning.

# 1. Vehicle parameters and physical constants
@dataclass(frozen=True)
class MotorParameters:
    """ASSUMED shaft data for one direct-drive in-wheel motor."""
    maximum_torque: float = 140.0  # ASSUMED N m peak motoring torque.
    maximum_power: float = 12000.0  # ASSUMED W mechanical shaft power.
    maximum_speed: float = 240.0  # ASSUMED rad/s rotor limit.
    efficiency: float = 0.92  # ASSUMED electrical/mechanical efficiency.
    regenerative_torque: float = 60.0  # ASSUMED N m peak regeneration.
    # ASSUMED (rad/s, N m) curve; interpolated and capped by torque/power.
    torque_speed_curve: tuple[tuple[float, float], ...] = (
        (0., 140.), (60., 140.), (90., 133.333333),
        (120., 100.), (180., 66.666667), (240., 50.),
    )


@dataclass(frozen=True)
class VehicleParameters:
    """Central editable ASSUMED configuration, separate from simulation state."""
    mass: float = 300.0  # ASSUMED kg including driver.
    cg_x: float = -0.80  # ASSUMED m behind front axle in X-forward frame.
    cg_y: float = 0.0  # ASSUMED m left of vehicle center plane.
    cg_z: float = 0.28  # ASSUMED m above flat ground.
    wheelbase: float = 1.60  # ASSUMED m.
    front_track: float = 1.25  # ASSUMED m front contact spacing.
    rear_track: float = 1.20  # ASSUMED m rear contact spacing.
    external_width: float = 1.50  # ASSUMED m body width.
    external_length: float = 2.80  # ASSUMED m conservative footprint length.
    yaw_inertia: float = 120.0  # ASSUMED kg m^2 about CG.
    rolling_radius: float = 0.23  # ASSUMED m effective tire radius.
    mu_longitudinal: float = 1.70  # ASSUMED at reference load.
    mu_lateral: float = 1.80  # ASSUMED at reference load.
    cornering_stiffness: float = 15000.0  # ASSUMED N/rad per reference tire.
    tire_reference_load: float = 750.0  # ASSUMED N calibration load.
    tire_load_sensitivity: float = 0.12  # ASSUMED capacity exponent = 1-s.
    friction_exponent: float = 2.0  # ASSUMED 2 = friction ellipse.
    small_slip_limit: float = 0.12  # ASSUMED rad linear model validity.
    front_left: MotorParameters = field(default_factory=MotorParameters)
    front_right: MotorParameters = field(default_factory=MotorParameters)
    rear_left: MotorParameters = field(default_factory=MotorParameters)
    rear_right: MotorParameters = field(default_factory=MotorParameters)
    battery_power: float = 40000.0  # ASSUMED W electrical discharge cap.
    battery_regen_power: float = 20000.0  # ASSUMED W charge acceptance cap.
    regen_minimum_speed: float = 1.0  # ASSUMED m/s regen cutoff.
    mechanical_brake_torque: float = 450.0  # ASSUMED N m per wheel cap.
    front_brake_bias: float = 0.60  # ASSUMED total brake force front fraction.
    front_drive_fraction: float = 0.50  # ASSUMED fixed drive axle fraction.
    air_density: float = 1.225  # ASSUMED kg/m^3 ambient density.
    downforce_area: float = 3.0  # ASSUMED m^2 positive Cl*A downforce.
    drag_area: float = 1.0  # ASSUMED m^2 Cd*A; drag acts at CG height.
    aero_front_fraction: float = 0.45  # ASSUMED front downforce fraction.
    front_roll_fraction: float = 0.55  # ASSUMED elastic roll stiffness split.
    front_roll_center: float = 0.035  # ASSUMED m above ground.
    rear_roll_center: float = 0.045  # ASSUMED m above ground.
    wheel_lift_threshold: float = 5.0  # ASSUMED N; reject unloading.
    maximum_speed: float = 45.0  # ASSUMED m/s.
    maximum_steering: float = 0.50  # ASSUMED rad road-wheel angle.
    grip_safety_factor: float = 0.92  # ASSUMED retained tire capacity fraction.
    boundary_margin: float = 0.40  # ASSUMED m beyond body footprint.
    gravity: float = 9.80665  # STANDARD m/s^2, not measured local gravity.


@dataclass(frozen=True)
class SolverSettings:
    """Numerical choices, not measured vehicle properties."""
    longitudinal_samples: int = 180
    lateral_candidates: int = 7
    interpolation_resolution: int = 1600
    curvature_resolution: int = 720
    control_points: int = 12
    optimization_sweeps: int = 2
    speed_iterations: int = 120
    bisection_iterations: int = 18
    speed_tolerance: float = 1e-4
    feasibility_tolerance: float = 2e-5
    constraint_substeps: int = 3
    closure_gap_ratio: float = 1.8
    maximum_ordering_jump_ratio: float = 5.0
    minimum_speed: float = 0.05
    acceleration_search_limit: float = 25.0  # m/s^2 checked numerical bracket.
    start_speed: float = 0.0  # m/s OPEN paths only.
    finish_speed: float | None = None  # Free OPEN terminal speed if None.


@dataclass
class Track:
    left: Array
    right: Array
    center: Array
    parameter: Array
    closed: bool
    length: float
    notes: tuple[str, ...]


@dataclass
class Trajectory:
    position: Array
    heading: Array
    curvature: Array
    curvature_gradient: Array
    distance: Array
    segment_length: Array
    length: float
    closed: bool
    boundary_clearance: float
    spline: CubicSpline
    geometry_parameter: Array
    geometry_distance: Array


@dataclass
class VehicleState:
    normal_load: Array
    available_force: Array
    requested_force: Array
    actual_force: Array
    utilization: Array
    battery_power: Array
    regenerative_power: Array
    yaw_moment: Array
    steering: Array
    slip_angles: Array
    wheel_lift: NDArray[np.bool_]
    feasible: NDArray[np.bool_]


@dataclass
class SimulationResult:
    trajectory: Trajectory
    speed: Array
    acceleration: Array
    state: VehicleState
    lap_time: float
    feasible: bool
    iterations: int
    message: str
    validation_distance: Array
    validation_acceleration: Array


def validate_configuration(vehicle: VehicleParameters, settings: SolverSettings) -> None:
    """Reject unsupported, nonfinite, or physically invalid model inputs."""
    positive = ("mass", "cg_z", "wheelbase", "front_track", "rear_track",
                "external_width", "external_length", "yaw_inertia", "rolling_radius",
                "mu_longitudinal", "mu_lateral", "cornering_stiffness",
                "tire_reference_load", "small_slip_limit", "maximum_speed", "gravity",
                "maximum_steering", "air_density")
    for name in positive:
        if not np.isfinite(getattr(vehicle, name)) or getattr(vehicle, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in asdict(vehicle).items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not -vehicle.wheelbase < vehicle.cg_x < 0:
        raise ValueError("cg_x must lie behind the front axle and ahead of the rear")
    if abs(vehicle.cg_y) >= min(vehicle.front_track, vehicle.rear_track) / 2:
        raise ValueError("CG must lie between the wheels")
    for name in ("front_drive_fraction", "front_brake_bias", "aero_front_fraction",
                 "front_roll_fraction", "grip_safety_factor"):
        if not 0 < getattr(vehicle, name) < 1:
            raise ValueError(f"{name} must lie strictly between zero and one")
    for name in ("battery_power", "battery_regen_power", "regen_minimum_speed",
                 "mechanical_brake_torque", "downforce_area", "drag_area",
                 "wheel_lift_threshold", "boundary_margin", "front_roll_center",
                 "rear_roll_center"):
        if getattr(vehicle, name) < 0:
            raise ValueError(f"{name} cannot be negative")
    if not 0 <= vehicle.tire_load_sensitivity < 1 or vehicle.friction_exponent < 1:
        raise ValueError("Invalid tire sensitivity or friction exponent")
    if max(vehicle.front_roll_center, vehicle.rear_roll_center) > vehicle.cg_z:
        raise ValueError("This approximate roll model requires roll centers below CG")
    for motor in (vehicle.front_left, vehicle.front_right, vehicle.rear_left, vehicle.rear_right):
        curve = np.asarray(motor.torque_speed_curve, dtype=float)
        if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2
                or not np.all(np.isfinite(curve)) or np.any(curve < 0)
                or curve[0, 0] != 0 or np.any(np.diff(curve[:, 0]) <= 0)
                or curve[-1, 0] < motor.maximum_speed):
            raise ValueError("Motor curve must span zero to maximum speed, in order")
        if (not all(np.isfinite(x) and x > 0 for x in (
                motor.maximum_torque, motor.maximum_power, motor.maximum_speed))
                or not 0 < motor.efficiency <= 1 or motor.regenerative_torque < 0):
            raise ValueError("Invalid motor limits")
    for name in ("longitudinal_samples", "interpolation_resolution", "curvature_resolution"):
        if getattr(settings, name) < 8:
            raise ValueError(f"{name} must be at least eight")
    for name in ("control_points", "lateral_candidates", "bisection_iterations",
                 "speed_iterations", "constraint_substeps"):
        if getattr(settings, name) < 2:
            raise ValueError(f"{name} must be at least two")
    if settings.optimization_sweeps < 0:
        raise ValueError("optimization_sweeps cannot be negative")
    for name in ("speed_tolerance", "feasibility_tolerance", "minimum_speed",
                 "acceleration_search_limit", "closure_gap_ratio",
                 "maximum_ordering_jump_ratio"):
        if not np.isfinite(getattr(settings, name)) or getattr(settings, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if settings.start_speed < 0 or (settings.finish_speed is not None and settings.finish_speed < 0):
        raise ValueError("Open-path endpoint speeds cannot be negative")

# 2. Static calculations

def static_vehicle_properties(vehicle: VehicleParameters) -> dict[str, Array]:
    """Static equilibrium with an explicit zero-diagonal-preload assumption.

    Axle totals follow pitch equilibrium. Each axle carries the same lateral
    CG offset using its own track width; suspension preload could change the
    diagonal corner weights without changing the total equilibrium.
    """
    a = -vehicle.cg_x
    b = vehicle.wheelbase - a
    axle = vehicle.mass * vehicle.gravity * np.array([b, a]) / vehicle.wheelbase
    left_fraction = 0.5 + vehicle.cg_y / np.array([vehicle.front_track, vehicle.rear_track])
    corners = (axle[:, None] * np.column_stack((left_fraction, 1-left_fraction))).ravel()
    positions = np.array([[a, vehicle.front_track/2-vehicle.cg_y],
                          [a, -vehicle.front_track/2-vehicle.cg_y],
                          [-b, vehicle.rear_track/2-vehicle.cg_y],
                          [-b, -vehicle.rear_track/2-vehicle.cg_y]])
    return {"axle_loads": axle, "corner_loads": corners,
            "axle_distances": np.array([a, b]), "wheel_positions": positions}

# 3. Dynamic calculations

def aerodynamic_forces(speed: Array | float, vehicle: VehicleParameters) -> tuple[Array, Array]:
    """Return per-wheel downforce and drag (N); positive ClA means downforce."""
    speed = np.asarray(speed, dtype=float)
    if np.any(~np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("Speed must be finite and nonnegative")
    pressure = 0.5 * vehicle.air_density * speed**2
    split = np.array([vehicle.aero_front_fraction/2]*2
                     + [(1-vehicle.aero_front_fraction)/2]*2)
    return pressure[..., None] * vehicle.downforce_area * split, pressure * vehicle.drag_area

def normal_loads(speed: Array | float, longitudinal_acceleration: Array | float,
                 lateral_acceleration: Array | float, vehicle: VehicleParameters) -> Array:
    """Estimate Fz without clipping negative loads (callers must reject them).

    Positive acceleration moves load rearward; a left turn loads right tires.
    Geometric roll moments plus the elastic remainder sum to m*ay*h exactly.
    Each axle transfer is SUBTRACTED from one wheel and ADDED to the other.
    Drag acts at CG height; downforce balance is specified at the ground axles.
    """
    speed, ax, ay = np.broadcast_arrays(speed, longitudinal_acceleration, lateral_acceleration)
    if not np.all(np.isfinite(ax)) or not np.all(np.isfinite(ay)):
        raise ValueError("Accelerations must be finite")
    static = static_vehicle_properties(vehicle)
    aero, drag = aerodynamic_forces(speed, vehicle)
    loads = static["corner_loads"] + aero
    pitch = (vehicle.mass * ax + drag) * vehicle.cg_z / vehicle.wheelbase
    loads = loads + pitch[..., None] * np.array([-0.5, -0.5, 0.5, 0.5])
    front_fraction = static["axle_loads"][0] / (vehicle.mass * vehicle.gravity)
    geometric_front = vehicle.mass * ay * front_fraction * vehicle.front_roll_center
    geometric_rear = vehicle.mass * ay * (1-front_fraction) * vehicle.rear_roll_center
    elastic = vehicle.mass * ay * vehicle.cg_z - geometric_front - geometric_rear
    front_transfer = (geometric_front + vehicle.front_roll_fraction*elastic) / vehicle.front_track
    rear_transfer = (geometric_rear + (1-vehicle.front_roll_fraction)*elastic) / vehicle.rear_track
    return loads + np.stack((-front_transfer, front_transfer, -rear_transfer, rear_transfer), axis=-1)

def tire_force_limits(loads: Array, vehicle: VehicleParameters) -> Array:
    """Return (..., wheel, [Fx_max, Fy_max]) in N with sublinear load sensitivity.

    Negative/unloaded tires have zero capacity solely to keep diagnostics finite;
    the state evaluator separately rejects negative loads and wheel unloading.
    """
    loads = np.asarray(loads, dtype=float)
    if not np.all(np.isfinite(loads)):
        raise ValueError("Normal loads must be finite")
    effective = vehicle.tire_reference_load * (
        np.maximum(loads, 0) / vehicle.tire_reference_load)**(1-vehicle.tire_load_sensitivity)
    return (effective[..., None] * vehicle.grip_safety_factor
            * np.array([vehicle.mu_longitudinal, vehicle.mu_lateral]))

def friction_utilization(force: Array, capacity: Array, exponent: float = 2.0) -> Array:
    """Return the ellipse norm, <=1 feasible; never silently clip requested force."""
    force, capacity = np.broadcast_arrays(np.asarray(force, float), np.asarray(capacity, float))
    if force.shape[-1] != 2 or not np.all(np.isfinite(force)) or not np.all(np.isfinite(capacity)):
        raise ValueError("Forces/capacities must be finite Fx,Fy pairs")
    ratio = np.divide(np.abs(force), capacity, out=np.zeros_like(force), where=capacity > EPS)
    ratio = np.where((capacity <= EPS) & (np.abs(force) > EPS), np.inf, ratio)
    return np.sum(ratio**exponent, axis=-1)**(1/exponent)

def motor_torque(speed: Array | float, motor: MotorParameters, regenerative: bool = False) -> Array:
    """Torque at rotor speed (rad/s), capped by curve, shaft power, and rotor limit."""
    omega = np.asarray(speed, dtype=float)
    if not np.all(np.isfinite(omega)) or np.any(omega < 0):
        raise ValueError("Rotor speed must be finite and nonnegative")
    curve = np.asarray(motor.torque_speed_curve)
    curve_limit = np.interp(omega, curve[:, 0], curve[:, 1])
    torque_limit = motor.regenerative_torque if regenerative else motor.maximum_torque
    torque = np.minimum(curve_limit, torque_limit)
    torque = np.minimum(torque, motor.maximum_power / np.maximum(omega, EPS))
    return np.where(omega <= motor.maximum_speed, torque, 0.0)

def slip_angles(speed: Array, curvature: Array, lateral_force: Array, loads: Array,
                vehicle: VehicleParameters) -> tuple[Array, Array, Array]:
    """Return steering, tire slip, and vehicle sideslip (rad), approximately.

    Tire slip here = wheel heading minus contact velocity heading (positive Fy).
    Vehicle sideslip beta = CG velocity heading minus body heading.
    Steering = wheel heading minus body heading. Linear Fy=C_alpha*alpha is
    only used below small_slip_limit; the state evaluator rejects larger slips.
    Ackermann steering and rigid-body wheel velocities supply individual angles.
    """
    a, b = static_vehicle_properties(vehicle)["axle_distances"]
    stiffness = vehicle.cornering_stiffness * (
        np.maximum(loads, EPS)/vehicle.tire_reference_load)**(1-vehicle.tire_load_sensitivity)
    axle_force = lateral_force.reshape((*lateral_force.shape[:-1], 2, 2)).sum(axis=-1)
    axle_stiffness = stiffness.reshape((*stiffness.shape[:-1], 2, 2)).sum(axis=-1)
    alpha = axle_force / np.maximum(axle_stiffness, EPS)
    beta = np.arctan(b*curvature) - alpha[..., 1]
    central_steering = np.arctan(np.tan(beta) + a*curvature) + alpha[..., 0]
    tangent = np.tan(central_steering)
    steering = np.zeros_like(lateral_force)
    steering[..., :2] = np.arctan2(
        tangent[..., None],
        1 - tangent[..., None]*np.array([vehicle.front_track/2, -vehicle.front_track/2])/vehicle.wheelbase)
    positions = static_vehicle_properties(vehicle)["wheel_positions"]
    yaw_rate = speed * curvature
    # Use unit speed at standstill to obtain the limiting geometric angles.
    u = np.where(speed > EPS, speed, 1.0)
    rate = np.where(speed > EPS, yaw_rate, curvature)
    vx = u[..., None]*np.cos(beta[..., None]) - rate[..., None]*positions[:, 1]
    vy = u[..., None]*np.sin(beta[..., None]) + rate[..., None]*positions[:, 0]
    tire_slip = steering - np.arctan2(vy, vx)
    return steering, tire_slip, beta

def allocate_longitudinal_force(request: Array, wheel_speed: Array, lateral_force: Array,
                                capacity: Array, vehicle: VehicleParameters) -> dict[str, Array]:
    """Fixed axle split, equal L/R forces; conservative extension point for QP allocation.

    Torque-vectoring yaw = sum(-y_i*Fx_i), using each axle's actual track width.
    Equal L/R commands avoid a spurious yaw demand for a centered CG; no claim
    of optimal allocation. Battery limits apply to ELECTRICAL power.
    """
    motors = (vehicle.front_left, vehicle.front_right, vehicle.rear_left, vehicle.rear_right)
    omega = np.maximum(wheel_speed, 0) / vehicle.rolling_radius
    drive = np.stack([motor_torque(omega[..., i], m) for i, m in enumerate(motors)], axis=-1)
    regen = np.stack([motor_torque(omega[..., i], m, True) for i, m in enumerate(motors)], axis=-1)
    drive /= vehicle.rolling_radius
    regen /= vehicle.rolling_radius
    efficiency = np.array([m.efficiency for m in motors])
    regen = np.where(wheel_speed >= vehicle.regen_minimum_speed, regen, 0.0)
    regen_power = np.sum(regen*wheel_speed*efficiency, axis=-1)
    regen *= np.minimum(1.0, vehicle.battery_regen_power/np.maximum(regen_power, EPS))[..., None]
    lateral_ratio = np.abs(lateral_force) / np.maximum(capacity[..., 1], EPS)
    remaining = capacity[..., 0] * np.maximum(
        0., 1-lateral_ratio**vehicle.friction_exponent)**(1/vehicle.friction_exponent)
    braking = request < 0
    fraction = np.where(braking, vehicle.front_brake_bias, vehicle.front_drive_fraction)
    weights = np.stack((fraction/2, fraction/2, (1-fraction)/2, (1-fraction)/2), axis=-1)
    brake_capacity = vehicle.mechanical_brake_torque/vehicle.rolling_radius + regen
    limits = np.minimum(remaining, np.where(braking[..., None], brake_capacity, drive))
    requested = request[..., None]*weights
    maximum_total = np.min(limits / weights, axis=-1)
    electrical_per_force = np.sum(weights*wheel_speed/efficiency, axis=-1)
    maximum_total = np.where(braking, maximum_total, np.minimum(
        maximum_total, vehicle.battery_power/np.maximum(electrical_per_force, EPS)))
    actual = np.sign(request)[..., None]*np.minimum(np.abs(request), maximum_total)[..., None]*weights
    used_regen = np.minimum(np.maximum(-actual, 0), regen)
    power = np.sum(np.maximum(actual, 0)*wheel_speed/efficiency, axis=-1)
    charge = np.sum(used_regen*wheel_speed*efficiency, axis=-1)
    y = static_vehicle_properties(vehicle)["wheel_positions"][:, 1]
    yaw = -np.sum(y*actual, axis=-1)
    return {"requested": requested, "actual": actual, "battery_power": power,
            "regen_power": charge, "yaw_moment": yaw, "limits": limits}

def evaluate_vehicle_state(speed: Array | float, curvature: Array | float,
                           acceleration: Array | float, vehicle: VehicleParameters,
                           curvature_gradient: Array | float = 0.0) -> VehicleState:
    """Solve planar force/yaw equilibrium, then check every requested tire force.

    Unknowns are total longitudinal tire force and front/rear lateral forces.
    Steering rotates tire forces into body axes. Yaw acceleration is
    ax*kappa + v^2*d(kappa)/ds. Steering/slip and force equilibrium are iterated;
    nonconvergence is rejected. This is quasi-steady slip, not transient tires.
    """
    v, k, ax, dk = np.broadcast_arrays(
        np.asarray(speed, float), np.asarray(curvature, float),
        np.asarray(acceleration, float), np.asarray(curvature_gradient, float))
    if not all(np.all(np.isfinite(x)) for x in (v, k, ax, dk)) or np.any(v < 0):
        raise ValueError("Vehicle state inputs must be finite, with nonnegative speed")
    loads = normal_loads(v, ax, v*v*k, vehicle)
    capacity = tire_force_limits(loads, vehicle)
    positions = static_vehicle_properties(vehicle)["wheel_positions"]
    _, drag = aerodynamic_forces(v, vehicle)
    demand = np.stack((vehicle.mass*ax+drag, vehicle.mass*v*v*k,
                       vehicle.yaw_inertia*(ax*k+v*v*dk)), axis=-1)
    lateral_weights = np.maximum(capacity[..., 1], EPS)
    lateral_weights = lateral_weights.reshape((*v.shape, 2, 2))
    lateral_weights /= lateral_weights.sum(axis=-1, keepdims=True)
    lateral_weights = lateral_weights.reshape((*v.shape, 4))
    steering = np.zeros((*v.shape, 4))
    lateral_force = np.zeros_like(steering)
    tire_slip = np.zeros_like(steering)
    force = demand[..., 0].copy()
    # Fixed-point iteration with a conservative small-slip validity check below.
    for _ in range(12):
        fraction = np.where(force < 0, vehicle.front_brake_bias, vehicle.front_drive_fraction)
        weights = np.stack((fraction/2, fraction/2, (1-fraction)/2, (1-fraction)/2), axis=-1)
        cosine, sine = np.cos(steering), np.sin(steering)
        front_basis = lateral_weights*np.array([1, 1, 0, 0])
        rear_basis = lateral_weights*np.array([0, 0, 1, 1])
        long_column = np.stack((np.sum(weights*cosine, axis=-1),
                                np.sum(weights*sine, axis=-1),
                                np.sum(weights*(positions[:, 0]*sine-positions[:, 1]*cosine), axis=-1)), axis=-1)
        front_column = np.stack((-np.sum(front_basis*sine, axis=-1),
                                 np.sum(front_basis*cosine, axis=-1),
                                 np.sum(front_basis*(positions[:, 0]*cosine+positions[:, 1]*sine), axis=-1)), axis=-1)
        rear_column = np.stack((-np.sum(rear_basis*sine, axis=-1),
                                np.sum(rear_basis*cosine, axis=-1),
                                np.sum(rear_basis*(positions[:, 0]*cosine+positions[:, 1]*sine), axis=-1)), axis=-1)
        matrix = np.stack((long_column, front_column, rear_column), axis=-1)
        solution = np.linalg.solve(matrix, demand[..., None])[..., 0]
        force = solution[..., 0]
        lateral_force = solution[..., 1, None]*front_basis + solution[..., 2, None]*rear_basis
        new_steering, tire_slip, _ = slip_angles(v, k, lateral_force, loads, vehicle)
        error = np.max(np.abs(new_steering-steering), axis=-1)
        steering = new_steering
        if np.all(error < 1e-8):
            break
    wheel_speed = np.maximum(v[..., None]*(1-k[..., None]*positions[:, 1]), 0)
    allocation = allocate_longitudinal_force(force, wheel_speed, lateral_force, capacity, vehicle)
    requested = np.stack((allocation["requested"], lateral_force), axis=-1)
    actual = np.stack((allocation["actual"], lateral_force), axis=-1)
    utilization = friction_utilization(requested, capacity, vehicle.friction_exponent)
    lift = loads <= vehicle.wheel_lift_threshold
    motor_maximum = np.array([m.maximum_speed for m in (
        vehicle.front_left, vehicle.front_right, vehicle.rear_left, vehicle.rear_right)])
    valid = ((v <= vehicle.maximum_speed) & np.all(~lift, axis=-1)
             & np.all(utilization <= 1+1e-8, axis=-1)
             & np.all(np.abs(allocation["requested"]-allocation["actual"]) <= 1e-5, axis=-1)
             & np.all(wheel_speed/vehicle.rolling_radius <= motor_maximum, axis=-1)
             & np.all(np.abs(steering) <= vehicle.maximum_steering, axis=-1)
             & np.all(np.abs(tire_slip) <= vehicle.small_slip_limit, axis=-1)
             & (error < 1e-6))
    yaw = np.sum(allocation["actual"]*(positions[:, 0]*np.sin(steering)
                                     -positions[:, 1]*np.cos(steering)), axis=-1)
    return VehicleState(loads, capacity, requested, actual, utilization,
                        allocation["battery_power"], allocation["regen_power"],
                        yaw, steering, tire_slip, lift, valid)

def longitudinal_limits(speed: Array, curvature: Array, gradient: Array,
                        vehicle: VehicleParameters, settings: SolverSettings) -> tuple[Array, Array]:
    """Find connected feasible acceleration/braking intervals about steady speed.

    A finite bracket is a conservative numerical cap, not an asserted physical
    maximum. The full force/load/motor model is re-evaluated during bisection.
    """
    speed, curvature, gradient = np.broadcast_arrays(speed, curvature, gradient)
    steady = evaluate_vehicle_state(speed, curvature, 0., vehicle, gradient).feasible
    limits = []
    for sign in (1., -1.):
        low = np.zeros_like(speed)
        high = np.full_like(speed, settings.acceleration_search_limit)
        for _ in range(settings.bisection_iterations):
            mid = (low+high)/2
            feasible = evaluate_vehicle_state(speed, curvature, sign*mid, vehicle, gradient).feasible & steady
            low = np.where(feasible, mid, low)
            high = np.where(feasible, high, mid)
        limits.append(sign*low)
    return limits[0], limits[1]

# 4. Track processing and racing-line generation

def interpolate_boundary(points: Array, resolution: int, closed: bool) -> Array:
    """Shape-preserving PCHIP in arc length; cyclic padding matches the seam."""
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.all(np.isfinite(points)):
        raise ValueError("A boundary needs at least four finite XY points")
    if np.linalg.norm(points[-1]-points[0]) < EPS:
        points = points[:-1]
    if closed:
        padded = np.vstack((points[-3:], points, points[:3]))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(padded, axis=0), axis=1))]
        start, finish = arc[3], arc[len(points)+3]
    else:
        padded = points
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        start, finish = arc[0], arc[-1]
    if np.any(np.diff(arc) <= EPS):
        raise ValueError("Duplicate consecutive boundary cones")
    return PchipInterpolator(arc, padded, axis=0)(
        np.linspace(start, finish, resolution, endpoint=not closed))

def has_self_intersection(points: Array, closed: bool) -> bool:
    """Detect nonadjacent polyline crossings, including collinear overlaps."""
    start = points if closed else points[:-1]
    end = np.roll(points, -1, axis=0) if closed else points[1:]
    delta = end-start
    for i in range(len(start)):
        j = np.arange(i+2, len(start))
        if closed and i == 0:
            j = j[j != len(start)-1]
        if not len(j):
            continue
        q = start[j]-start[i]
        cross = delta[i, 0]*delta[j, 1]-delta[i, 1]*delta[j, 0]
        t = np.divide(q[:, 0]*delta[j, 1]-q[:, 1]*delta[j, 0], cross,
                      out=np.full(len(j), np.inf), where=np.abs(cross) > EPS)
        u = np.divide(q[:, 0]*delta[i, 1]-q[:, 1]*delta[i, 0], cross,
                      out=np.full(len(j), np.inf), where=np.abs(cross) > EPS)
        if np.any((t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            return True
        collinear = (np.abs(cross) <= EPS) & (np.abs(q[:, 0]*delta[i, 1]-q[:, 1]*delta[i, 0]) <= EPS)
        projection = q @ delta[i] / max(float(delta[i] @ delta[i]), EPS)
        projection_end = (end[j]-start[i]) @ delta[i] / max(float(delta[i] @ delta[i]), EPS)
        if np.any(collinear & (np.maximum(projection, projection_end) >= 0)
                  & (np.minimum(projection, projection_end) <= 1)):
            return True
    return False

def load_track(path: str | Path, settings: SolverSettings,
               topology: str = "auto") -> Track:
    """Read untouched CSV, validate ordering, and match boundaries transversely.

    Auto closure is a documented spacing/tangent heuristic; explicit open/closed
    overrides are available. Equal row counts never imply cone correspondence.
    Left-boundary normals are intersected with the right polyline, followed by
    a monotonicity check on matched right-boundary stations.
    """
    if topology not in ("auto", "open", "closed"):
        raise ValueError("topology must be auto, open or closed")
    data = pd.read_csv(path)
    if not {"x", "y", "side"}.issubset(data.columns):
        raise ValueError("CSV requires x,y,side columns")
    empty = data[["x", "y"]].isna().all(axis=1)
    notes = [f"Ignored {int(empty.sum())} empty coordinate separator rows; source unchanged."]
    data = data.loc[~empty].copy()
    data["side"] = data["side"].astype(str).str.strip().str.lower()
    if not data["side"].isin(("left", "right")).all():
        raise ValueError("Unknown boundary side")
    boundaries = []
    closure = []
    for side in ("left", "right"):
        points = data.loc[data.side == side, ["x", "y"]].to_numpy(dtype=float)
        if len(points) < 4 or not np.all(np.isfinite(points)):
            raise ValueError(f"Invalid {side} boundary coordinates")
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(steps <= EPS) or steps.max() > settings.maximum_ordering_jump_ratio*np.median(steps):
            raise ValueError(f"Duplicate or inconsistent ordering in {side} cones")
        gap = np.linalg.norm(points[-1]-points[0])
        if gap <= EPS:
            closure.append(True)
            points = points[:-1]
        else:
            seam = (points[0]-points[-1])/gap
            alignment = min(np.dot(seam, (points[1]-points[0])/steps[0]),
                            np.dot(seam, (points[-1]-points[-2])/steps[-1]))
            closure.append(gap <= settings.closure_gap_ratio*np.median(steps) and alignment > 0.5)
        boundaries.append(points)
    if topology == "auto" and closure[0] != closure[1]:
        raise ValueError("Ambiguous topology: specify --topology open or closed")
    closed = all(closure) if topology == "auto" else topology == "closed"
    notes.append(f"Topology: {'closed' if closed else 'open'} ({topology}; closure evidence {closure}).")
    left_raw, right_raw = boundaries
    # Orient the right boundary to the left without assuming matching row indices.
    left_tangent = np.roll(left_raw, -1, axis=0)-left_raw if closed else np.gradient(left_raw, axis=0)
    right_tangent = np.roll(right_raw, -1, axis=0)-right_raw if closed else np.gradient(right_raw, axis=0)
    _, nearest = cKDTree(right_raw).query(left_raw)
    if np.median(np.sum(left_tangent*right_tangent[nearest], axis=1)) < 0:
        right_raw = right_raw[::-1]
        notes.append("Reversed right boundary in memory to match traversal.")
    for points in (left_raw, right_raw):
        if has_self_intersection(points, closed):
            raise ValueError("Boundary self-intersects; repair source ordering explicitly")
    paired = validated_cone_pairs(left_raw, right_raw, closed, settings.interpolation_resolution)
    if paired is not None:
        left, right = paired
        center = (left+right)/2
        if has_self_intersection(center, closed):
            raise ValueError("Paired centerline self-intersects")
        length = float(np.linalg.norm(np.diff(center, axis=0), axis=1).sum())
        if closed:
            length += float(np.linalg.norm(center[-1]-center[0]))
        notes.append("Original row pairs verified: transverse alignment, side orientation, no connector crossings.")
        return Track(left, right, center,
                     np.linspace(0., 1., len(center), endpoint=not closed),
                     closed, length, tuple(notes))
    left = interpolate_boundary(left_raw, settings.interpolation_resolution, closed)
    right = interpolate_boundary(right_raw, settings.interpolation_resolution, closed)
    tangent = np.roll(left, -1, axis=0)-np.roll(left, 1, axis=0) if closed else np.gradient(left, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    _, nearest = cKDTree(right).query(left)
    direction = right[nearest]-left
    cross = tangent[:, 0]*direction[:, 1]-tangent[:, 1]*direction[:, 0]
    if np.median(cross) > 0:
        left, right = left[::-1], right[::-1]
        tangent = -tangent[::-1]
        notes.append("Reversed traversal in memory so left cones lie on the left.")
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    ends = np.roll(right, -1, axis=0) if closed else right[1:]
    starts = right if closed else right[:-1]
    segments = ends-starts
    matched, stations = [], []
    for point, ray in zip(left, normal):
        q = starts-point
        denominator = ray[0]*segments[:, 1]-ray[1]*segments[:, 0]
        distance = np.divide(q[:, 0]*segments[:, 1]-q[:, 1]*segments[:, 0], denominator,
                             out=np.full(len(starts), np.inf), where=np.abs(denominator) > EPS)
        fraction = np.divide(q[:, 0]*ray[1]-q[:, 1]*ray[0], denominator,
                             out=np.full(len(starts), np.inf), where=np.abs(denominator) > EPS)
        valid = (distance > 0) & (fraction >= 0) & (fraction <= 1)
        if not np.any(valid):
            if not closed and (np.linalg.norm(point-left[0]) < EPS or np.linalg.norm(point-left[-1]) < EPS):
                index = 0 if np.linalg.norm(point-left[0]) < EPS else len(right)-1
                matched.append(right[index])
                stations.append(float(index))
                continue
            raise ValueError("Cannot pair boundaries along transverse normals")
        index = int(np.argmin(np.where(valid, distance, np.inf)))
        matched.append(point+distance[index]*ray)
        stations.append(index+fraction[index])
    station = np.asarray(stations)
    if closed:
        station = np.unwrap(station*2*np.pi/len(right))*len(right)/(2*np.pi)
    if np.any(np.diff(station) < -1e-6):
        raise ValueError("Boundary correspondence reverses direction")
    right_matched = np.asarray(matched)
    center = (left+right_matched)/2
    if has_self_intersection(center, closed):
        raise ValueError("Paired centerline self-intersects")
    length = float(np.linalg.norm(np.diff(center, axis=0), axis=1).sum())
    if closed:
        length += float(np.linalg.norm(center[-1]-center[0]))
    notes.append("Boundary pairing checked by normal intersections and monotone stations.")
    return Track(left, right_matched, center,
                 np.linspace(0., 1., len(center), endpoint=not closed), closed, length, tuple(notes))

def trajectory_clearance(points: Array, track: Track, vehicle: VehicleParameters) -> float:
    """Conservative swept-disc clearance to lateral boundaries, in meters.

    A disc enclosing the ASSUMED rectangular body accounts for both width and
    length at any heading. Subtract half the largest sample spacing so a
    between-sample excursion cannot be accepted just from sampled clearance.
    Open path end caps are entry/exit gates, not physical walls.
    """
    from matplotlib.path import Path as PolygonPath

    if track.closed:
        left_polygon = PolygonPath(np.vstack((track.left, track.left[0])))
        right_polygon = PolygonPath(np.vstack((track.right, track.right[0])))
        inside = left_polygon.contains_points(points) ^ right_polygon.contains_points(points)
    else:
        polygon = np.vstack((track.left, track.right[::-1], track.left[:1]))
        inside = PolygonPath(polygon).contains_points(points, radius=1e-7)
        # Entry/exit gate centers are allowed on the cap itself.
        for endpoint in (track.center[0], track.center[-1]):
            inside |= np.linalg.norm(points-endpoint, axis=1) < 1e-7
    if not np.all(inside):
        return -np.inf
    minimum = np.inf
    for boundary in (track.left, track.right):
        # Nearest vertices identify neighboring segments on the dense polyline.
        _, nearest = cKDTree(boundary).query(points, k=min(4, len(boundary)))
        indices = np.concatenate((nearest, nearest-1), axis=1)
        if track.closed:
            indices %= len(boundary)
            following = (indices+1) % len(boundary)
        else:
            indices = np.clip(indices, 0, len(boundary)-2)
            following = indices+1
        a, b = boundary[indices], boundary[following]
        segment = b-a
        fraction = np.clip(np.sum((points[:, None]-a)*segment, axis=-1)
                           / np.maximum(np.sum(segment*segment, axis=-1), EPS), 0., 1.)
        distance = np.linalg.norm(points[:, None]-a-fraction[..., None]*segment, axis=-1)
        minimum = min(minimum, float(distance.min()))
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if track.closed:
        lengths = np.r_[lengths, np.linalg.norm(points[0]-points[-1])]
    body_radius = np.hypot(vehicle.external_length, vehicle.external_width)/2
    return float(minimum-body_radius-vehicle.boundary_margin-lengths.max()/2)

def generate_trajectory(track: Track, controls: Array, vehicle: VehicleParameters,
                        settings: SolverSettings) -> Trajectory:
    """Smooth lateral controls -> continuous XY spline -> own arc length/curvature.

    Periodic splines use an explicit duplicated seam knot but output arrays do
    not duplicate it. Cubic overshoot is permitted only if clearance checks
    succeed. The centerline is controls==0, not a replacement for optimization.
    """
    controls = np.asarray(controls, float)
    if controls.ndim != 1 or len(controls) < 4 or not np.all(np.isfinite(controls)):
        raise ValueError("At least four finite lateral controls are required")
    if np.any(np.abs(controls) > 1):
        raise ValueError("Lateral controls must be in [-1,1]")
    stations = np.linspace(0., 1., settings.longitudinal_samples, endpoint=not track.closed)
    control_stations = np.linspace(0., 1., len(controls), endpoint=not track.closed)
    if track.closed:
        offset = CubicSpline(np.r_[control_stations, 1.], np.r_[controls, controls[0]], bc_type="periodic")(stations)
        source_stations = np.r_[track.parameter, 1.]
        left = PchipInterpolator(source_stations, np.vstack((track.left, track.left[0])), axis=0)(stations)
        right = PchipInterpolator(source_stations, np.vstack((track.right, track.right[0])), axis=0)(stations)
    else:
        offset = CubicSpline(control_stations, controls, bc_type="natural")(stations)
        left = PchipInterpolator(track.parameter, track.left, axis=0)(stations)
        right = PchipInterpolator(track.parameter, track.right, axis=0)(stations)
    radius = np.hypot(vehicle.external_length, vehicle.external_width)/2+vehicle.boundary_margin
    half_width = np.linalg.norm(right-left, axis=1)/2
    if np.any(half_width <= radius):
        raise ValueError("Track too narrow for the assumed vehicle footprint")
    anchors = (left+right)/2 + offset[:, None]*(right-left)/2*(1-radius/half_width)[:, None]
    if track.closed:
        spline = CubicSpline(np.r_[stations, 1.], np.vstack((anchors, anchors[0])), axis=0, bc_type="periodic")
    else:
        spline = CubicSpline(stations, anchors, axis=0, bc_type="natural")
    dense_parameter = np.linspace(0., 1., settings.curvature_resolution+1)
    dense_points = spline(dense_parameter)
    dense_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense_points, axis=0), axis=1))]
    if np.any(np.diff(dense_arc) <= EPS):
        raise ValueError("Degenerate continuous trajectory")
    length = float(dense_arc[-1])
    sample_arc = np.linspace(0., length, settings.longitudinal_samples, endpoint=not track.closed)
    parameter = np.interp(sample_arc, dense_arc, dense_parameter)
    position = spline(parameter)
    first, second, third = spline(parameter, 1), spline(parameter, 2), spline(parameter, 3)
    tangent_norm = np.linalg.norm(first, axis=1)
    if np.any(tangent_norm <= EPS):
        raise ValueError("Zero trajectory tangent")
    numerator = first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]
    curvature = numerator/tangent_norm**3
    numerator_derivative = first[:, 0]*third[:, 1]-first[:, 1]*third[:, 0]
    gradient = (numerator_derivative/tangent_norm**3
                -3*numerator*np.sum(first*second, axis=1)/tangent_norm**5)/tangent_norm
    segment_length = np.diff(np.r_[sample_arc, length]) if track.closed else np.diff(sample_arc)
    dense_check = dense_points[:-1] if track.closed else dense_points
    clearance = trajectory_clearance(dense_check, track, vehicle)
    if has_self_intersection(dense_check, track.closed):
        raise ValueError("Candidate trajectory self-intersects")
    return Trajectory(position, np.unwrap(np.arctan2(first[:, 1], first[:, 0])),
                      curvature, gradient, sample_arc, segment_length, length,
                      track.closed, clearance, spline, dense_parameter, dense_arc)

def validated_cone_pairs(left: Array, right: Array, closed: bool, resolution: int) -> tuple[Array, Array] | None:
    """Use row pairs only after transverse, cell-orientation, and crossing checks.

    The 0.35 tangent fraction allows modest survey skew (<~20 degrees).
    Shared center-station interpolation retains verified correspondence.
    """
    if left.shape != right.shape:
        return None
    center = (left+right)/2
    tangent = np.roll(center, -1, axis=0)-np.roll(center, 1, axis=0) if closed else np.gradient(center, axis=0)
    width_vector = right-left
    width = np.linalg.norm(width_vector, axis=1)
    tangent_length = np.linalg.norm(tangent, axis=1)
    if np.any(width <= EPS) or np.any(tangent_length <= EPS):
        return None
    tangent /= tangent_length[:, None]
    if np.any(np.abs(np.sum(tangent*width_vector, axis=1))/width > 0.35):
        return None
    side = tangent[:, 0]*width_vector[:, 1]-tangent[:, 1]*width_vector[:, 0]
    if not (np.all(side < 0) or np.all(side > 0)):
        return None
    # Explicitly reject any two crossing transverse connectors.
    for i in range(len(left)-1):
        q = left[i+1:]-left[i]
        d = width_vector[i+1:]
        denominator = width_vector[i, 0]*d[:, 1]-width_vector[i, 1]*d[:, 0]
        a = np.divide(q[:, 0]*d[:, 1]-q[:, 1]*d[:, 0], denominator,
                      out=np.full(len(d), np.inf), where=np.abs(denominator) > EPS)
        b = np.divide(q[:, 0]*width_vector[i, 1]-q[:, 1]*width_vector[i, 0], denominator,
                      out=np.full(len(d), np.inf), where=np.abs(denominator) > EPS)
        if np.any((a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)):
            return None
    if np.all(side > 0):
        left, right, center = left[::-1], right[::-1], center[::-1]
    combined = np.column_stack((left, right))
    if closed:
        padded_center = np.vstack((center[-3:], center, center[:3]))
        padded = np.vstack((combined[-3:], combined, combined[:3]))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(padded_center, axis=0), axis=1))]
        stations = np.linspace(arc[3], arc[len(center)+3], resolution, endpoint=False)
    else:
        padded = combined
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
        stations = np.linspace(0., arc[-1], resolution)
    if np.any(np.diff(arc) <= EPS):
        return None
    result = PchipInterpolator(arc, padded, axis=0)(stations)
    return result[:, :2], result[:, 2:]

# 5. Optimization

def cornering_speed_limits(trajectory: Trajectory, vehicle: VehicleParameters,
                           settings: SolverSettings) -> Array:
    """Bisection on steady-speed feasibility, including aero, power and yaw demand."""
    low = np.zeros_like(trajectory.curvature)
    high = np.full_like(low, vehicle.maximum_speed)
    stationary = evaluate_vehicle_state(low, trajectory.curvature, 0., vehicle,
                                        trajectory.curvature_gradient).feasible
    if not np.all(stationary):
        raise ValueError("Trajectory fails steering geometry even at zero speed")
    for _ in range(settings.bisection_iterations):
        middle = (low+high)/2
        valid = evaluate_vehicle_state(middle, trajectory.curvature, 0., vehicle,
                                      trajectory.curvature_gradient).feasible
        low = np.where(valid, middle, low)
        high = np.where(valid, high, middle)
    return low

def validate_speed_profile(trajectory: Trajectory, speed: Array, vehicle: VehicleParameters,
                           settings: SolverSettings) -> tuple[Array, Array, Array, VehicleState, Array]:
    """Check segment endpoints from BOTH sides and dense spline-interior states.

    Constant tangential acceleration on each segment gives linear v^2(s).
    Both sides of a speed knot are checked because acceleration may jump there.
    Includes the closing segment and the same start/finish speed for circuits.
    """
    speed = np.asarray(speed, float)
    if speed.shape != trajectory.distance.shape or not np.all(np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("Speed profile must match trajectory and be finite/nonnegative")
    count = len(trajectory.segment_length)
    following = np.roll(speed, -1) if trajectory.closed else speed[1:]
    starting = speed[:count]
    acceleration = (following**2-starting**2)/(2*trajectory.segment_length)
    fractions = np.linspace(0., 1., settings.constraint_substeps+1)
    distance = (trajectory.distance[:count, None]
                +trajectory.segment_length[:, None]*fractions).ravel()
    segments = np.repeat(np.arange(count), len(fractions))
    dense_distance = trajectory.geometry_distance
    dense_segment = np.minimum(np.searchsorted(trajectory.distance, dense_distance, side="right")-1, count-1)
    distance = np.r_[distance, dense_distance]
    segments = np.r_[segments, dense_segment]
    order = np.argsort(distance, kind="stable")
    distance, segments = distance[order], segments[order]
    local = distance-trajectory.distance[segments]
    squared_speed = starting[segments]**2+2*acceleration[segments]*local
    check_speed = np.sqrt(np.maximum(squared_speed, 0.))
    parameter = np.interp(distance, trajectory.geometry_distance, trajectory.geometry_parameter)
    first = trajectory.spline(parameter, 1)
    second = trajectory.spline(parameter, 2)
    third = trajectory.spline(parameter, 3)
    norm = np.linalg.norm(first, axis=1)
    numerator = first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]
    curvature = numerator/np.maximum(norm**3, EPS)
    derivative = first[:, 0]*third[:, 1]-first[:, 1]*third[:, 0]
    gradient = (derivative/np.maximum(norm**3, EPS)
                -3*numerator*np.sum(first*second, axis=1)/np.maximum(norm**5, EPS))/np.maximum(norm, EPS)
    state = evaluate_vehicle_state(check_speed, curvature, acceleration[segments], vehicle, gradient)
    return acceleration, distance, acceleration[segments], state, segments

def calculate_speed_profile(trajectory: Trajectory, vehicle: VehicleParameters,
                            settings: SolverSettings) -> SimulationResult:
    """Coupled forward traction/backward braking sweeps, then dense validation.

    Speeds decrease monotonically from local ceilings. Closed sweeps wrap through
    the last-to-first segment; no artificial standing start is imposed. A small
    numerical reserve and, if needed, uniform speed reduction make the result a
    conservative feasible estimate, not a proof of the globally fastest profile.
    """
    if trajectory.boundary_clearance < 0:
        raise ValueError("Candidate violates vehicle footprint/boundary clearance")
    ceiling = cornering_speed_limits(trajectory, vehicle, settings)
    speed = ceiling*0.98  # 2% numerical grip reserve before coupled propagation.
    if not trajectory.closed:
        if settings.start_speed > speed[0] or (settings.finish_speed is not None and settings.finish_speed > speed[-1]):
            raise ValueError("Prescribed endpoint speed exceeds local physical limit")
        speed[0] = settings.start_speed
        if settings.finish_speed is not None:
            speed[-1] = settings.finish_speed
    count = len(trajectory.segment_length)
    converged = False
    iteration = 0
    for iteration in range(1, settings.speed_iterations+1):
        previous = speed.copy()
        accelerate, brake = longitudinal_limits(
            speed, trajectory.curvature, trajectory.curvature_gradient, vehicle, settings)
        for i in range(count):
            j = (i+1) % len(speed)
            reachable = np.sqrt(max(0., speed[i]**2+2*accelerate[i]*trajectory.segment_length[i]))
            speed[j] = min(speed[j], reachable)
        for i in range(count-1, -1, -1):
            j = (i+1) % len(speed)
            reachable = np.sqrt(max(0., speed[j]**2-2*brake[j]*trajectory.segment_length[i]))
            speed[i] = min(speed[i], reachable)
        if np.max(np.abs(previous-speed)) < settings.speed_tolerance:
            converged = True
            break
    # Validate actual acceleration and the candidate's own interior spline curvature.
    acceleration, distance, check_acceleration, state, segments = validate_speed_profile(
        trajectory, speed, vehicle, settings)
    repair_count = 0
    while not np.all(state.feasible) and repair_count < settings.speed_iterations:
        speed *= 0.98
        repair_count += 1
        acceleration, distance, check_acceleration, state, segments = validate_speed_profile(
            trajectory, speed, vehicle, settings)
    next_speed = np.roll(speed, -1) if trajectory.closed else speed[1:]
    denominator = speed[:count]+next_speed
    lap_time = float(np.sum(2*trajectory.segment_length/np.maximum(denominator, EPS)))
    endpoints_ok = trajectory.closed or (
        abs(speed[0]-settings.start_speed) <= settings.speed_tolerance
        and (settings.finish_speed is None or abs(speed[-1]-settings.finish_speed) <= settings.speed_tolerance))
    feasible = bool(converged and np.all(state.feasible) and np.all(denominator > settings.minimum_speed)
                    and endpoints_ok and np.isfinite(lap_time))
    message = (f"Propagation {'converged' if converged else 'did not converge'}; "
               f"{repair_count} dense-validation speed reductions; "
               f"endpoint speeds {'satisfied' if endpoints_ok else 'NOT satisfied'}.")
    return SimulationResult(trajectory, speed, acceleration, state, lap_time, feasible,
                            iteration, message, distance, check_acceleration)

def optimize_racing_line(track: Track, vehicle: VehicleParameters, settings: SolverSettings
                         ) -> tuple[SimulationResult, SimulationResult, Array, int]:
    """Optimize smooth lateral controls against fully evaluated estimated lap time.

    Deterministic bounded coordinate search is a local method. Unlike the old
    two-index DP, every trial has its own geometry and coupled speed history.
    Failed constraints give an inadmissible candidate, never a lap-time reward.
    """
    controls = np.zeros(settings.control_points)
    baseline = calculate_speed_profile(generate_trajectory(track, controls, vehicle, settings), vehicle, settings)
    if not baseline.feasible:
        raise ValueError("Centerline baseline failed physical checks: "+baseline.message)
    best = baseline
    evaluations = 1
    for sweep in range(settings.optimization_sweeps):
        changed = False
        span = 0.75/(sweep+1)
        for index in range(len(controls)):
            best_value = controls[index]
            candidates = np.unique(np.clip(
                controls[index]+np.linspace(-span, span, settings.lateral_candidates), -1., 1.))
            for value in candidates:
                if abs(value-controls[index]) < EPS:
                    continue
                trial = controls.copy()
                trial[index] = value
                evaluations += 1
                try:
                    trajectory = generate_trajectory(track, trial, vehicle, settings)
                    result = calculate_speed_profile(trajectory, vehicle, settings)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if result.feasible and result.lap_time < best.lap_time-settings.speed_tolerance:
                    best, best_value = result, float(value)
            if best_value != controls[index]:
                controls[index] = best_value
                changed = True
            print(f"Search sweep {sweep+1}, control {index+1}/{len(controls)}: "
                  f"{best.lap_time:.3f} s ({evaluations} candidates)", flush=True)
        if not changed:
            break
    return best, baseline, controls, evaluations

# 6. Visualization and printed output

def simulation_summary(result: SimulationResult, baseline: SimulationResult, track: Track,
                       vehicle: VehicleParameters, evaluations: int) -> dict[str, Any]:
    """Report model checks and assumptions; never label a failed profile feasible."""
    check_speed = np.sqrt(np.maximum(np.interp(
        result.validation_distance,
        np.r_[result.trajectory.distance, result.trajectory.length] if track.closed else result.trajectory.distance,
        np.r_[result.speed**2, result.speed[0]**2] if track.closed else result.speed**2), 0.))
    parameter = np.interp(result.validation_distance, result.trajectory.geometry_distance,
                          result.trajectory.geometry_parameter)
    first, second = result.trajectory.spline(parameter, 1), result.trajectory.spline(parameter, 2)
    curvature = (first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0])/np.maximum(np.linalg.norm(first, axis=1)**3, EPS)
    lift = result.state.wheel_lift
    events = int(np.sum(lift[0]) + np.sum(lift[1:] & ~lift[:-1]))
    return {
        "result_type": "SIMULATION ESTIMATE — ASSUMED vehicle parameters; no experimental validation",
        "topology": "closed" if track.closed else "open",
        "vehicle_mass_kg": vehicle.mass,
        "maximum_vehicle_speed_m_per_s": vehicle.maximum_speed,
        "track_centerline_length_m": track.length,
        "racing_line_length_m": result.trajectory.length,
        "estimated_lap_time_s": result.lap_time,
        "centerline_baseline_time_s": baseline.lap_time,
        "improvement_s": baseline.lap_time-result.lap_time,
        "improvement_percent": 100*(baseline.lap_time-result.lap_time)/baseline.lap_time,
        "average_speed_m_per_s": result.trajectory.length/result.lap_time,
        "peak_speed_m_per_s": float(result.speed.max()),
        "maximum_lateral_acceleration_m_per_s2": float(np.max(np.abs(check_speed**2*curvature))),
        "maximum_braking_deceleration_m_per_s2": float(max(0., -result.validation_acceleration.min())),
        "maximum_longitudinal_acceleration_m_per_s2": float(max(0., result.validation_acceleration.max())),
        "maximum_tire_utilization": float(result.state.utilization.max()),
        "minimum_normal_force_N": float(result.state.normal_load.min()),
        "wheel_lift_events": events,
        "maximum_battery_power_W": float(result.state.battery_power.max()),
        "maximum_regenerative_power_W": float(result.state.regenerative_power.max()),
        "minimum_conservative_boundary_clearance_m": result.trajectory.boundary_clearance,
        "all_model_constraints_satisfied": result.feasible,
        "baseline_model_constraints_satisfied": baseline.feasible,
        "candidate_evaluations": evaluations,
        "speed_solver": result.message,
        "track_notes": list(track.notes),
    }

# 7. Main execution
