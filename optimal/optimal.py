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

# 4. Track processing and racing-line generation

# 5. Optimization

# 6. Visualization and printed output

# 7. Main execution
