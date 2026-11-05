# Model, assumptions, and limits

All quantities use SI units. X is forward, Y left, Z up. The configuration
origin is the front axle midpoint, so the assumed CG coordinate is
`(-0.80, 0, 0.28) m`. Wheel-force arrays are ordered FL, FR, RL, RR.
Positive curvature, yaw, lateral acceleration, and tire lateral force turn left.

## Central assumed inputs

The executable configuration is the authoritative list, and every output JSON
includes it. None of the following defaults is measured BGRacing data.

| Group | ASSUMED defaults |
|---|---|
| Mass/geometry | 300 kg including driver; CG (-0.80, 0, 0.28) m; wheelbase 1.60 m; front/rear tracks 1.25/1.20 m; body width/length 1.50/2.80 m; yaw inertia 120 kg m² |
| Tires | Radius 0.23 m; reference load 750 N; longitudinal/lateral coefficients 1.70/1.80; per-tire cornering stiffness 15,000 N/rad; load-sensitivity exponent 0.12; ellipse exponent 2; small-slip limit 0.12 rad |
| Each of four motors | Direct drive; 140 N m peak; 12 kW shaft power; 240 rad/s limit; 0.92 efficiency; 60 N m maximum regenerative torque |
| Torque-speed samples | (0,140), (60,140), (90,133.333333), (120,100), (180,66.666667), (240,50), in (rad/s,N m); sampled from a constant-torque/constant-power envelope |
| Battery | 40 kW electrical discharge; 20 kW charge acceptance; regeneration disabled below 1 m/s |
| Brakes/allocation | 450 N m mechanical torque per wheel; front brake fraction 0.60; front driving fraction 0.50 |
| Aerodynamics | Density 1.225 kg/m³; positive downforce ClA 3.0 m²; CdA 1.0 m²; front downforce fraction 0.45; drag acts at CG height |
| Suspension | Front elastic roll-stiffness fraction 0.55; front/rear roll centers 0.035/0.045 m; no diagonal preload at static equilibrium |
| Limits/margins | Speed 45 m/s; road-wheel steering 0.50 rad; retain 0.92 of nominal tire grip; reject wheel loads at/below 5 N; boundary margin 0.40 m |

Gravity is the standard 9.80665 m/s², not a local measurement. Solver settings
are numerical choices rather than physical observations. Source CSV coordinates
are assumed to be metres, without modifying or rescaling them.

## Static equilibrium and load transfer

Let a=-cg_x be CG-to-front distance, b=L-a be CG-to-rear distance, and L be
wheelbase. Static axle loads are Wf=mg b/L and Wr=mg a/L. The assumed left
fraction on each axle is 1/2+cg_y/track. This supplies an explicit corner-weight
baseline; static equilibrium does not uniquely determine suspension preload
or all four corner weights.

Downforce and drag use Df=0.5 rho v² ClA and D=0.5 rho v² CdA. Downforce is
split by aerodynamic balance and equally left/right. Longitudinal transfer
from front to rear is (m ax+D)h/L, consistent with the assumed drag application
height. Half that axle transfer is applied to each wheel.

The total lateral roll moment is m ay h. Front/rear geometric contributions
use the static mass fractions and roll-center heights. The remaining elastic
moment is split by roll stiffness. Dividing each axle roll moment by its own
track gives the load moved from its inside wheel to its outside wheel.
Subtracting and adding that amount conserves total load without counting it twice.
Negative normal loads are retained in diagnostics and rejected; they are never
silently accepted by clipping them to zero.

These are flat-road quasi-static approximations: no pitch/roll dynamics, banking,
elevation, suspension travel, jacking model, unsprung-mass separation, or curb impacts.

## Tires, steering, and planar equilibrium

Nominal tire capacity is proportional to
Fref*(max(Fz,0)/Fref)^(1-load_sensitivity), multiplied by the relevant friction
coefficient and the grip safety factor. Thus heavily loaded tires gain less
capacity per added newton. Requested combined forces must satisfy

```text
(|Fx|/Fx_max)^p + (|Fy|/Fy_max)^p <= 1, with default p=2.
```

Available capacities, requested forces, and bounded actual forces are recorded
separately. An allocation that cannot supply the requested force is infeasible.

Bicycle slip estimates use load-scaled linear cornering stiffness. In this
implementation tire slip is wheel heading minus contact-velocity heading,
vehicle sideslip is CG velocity heading minus body heading, and steering is
wheel heading minus body heading. Ackermann geometry and rigid-body wheel
velocities give individual-wheel slip estimates. States outside the assumed
small-slip validity range are rejected. This does not predict nonlinear tire
saturation or transient relaxation.

A small-sideslip quasi-steady approximation identifies the path tangential and
normal accelerations with body longitudinal and lateral acceleration. Tire
forces are then rotated through steering angles, and a 3x3 system solves
longitudinal/lateral force and yaw-moment balance. The yaw demand includes
Iz*(ax*kappa+v²*dkappa/ds). Steering and tire-force equilibrium are iterated;
nonconverged states are rejected. Vehicle sideslip dynamics and its time
derivative are not integrated.

The conventions and scope are consistent with standard planar force and
slip-angle modeling, while remaining deliberately simpler than calibrated
nonlinear tire models. Primary references:
[vehicle force/slip equations](https://www.mathworks.com/help/vdynblks/ref/vehiclebody3dof.html),
[longitudinal axle equilibrium](https://www.mathworks.com/help/sdl/ref/vehiclebody.html),
and [the calibration needs of a Magic Formula tire model](https://www.mathworks.com/help/sdl/ref/tiremagicformula.html).
Axis and slip-sign conventions in those references can differ; the definitions
above govern this implementation.

## Motors, braking, and conservative torque allocation

Each interpolated motor curve is bounded by peak torque, shaft power divided by
rotor speed, and the rotor-speed limit. The zero-speed power division is guarded;
there is no fictitious loss factor based on steering angle. Wheel rotational
speeds follow their individual steered contact velocities and rolling radius.

A fixed front/rear force split with equal left/right longitudinal forces on each
axle is conservative: the weaker tire/motor can limit the axle. Available
longitudinal tire force is the remaining combined-slip capacity. Electrical
discharge is sum(Fx_i*vwheel_i/efficiency_i). Regeneration is constrained by
individual torque/power, low-speed cutoff, and battery charge acceptance;
mechanical brakes provide the remaining force subject to torque and brake bias.

Longitudinal-force yaw includes the actual wheel positions and steering;
unsteered left/right differences use -sum(y_i*Fx_i), with distinct front and rear
tracks. The allocator is an extension point for a constrained torque-allocation
optimization. It does not claim optimal torque vectoring or motor efficiency.
Thermal limits, SOC evolution, voltage sag, and rolling resistance are omitted.

## Trajectories, speed, and the optimization objective

Boundary interpolation retains supplied geometry. Verified cone pairs use common
center-station interpolation; unmatched boundaries require geometric pairing.
Smooth lateral controls generate a continuous XY cubic spline with periodic
derivatives on closed tracks. Position, heading, signed curvature, curvature
gradient, and arc length are calculated from that candidate.

Clearance uses a disc enclosing the assumed centered body rectangle, plus the
boundary margin. This deliberately exceeds a width-only envelope. The smallest
sampled boundary distance is further reduced by half the largest dense sample
spacing to guard intervening excursions. Boundary distance and arc length remain
numerical approximations, not a formal analytic containment certificate. Open
entry/exit caps are gates rather than walls. Vehicle length and its centered
bounding footprint are assumptions to replace with an actual body envelope.

Local cornering ceilings are found using speed-dependent force feasibility.
Forward acceleration and backward braking passes iterate the load/grip/motor
constraints. The last-to-first segment participates in every closed-track pass.
Each segment uses constant tangential acceleration, so v² is linear in distance
and its time is 2 ds/(v_start+v_end), including launch from rest.

Dense spline-interior checks and both sides of every speed knot validate actual
acceleration, forces, slip, steering, and power. If needed, uniform speed reduction
repairs residual violations; the numerical reserve is configurable. This is
conservative and can reduce speed on unaffected parts of the lap. Prescribed
nonzero open-path endpoint speeds are never silently changed into accepted
results: a solution that cannot retain them is reported infeasible.

Deterministic coordinate search minimizes the estimated traversal time over
feasible smooth lateral controls. It is local, resolution dependent, and can
retain the centerline if no tried alternative improves it. Arc integration,
corner sampling, and conservative speed repair introduce numerical sensitivity.
Increase speed nodes, spline geometry anchors, curvature checks, and lateral
controls separately when studying convergence. There is no claim of a globally
optimal trajectory, continuous-time proof of all constraints, or measured lap-time accuracy.
