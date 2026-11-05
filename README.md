# BGRacing racing-line optimizer

The primary implementation is [optimal/optimal.py](optimal/optimal.py). It searches
smooth lateral trajectories for a lower **estimated lap time** using a four-wheel,
four-motor quasi-static vehicle model. Every trial uses its own curvature and a
coupled acceleration/braking speed profile. This is a local optimizer, not a
guarantee of the globally fastest line.

All vehicle values are explicitly **ASSUMED**. Results are simulation estimates,
not measured BGRacing performance. The original `BrandsHatchLayout.csv` and legacy
`bgracing.py` are preserved. CSV X/Y coordinates are assumed to be metres.

## Run

Target Python: **3.12**. Create a virtual environment and install the four dependencies:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python optimal/optimal.py
```

The default run saves files under `output/` and requires no graphical display.
For a shorter search, use `--quick`; coarse searches can legitimately return the
centerline if none of the tried candidates improves its feasible estimated time.

```bash
.venv/bin/python optimal/optimal.py --quick
.venv/bin/python optimal/optimal.py --samples 360 --controls 12 --sweeps 2
.venv/bin/python optimal/optimal.py --topology open --start-speed 0 --finish-speed 0
.venv/bin/python optimal/optimal.py --output output/experiment --show
.venv/bin/python -m unittest discover -s tests -v
```

Use `--help` for all command-line options. For an open path, omitted
`--finish-speed` means a free terminal speed. For a closed path, the closing
segment is included and the start/finish speed is the same periodic variable;
open-path endpoint settings do not impose a standing start on a circuit.

Edit the frozen `VehicleParameters`, `MotorParameters`, and `SolverSettings`
dataclasses at the top of the module to change model inputs. Each wheel has its
own immutable motor configuration. Simulation arrays live in separate state
dataclasses. Importing the module does not run the simulation or open a plot.

## Outputs

- `racing_line.png`: boundaries, centerline, speed-colored optimized line,
  speed comparison, both accelerations, four normal loads, tire utilization,
  battery power, and approximate tire slip.
- `summary.json`: lap-time comparison, physical checks, source CSV SHA-256,
  complete assumed vehicle configuration, solver settings, and lateral controls.
- `speed_profile.csv`: position, heading, curvature, arc station, and speed.
- `constraint_validation.csv`: dense endpoint/interior checks including
  available, requested, and actual forces on every wheel.

Closed-path arrays omit the duplicated endpoint; the closing segment still
contributes to time and physical checks. Validation CSV distances can repeat:
acceleration immediately before and after each speed knot is checked separately.
The summary only marks model constraints satisfied when all implemented checks
pass. A failed baseline or invalid input raises an error.

## What changed

The previous position-only dynamic program used centerline curvature for all
lateral candidates and did not carry speed/history state. The new deterministic
coordinate search evaluates a continuous spline and a full speed profile for
each candidate. It retains the best feasible result, including the centerline
baseline, and reports the comparison.

Track input validates columns, finite values, ordering, crossings, orientation,
and topology. Matching row counts alone are insufficient: original row pairs
must pass transverse alignment and connector-crossing checks. Other datasets
use transverse normal intersections with monotone matching checks, or fail
explicitly if a safe correspondence cannot be established. Boundary splines
use shape-preserving interpolation. Candidate spline overshoot is rejected
by corridor checks.

Automatic topology detection uses endpoint spacing and tangent alignment, and
prints the decision. Use `--topology open` or `--topology closed` when known.
The supplied data passes closed-track checks. Two fully empty coordinate rows
are treated as separators; partially missing or nonfinite coordinates are errors.

See [the physical model and assumption list](docs/MODEL.md) and
[recorded validation and example results](docs/VALIDATION.md).

## Measurements to obtain first

1. Loaded corner weights, total mass, CG position/height, wheelbase, tracks,
   body envelope, and effective rolling radius.
2. Tire lateral/longitudinal force versus normal load, slip angle/ratio, and
   combined slip. These replace the assumed grip, load sensitivity, stiffness,
   and small-slip limit.
3. Each motor's torque-speed and regenerative envelopes, efficiency maps,
   and battery discharge/charge limits versus state of charge and temperature.
4. Mechanical brake torque and bias; front/rear roll stiffness and roll centers.
5. Downforce, drag, aerodynamic balance, yaw inertia, and surveyed track units
   and boundary uncertainty.

No ROS2 or Docker integration is included in this phase.
