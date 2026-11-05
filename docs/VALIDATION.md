# Validation and reproducible example

These are numerical simulation results with ASSUMED vehicle inputs, not experimental validation.

## Execution

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python optimal/optimal.py --output output/final
```

The final regression suite passed **12 tests**. The full default search evaluated **144 candidates** and completed successfully. Test runtime: Python **3.13.7** (the available Linux environment); production source syntax was also checked with Python's 3.12 grammar. Python 3.12 is the project target, but a 3.12 interpreter was not available for runtime tests.

Installed scientific libraries: NumPy 2.5.4, SciPy 1.18.1, Pandas 3.1.1, Matplotlib 3.11.2.

## Default-grid example

| Quantity | Result |
|---|---:|
| Centerline baseline estimate | 123.948 s |
| Optimized estimate | 121.005 s |
| Estimated improvement at this grid | 2.943 s (2.37%) |
| Interpolated track length | 3920.257 m |
| Racing-line length | 3891.601 m |
| Peak speed | 37.605 m/s |
| Maximum lateral acceleration | 23.023 m/s² |
| Maximum longitudinal acceleration | 5.117 m/s² |
| Maximum braking deceleration | 15.594 m/s² |
| Maximum tire utilization | 0.999985 |
| Minimum normal load | 215.725 N |
| Wheel-lift events | 0 |
| Peak electrical discharge | 40.000 kW |
| Peak electrical regeneration | 20.000 kW |
| Conservative residual boundary clearance | 0.195 m |
| All implemented physical checks | PASS |

Propagation converged; 34 local dense-validation repair passes; endpoint speeds satisfied.

Complete settings, per-wheel motor data, and lateral controls are recorded in [example_summary.json](example_summary.json). Generated plots and detailed CSVs are available locally under `output/final/`; output directories are intentionally ignored by Git.

## Speed-grid sensitivity

This check re-evaluated the SAME final continuous line and centerline, with unchanged spline geometry and controls. It did not re-optimize at each resolution.

| Speed nodes | Centerline [s] | Fixed optimized line [s] | Model checks |
|---:|---:|---:|---|
| 180 | 123.948 | 121.005 | PASS |
| 360 | 122.268 | 121.264 | PASS |
| 720 | 122.451 | 120.245 | PASS |

The lap-time estimate is resolution dependent. This table is a sensitivity check, not a claim of numerical convergence. Re-optimize at the intended resolution and test further refinement before interpreting a small estimated advantage as a reliable design improvement. Local feasibility repair remains conservative; the search is local, and the vehicle parameters are hypothetical.

## Test coverage and source preservation

Tests cover axle/corner equilibrium and moments, total normal-load conservation, quadratic aero scaling, longitudinal/lateral transfer, wheel unloading, combined-slip utilization and load sensitivity, motor torque/power/speed limits, independent allocation, battery discharge/regen limits, rotated force/yaw equilibrium, invalid inputs, candidate-specific curvature, footprint compliance, open-path launch/stop, closed-track periodicity, and acceleration propagation against an independently feasible reference.

The CSV contains 156 cones on each side and two empty coordinate separators. Verified original row correspondence is used; automatic topology checks identify a closed track. The source CSV and legacy `bgracing.py` are unchanged from the starting commit. CSV SHA-256:

`890bddb3b8a2e9844bcc16555ff51ae5aad901b4cd1fc7f8bec3436596ba68e1`

No experimental vehicle or tire data were introduced. The remaining physical approximations and measurement priorities are listed in [MODEL.md](MODEL.md) and the [README](../README.md).
