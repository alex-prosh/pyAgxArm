# IK Reachability Map — Design Spec
**Date:** 2026-03-25
**File to create:** `pyAgxArm/demos/piper/ik_reachability_map.py`

## Overview

A single script that sweeps a configurable 3D Cartesian grid, moves the Piper arm to each point via `move_p`, checks whether the arm actually reached the target (proprioceptive check), logs results to CSV, then generates a matplotlib figure showing reachable vs unreachable positions.

## Configuration

All parameters live at the top of the file so users can adjust without reading the logic:

```python
X_MIN, X_MAX, X_STEP = 0.05, 0.35, 0.05   # meters
Y_MIN, Y_MAX, Y_STEP = -0.25, 0.25, 0.05
Z_MIN, Z_MAX, Z_STEP = 0.20, 0.40, 0.05
ROLL, PITCH, YAW     = math.pi, 0.0, 0.0   # flange pointing down
TOL      = 0.03   # 30 mm — distance threshold to count as "reached"
SPEED    = 15     # % speed
LOG_FILE = "reachability.csv"
```

Default grid at 5 cm steps: ~7×11×5 = ~385 points.

## Robot Setup

Same pattern as all other demos:
- `create_agx_arm_config(robot="piper", comm="can", channel="PCAN_USBBUS1", interface="pcan")`
- `set_follower_mode()` → `enable()` → `set_speed_percent(SPEED)`

## Pre-positioning

Before the sweep starts, move to `PRE_SWEEP_JOINTS` (arm upright, J6=π) via `move_j`. This ensures no surprise wrist spin on the first Cartesian command, consistent with the approach used in `workspace_sweep.py`.

```python
PRE_SWEEP_JOINTS = [0.0, 0.5, -1.0, 0.0, 0.5, math.pi]
```

## Sweep Loop

Iteration order: X (outermost) → Y → Z (innermost).

For each `(x, y, z)`:
1. Send `robot.move_p([x, y, z, ROLL, PITCH, YAW])`
2. Call `wait_motion_done(robot)` — two-phase: wait for `motion_status != 0` (started), then wait for `motion_status == 0` (finished)
3. Call `robot.get_flange_pose()` — read actual position
4. Compute Euclidean error between actual and target
5. `reachable = (error < TOL)` — if `get_flange_pose()` returns `None`, mark as unreachable
6. Append one row to `LOG_FILE` immediately (no buffering)
7. Print per-point status to stdout

## CSV Format

File is opened with `mode='w'` at the start of each run (overwrites previous results). Header written once, then one row appended per point as it completes.

```
x,y,z,actual_x,actual_y,actual_z,error_m,reachable
0.10,0.00,0.20,0.101,0.001,0.199,0.002,True
0.35,0.25,0.40,0.000,0.000,0.000,0.612,False
```

## Interrupt & Cleanup

Both `KeyboardInterrupt` and normal end:
1. `move_j(HOME_JOINTS)` — wait for RTH to complete
2. `robot.disable()`
3. Generate and show plot from whatever rows were collected

`HOME_JOINTS` matches the arm's natural resting pose (same values used across all demos).

## `wait_motion_done`

Two-phase implementation (same as `up_down.py` and `wrist_oscillation.py`):
- Phase 1: sleep 0.3s, then poll up to 30× at 0.05s intervals for `motion_status != 0`
- Phase 2: poll until `motion_status == 0` or timeout (default 20s)

## Plot

Generated with matplotlib from the collected results list (not from the CSV file — results are kept in memory during the run).

**Layout (wide):** `plt.figure(figsize=(14, 6))` with `gridspec`:
- Left half (`colspan=2`): 3D scatter plot (`projection='3d'`)
  - Green dots (`marker='o'`): reachable points
  - Red dots (`marker='x'`): unreachable points
  - Axes labeled X, Y, Z in meters
  - Title: "Reachable workspace (N/M points)"
- Right half: 3 stacked 2D heatmap subplots
  - **XZ** at middle Y value — axes: X (horizontal), Z (vertical)
  - **XY** at middle Z value — axes: X (horizontal), Y (vertical)
  - **YZ** at middle X value — axes: Y (horizontal), Z (vertical)
  - Each cell: green if reachable, red if unreachable, grey if not tested. Implemented as a 2D numpy array initialised to `NaN`; only tested points write their value (1.0=reachable, 0.0=unreachable); `NaN` cells render grey via `cmap.set_bad()`.
  - Slice label shows the fixed coordinate value (e.g. "XZ plane at Y=0.00 m")

Ends with `plt.tight_layout()` then `plt.show()` (interactive, rotatable 3D).

## Dependencies

- `matplotlib` (already expected in the project environment)
- `numpy` (for `arange` grid generation)
- `pyAgxArm` (existing)

## Error Handling

- If `get_flange_pose()` returns `None`: log as unreachable with `actual_x/y/z = NaN`, continue sweep
- If `wait_motion_done` times out: log as unreachable (arm couldn't reach), continue to next point
- No retry logic — each point is attempted once
