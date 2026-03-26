# IK Reachability Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `pyAgxArm/demos/piper/ik_reachability_map.py` — a script that sweeps a configurable 3D Cartesian grid, moves the Piper arm to each point, checks reachability by reading actual flange pose, logs to CSV, then shows a matplotlib figure (3D scatter + 3 slice heatmaps).

**Architecture:** Single script with a configurable constant block at the top. Robot setup → pre-position → sweep loop writing CSV → RTH + disable → generate plot. `generate_plot()` is a pure function taking the results list, making it independently testable with synthetic data.

**Tech Stack:** Python, pyAgxArm (piper/pcan), numpy, matplotlib

**Spec:** `docs/superpowers/specs/2026-03-25-ik-reachability-map-design.md`

---

## File Map

| Action | Path |
|--------|------|
| Create | `pyAgxArm/demos/piper/ik_reachability_map.py` |
| Create | `tests/demos/test_ik_reachability_plot.py` |

---

### Task 1: Scaffold file, config block, and grid generation

**Files:**
- Create: `pyAgxArm/demos/piper/ik_reachability_map.py`
- Create: `tests/demos/test_ik_reachability_plot.py`

- [ ] **Step 1: Create the test file for grid generation**

```python
# tests/demos/test_ik_reachability_plot.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def make_grid(lookup, fixed_axis, fixed_val, ax1_vals, ax2_vals):
    """Build 2D array for a slice. Returns array[ax2_idx, ax1_idx]."""
    arr = np.full((len(ax2_vals), len(ax1_vals)), np.nan)
    for i2, v2 in enumerate(ax2_vals):
        for i1, v1 in enumerate(ax1_vals):
            if fixed_axis == 'y':
                key = (round(v1, 4), round(fixed_val, 4), round(v2, 4))
            elif fixed_axis == 'z':
                key = (round(v1, 4), round(v2, 4), round(fixed_val, 4))
            else:
                key = (round(fixed_val, 4), round(v1, 4), round(v2, 4))
            if key in lookup:
                arr[i2, i1] = 1.0 if lookup[key] else 0.0
    return arr


def test_grid_xz_slice():
    xs = np.array([0.10, 0.15, 0.20])
    ys = np.array([0.00])
    zs = np.array([0.20, 0.25])
    lookup = {
        (0.10, 0.00, 0.20): True,
        (0.15, 0.00, 0.20): True,
        (0.20, 0.00, 0.20): False,
        (0.10, 0.00, 0.25): True,
        (0.15, 0.00, 0.25): False,
        (0.20, 0.00, 0.25): False,
    }
    mid_y = ys[len(ys) // 2]
    arr = make_grid(lookup, 'y', mid_y, xs, zs)
    assert arr.shape == (2, 3)
    assert arr[0, 0] == 1.0  # (x=0.10, z=0.20) reachable
    assert arr[0, 2] == 0.0  # (x=0.20, z=0.20) unreachable
    assert arr[1, 1] == 0.0  # (x=0.15, z=0.25) unreachable


def test_grid_untested_cells_are_nan():
    xs = np.array([0.10, 0.15])
    ys = np.array([0.00])
    zs = np.array([0.20, 0.25])
    lookup = {(0.10, 0.00, 0.20): True}   # only one point tested
    mid_y = ys[0]
    arr = make_grid(lookup, 'y', mid_y, xs, zs)
    assert arr[0, 0] == 1.0
    assert np.isnan(arr[0, 1])
    assert np.isnan(arr[1, 0])
    assert np.isnan(arr[1, 1])
```

- [ ] **Step 2: Run the test — expect ImportError or NameError (file doesn't exist yet)**

```bash
cd /Users/alexpro/PycharmProjects/pyAgxArm
.venv/bin/python -m pytest tests/demos/test_ik_reachability_plot.py -v
```

Expected: tests fail (module not found or similar). That's fine — we're verifying the test runs.

- [ ] **Step 3: Create the main script with config block and grid generation**

```python
# pyAgxArm/demos/piper/ik_reachability_map.py
"""Piper arm IK reachability map.

Sweeps a 3D Cartesian grid, moves the arm to each point, checks
reachability via proprioceptive feedback, logs to CSV, and shows a plot.

Usage:
    python pyAgxArm/demos/piper/ik_reachability_map.py
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from pyAgxArm import create_agx_arm_config, AgxArmFactory

# ---------------------------------------------------------------------------
# Configuration — edit these before each run
# ---------------------------------------------------------------------------
X_MIN, X_MAX, X_STEP = 0.05, 0.35, 0.05   # meters
Y_MIN, Y_MAX, Y_STEP = -0.25, 0.25, 0.05
Z_MIN, Z_MAX, Z_STEP = 0.20, 0.40, 0.05
ROLL, PITCH, YAW     = math.pi, 0.0, 0.0  # flange pointing down
TOL      = 0.03   # metres — counts as "reached"
SPEED    = 15     # % speed
LOG_FILE = "reachability.csv"

PRE_SWEEP_JOINTS = [0.0, 0.5, -1.0, 0.0, 0.5, math.pi]
HOME_JOINTS = [
    math.radians(-0.03),
    math.radians(-0.79),
    math.radians(+1.38),
    math.radians(+6.80),
    math.radians(+24.89),
    math.radians(+19.52),
]

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
xs = np.round(np.arange(X_MIN, X_MAX + X_STEP / 2, X_STEP), 4)
ys = np.round(np.arange(Y_MIN, Y_MAX + Y_STEP / 2, Y_STEP), 4)
zs = np.round(np.arange(Z_MIN, Z_MAX + Z_STEP / 2, Z_STEP), 4)
```

- [ ] **Step 4: Run the tests — grid tests should now pass**

```bash
.venv/bin/python -m pytest tests/demos/test_ik_reachability_plot.py -v
```

Expected: 2 tests pass. (The test file defines `make_grid` locally — it doesn't need to import the main script yet.)

- [ ] **Step 5: Commit**

```bash
git add pyAgxArm/demos/piper/ik_reachability_map.py tests/demos/test_ik_reachability_plot.py
git commit -m "feat: scaffold ik_reachability_map with config block and grid tests"
```

---

### Task 2: `wait_motion_done` and robot connection

**Files:**
- Modify: `pyAgxArm/demos/piper/ik_reachability_map.py` (append to end of file)

- [ ] **Step 1: Append `wait_motion_done` and robot setup to the script**

Append after the grid block:

```python
# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------

def wait_motion_done(robot, timeout: float = 20.0, poll_interval: float = 0.1) -> bool:
    """Two-phase wait: first for motion to start, then for it to finish."""
    # Phase 1: wait for motion to start (status goes non-zero)
    time.sleep(0.3)
    for _ in range(30):
        s = robot.get_arm_status()
        if s and getattr(s.msg, "motion_status", 0) != 0:
            break
        time.sleep(0.05)
    # Phase 2: wait for motion to finish (status returns to zero)
    start_t = time.monotonic()
    while True:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            return True
        if time.monotonic() - start_t > timeout:
            print(f"  timeout ({timeout:.0f}s)")
            return False
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Robot setup
# ---------------------------------------------------------------------------
robot_cfg = create_agx_arm_config(
    robot="piper", comm="can", channel="PCAN_USBBUS1", interface="pcan"
)
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
robot.set_follower_mode()
time.sleep(0.1)

while not robot.enable():
    time.sleep(0.01)
robot.set_speed_percent(SPEED)

# Pre-position: arm upright, J6=π — prevents surprise wrist spin on first move_p
print("Pre-positioning arm...")
robot.move_j(PRE_SWEEP_JOINTS)
wait_motion_done(robot)
time.sleep(0.3)
print(f"Grid: {len(xs)}×{len(ys)}×{len(zs)} = {len(xs)*len(ys)*len(zs)} points")
print(f"Logging to {LOG_FILE}")
```

- [ ] **Step 2: Commit**

```bash
git add pyAgxArm/demos/piper/ik_reachability_map.py
git commit -m "feat: add wait_motion_done and robot setup to reachability map"
```

---

### Task 3: Sweep loop with CSV logging

**Files:**
- Modify: `pyAgxArm/demos/piper/ik_reachability_map.py` (append)

- [ ] **Step 1: Append the sweep loop**

```python
# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
csv_f = open(LOG_FILE, "w")
csv_f.write("x,y,z,actual_x,actual_y,actual_z,error_m,reachable\n")
csv_f.flush()

results = []   # list of (x, y, z, ax, ay, az, error, reachable)
total = len(xs) * len(ys) * len(zs)
count = 0

try:
    for x in xs:
        for y in ys:
            for z in zs:
                count += 1
                print(f"[{count}/{total}] X={x:.2f} Y={y:.2f} Z={z:.2f} ... ",
                      end="", flush=True)
                robot.move_p([x, y, z, ROLL, PITCH, YAW])
                reached = wait_motion_done(robot)

                pose = robot.get_flange_pose()
                if pose is None or not reached:
                    ax, ay, az = float("nan"), float("nan"), float("nan")
                    error = float("inf")
                    reachable = False
                else:
                    ax, ay, az = pose.msg[0], pose.msg[1], pose.msg[2]
                    error = math.sqrt((ax - x) ** 2 + (ay - y) ** 2 + (az - z) ** 2)
                    reachable = error < TOL

                tag = "OK  " if reachable else "MISS"
                print(f"err={error * 1000:6.1f}mm [{tag}]")

                row = (x, y, z, ax, ay, az, error, reachable)
                results.append(row)
                csv_f.write(
                    f"{x},{y},{z},{ax},{ay},{az},{error},{reachable}\n"
                )
                csv_f.flush()

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    csv_f.close()
```

- [ ] **Step 2: Commit**

```bash
git add pyAgxArm/demos/piper/ik_reachability_map.py
git commit -m "feat: add sweep loop with per-point CSV logging"
```

---

### Task 4: RTH, disable, and summary

**Files:**
- Modify: `pyAgxArm/demos/piper/ik_reachability_map.py` (append)

- [ ] **Step 1: Append RTH, disable, and summary**

```python
# ---------------------------------------------------------------------------
# Return to home & disable
# ---------------------------------------------------------------------------
print("\nReturning to home...")
robot.move_j(HOME_JOINTS)
wait_motion_done(robot)

while not robot.disable():
    time.sleep(0.01)
print("Arm disabled.")

n_reach = sum(1 for r in results if r[7])
print(f"\nSummary: {n_reach}/{len(results)} points reachable")
print(f"Logged to {LOG_FILE}")
```

- [ ] **Step 2: Commit**

```bash
git add pyAgxArm/demos/piper/ik_reachability_map.py
git commit -m "feat: add RTH, disable, and summary to reachability map"
```

---

### Task 5: `generate_plot` function and tests

**Files:**
- Modify: `pyAgxArm/demos/piper/ik_reachability_map.py` (insert before robot setup block)
- Modify: `tests/demos/test_ik_reachability_plot.py` (add plot tests)

- [ ] **Step 1: Add plot tests using synthetic data**

Append to `tests/demos/test_ik_reachability_plot.py`:

```python
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for tests
import matplotlib.pyplot as plt
import importlib.util, pathlib

# Load generate_plot without executing robot code
# We test by calling generate_plot with synthetic results
def _load_generate_plot():
    """Parse generate_plot out of the main script without running robot setup."""
    src = pathlib.Path("pyAgxArm/demos/piper/ik_reachability_map.py").read_text()
    # Extract only up to the robot setup line
    cutoff = src.find("robot_cfg = create_agx_arm_config")
    trimmed = src[:cutoff]
    ns = {}
    exec(trimmed, ns)
    return ns["generate_plot"]


def _synthetic_results():
    xs = np.round(np.arange(0.10, 0.21, 0.05), 4)
    ys = np.round(np.arange(-0.05, 0.06, 0.05), 4)
    zs = np.round(np.arange(0.20, 0.31, 0.05), 4)
    results = []
    for x in xs:
        for y in ys:
            for z in zs:
                reachable = (x <= 0.15 and abs(y) <= 0.05 and z <= 0.25)
                results.append((x, y, z, x, y, z, 0.0, reachable))
    return results, xs, ys, zs


def test_generate_plot_runs_without_error():
    generate_plot = _load_generate_plot()
    results, xs, ys, zs = _synthetic_results()
    generate_plot(results, xs, ys, zs)
    plt.close('all')


def test_generate_plot_empty_results():
    generate_plot = _load_generate_plot()
    generate_plot([], np.array([0.1]), np.array([0.0]), np.array([0.2]))
    plt.close('all')
```

- [ ] **Step 2: Run tests — expect failure (generate_plot not defined yet)**

```bash
.venv/bin/python -m pytest tests/demos/test_ik_reachability_plot.py -v
```

Expected: `test_generate_plot_runs_without_error` and `test_generate_plot_empty_results` fail.

- [ ] **Step 3: Insert `generate_plot` into the script — place it just before the `robot_cfg = ...` line**

```python
# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_slice_grid(lookup, fixed_axis, fixed_val, ax1_vals, ax2_vals):
    """Build 2D array for a heatmap slice.

    Returns array[ax2_idx, ax1_idx]:
      fixed_axis='y' → ax1=X, ax2=Z  (XZ plane)
      fixed_axis='z' → ax1=X, ax2=Y  (XY plane)
      fixed_axis='x' → ax1=Y, ax2=Z  (YZ plane)
    NaN = not tested, 1.0 = reachable, 0.0 = unreachable.
    """
    arr = np.full((len(ax2_vals), len(ax1_vals)), np.nan)
    for i2, v2 in enumerate(ax2_vals):
        for i1, v1 in enumerate(ax1_vals):
            if fixed_axis == "y":
                key = (round(v1, 4), round(fixed_val, 4), round(v2, 4))
            elif fixed_axis == "z":
                key = (round(v1, 4), round(v2, 4), round(fixed_val, 4))
            else:  # fixed x
                key = (round(fixed_val, 4), round(v1, 4), round(v2, 4))
            if key in lookup:
                arr[i2, i1] = 1.0 if lookup[key] else 0.0
    return arr


def generate_plot(results, xs, ys, zs):
    """Build and show the reachability figure from a results list.

    results: list of (x, y, z, actual_x, actual_y, actual_z, error_m, reachable)
    xs, ys, zs: 1D numpy arrays of grid values (used for heatmap extent/ticks)
    """
    if not results:
        print("No results to plot.")
        return

    reach_pts  = [(r[0], r[1], r[2]) for r in results if r[7]]
    miss_pts   = [(r[0], r[1], r[2]) for r in results if not r[7]]
    lookup     = {(round(r[0],4), round(r[1],4), round(r[2],4)): r[7] for r in results}

    fig = plt.figure(figsize=(14, 6))
    gs  = GridSpec(3, 3, figure=fig)

    # --- 3D scatter (left 2 columns) ---
    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    if reach_pts:
        ax3d.scatter(*zip(*reach_pts), c="green", marker="o", s=30,
                     label="reachable", alpha=0.7)
    if miss_pts:
        ax3d.scatter(*zip(*miss_pts), c="red", marker="x", s=30,
                     label="unreachable", alpha=0.7)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"Reachable workspace ({len(reach_pts)}/{len(results)} points)")
    ax3d.legend(loc="upper left", fontsize=8)

    # Colormap: 0=red, 1=green, NaN=grey
    cmap = mcolors.LinearSegmentedColormap.from_list("rg", ["red", "green"], N=2)
    cmap.set_bad(color="lightgrey")

    mid_y = ys[len(ys) // 2]
    mid_z = zs[len(zs) // 2]
    mid_x = xs[len(xs) // 2]

    x_step = xs[1] - xs[0] if len(xs) > 1 else 0.05
    y_step = ys[1] - ys[0] if len(ys) > 1 else 0.05
    z_step = zs[1] - zs[0] if len(zs) > 1 else 0.05

    def _show_slice(subplot_spec, fixed_axis, fixed_val, ax1_vals, ax2_vals,
                    xlabel, ylabel, title, x_extent, y_extent):
        ax = fig.add_subplot(subplot_spec)
        arr = _make_slice_grid(lookup, fixed_axis, fixed_val, ax1_vals, ax2_vals)
        ax.imshow(arr, origin="lower", cmap=cmap, vmin=0, vmax=1,
                  extent=x_extent + y_extent, aspect="auto")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)

    _show_slice(
        gs[0, 2], "y", mid_y, xs, zs,
        "X (m)", "Z (m)", f"XZ plane (Y={mid_y:.2f} m)",
        [xs[0] - x_step/2, xs[-1] + x_step/2],
        [zs[0] - z_step/2, zs[-1] + z_step/2],
    )
    _show_slice(
        gs[1, 2], "z", mid_z, xs, ys,
        "X (m)", "Y (m)", f"XY plane (Z={mid_z:.2f} m)",
        [xs[0] - x_step/2, xs[-1] + x_step/2],
        [ys[0] - y_step/2, ys[-1] + y_step/2],
    )
    _show_slice(
        gs[2, 2], "x", mid_x, ys, zs,
        "Y (m)", "Z (m)", f"YZ plane (X={mid_x:.2f} m)",
        [ys[0] - y_step/2, ys[-1] + y_step/2],
        [zs[0] - z_step/2, zs[-1] + z_step/2],
    )

    plt.tight_layout()
    plt.show()
```

- [ ] **Step 4: Also append the `generate_plot` call at the very end of the file (after the summary print)**

```python
generate_plot(results, xs, ys, zs)
```

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/python -m pytest tests/demos/test_ik_reachability_plot.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyAgxArm/demos/piper/ik_reachability_map.py tests/demos/test_ik_reachability_plot.py
git commit -m "feat: add generate_plot with 3D scatter and 3 slice heatmaps, add plot tests"
```

---

### Task 6: Verify final file structure and do a dry-run check

**Files:**
- Read: `pyAgxArm/demos/piper/ik_reachability_map.py`

- [ ] **Step 1: Check the full file looks right**

```bash
cat -n pyAgxArm/demos/piper/ik_reachability_map.py
```

Verify section order:
1. Docstring
2. Imports
3. Config block (X_MIN … LOG_FILE)
4. PRE_SWEEP_JOINTS, HOME_JOINTS
5. Grid (xs, ys, zs)
6. `_make_slice_grid()`
7. `generate_plot()`
8. Robot setup (robot_cfg … pre-position)
9. CSV open
10. Sweep try/except/finally
11. RTH + disable
12. Summary print
13. `generate_plot(results, xs, ys, zs)`

- [ ] **Step 2: Run all tests one final time**

```bash
.venv/bin/python -m pytest tests/demos/test_ik_reachability_plot.py -v
```

Expected: 4 tests pass.

- [ ] **Step 3: Syntax-check the script without running robot code**

```bash
.venv/bin/python -c "
import ast, pathlib
src = pathlib.Path('pyAgxArm/demos/piper/ik_reachability_map.py').read_text()
ast.parse(src)
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 4: Final commit if any fixups were needed**

```bash
git add pyAgxArm/demos/piper/ik_reachability_map.py
git commit -m "fix: final cleanup of ik_reachability_map" --allow-empty
```

---

## Notes

- All `pytest` and `python` commands must be run from the **project root** (`/Users/alexpro/PycharmProjects/pyAgxArm`). The test file uses a relative path to load the main script.
- `tests/demos/` directory may not exist yet — create it with `mkdir -p tests/demos` and add an empty `__init__.py` if needed.

## Running on the robot

```bash
.venv/bin/python pyAgxArm/demos/piper/ik_reachability_map.py
```

- Arm pre-positions, then sweeps ~385 points at 15% speed (~45–70 min at defaults)
- Progress printed per point: `[42/385] X=0.10 Y=0.05 Z=0.25 ... err=  8.3mm [OK  ]`
- Results saved to `reachability.csv` as it runs
- Ctrl+C at any time → RTH → disable → plot whatever was collected
- Interactive matplotlib window opens at end — rotate the 3D scatter to explore workspace shape
