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

generate_plot(results, xs, ys, zs)
