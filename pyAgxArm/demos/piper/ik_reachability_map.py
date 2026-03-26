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
