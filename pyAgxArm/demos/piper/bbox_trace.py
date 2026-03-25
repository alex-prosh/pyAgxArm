"""Piper arm bounding-box trace demo.

Moves the end-effector through all 8 corners of a Cartesian bounding box
using move_p, very slowly (5% speed). Tests the reachable workspace.

Bounding box:
    X: 0.00 – 0.30 m
    Y: -0.30 – +0.30 m
    Z: 0.075 – 0.20 m
    Orientation: roll=π, pitch=0, yaw=0 (flange pointing down)

Usage:
    1. Activate CAN: bash pyAgxArm/scripts/can_activate.sh can0
    2. Run: python pyAgxArm/demos/piper/bbox_trace.py
    3. Ctrl+C to stop — the arm will disable gracefully.
"""

import argparse
import math
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

parser = argparse.ArgumentParser(description="Trace bounding box corners")
parser.add_argument("--linear", "-l", action="store_true",
                    help="Use move_l (straight-line) instead of move_p (point-to-point)")
args = parser.parse_args()

# Orientation: roll=180°, pitch=0, yaw=0
ROLL = math.pi
PITCH = 0.0
YAW = 0.0

# --- Bounding box limits (edit these) ---
X_MIN, X_MAX = 0.00, 0.25
Y_MIN, Y_MAX = -0.25, 0.25
Z_MIN, Z_MAX = 0.075, 0.20

# 8 corners generated from limits, ordered to trace all 12 edges:
# bottom rectangle → up → top rectangle → back down
ORI = [ROLL, PITCH, YAW]
CORNERS = [
    [X_MIN, Y_MIN, Z_MIN] + ORI,  # P0 bottom-near-left
    [X_MAX, Y_MIN, Z_MIN] + ORI,  # P1 bottom-far-left
    [X_MAX, Y_MAX, Z_MIN] + ORI,  # P2 bottom-far-right
    [X_MIN, Y_MAX, Z_MIN] + ORI,  # P3 bottom-near-right
    [X_MIN, Y_MAX, Z_MAX] + ORI,  # P4 top-near-right
    [X_MAX, Y_MAX, Z_MAX] + ORI,  # P5 top-far-right
    [X_MAX, Y_MIN, Z_MAX] + ORI,  # P6 top-far-left
    [X_MIN, Y_MIN, Z_MAX] + ORI,  # P7 top-near-left
]


def wait_motion_done(robot, timeout: float = 15.0, poll_interval: float = 0.1) -> bool:
    """Wait until `robot.get_arm_status().msg.motion_status == 0` or timeout."""
    time.sleep(0.5)
    start_t = time.monotonic()
    while True:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            print("  motion done")
            return True
        if time.monotonic() - start_t > timeout:
            print(f"  wait motion done timeout ({timeout:.1f}s)")
            return False
        time.sleep(poll_interval)


# --- Connect ---
robot_cfg = create_agx_arm_config(robot="piper", comm="can", channel="can0", interface="socketcan")
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()

# --- Set follower mode so arm responds to motion commands ---
robot.set_follower_mode()
time.sleep(0.1)

# --- Enable & configure ---
while not robot.enable():
    time.sleep(0.01)
robot.set_speed_percent(25)  # very slow for workspace mapping

move_fn = robot.move_l if args.linear else robot.move_p
mode_name = "move_l (linear)" if args.linear else "move_p (point-to-point)"
print(f"Motion mode: {mode_name}")

# --- Trace bounding box corners ---
try:
    for i, pose in enumerate(CORNERS):
        print(f"P{i}: x={pose[0]:.3f} y={pose[1]:.3f} z={pose[2]:.3f}")
        move_fn(pose)
        wait_motion_done(robot)
        print(f"  holding P{i} for 1s...")
        time.sleep(1.0)

    # Close the loop: return to P0
    print(f"P0 (close loop): x={CORNERS[0][0]:.3f} y={CORNERS[0][1]:.3f} z={CORNERS[0][2]:.3f}")
    move_fn(CORNERS[0])
    wait_motion_done(robot)
    time.sleep(1.0)

    print("Bounding box trace complete.")
except KeyboardInterrupt:
    print("\nInterrupted — stopping...")

# --- Return to home (resting pose when unpowered) & disable ---
# Joint angles in radians matching the arm's natural resting position
HOME_JOINTS = [
    math.radians(-0.03),   # J1
    math.radians(-0.79),   # J2
    math.radians(+1.38),   # J3
    math.radians(+6.80),   # J4
    math.radians(+24.89),  # J5
    math.radians(+19.52),  # J6
]
print("Returning to home...")
robot.set_speed_percent(30)
robot.move_j(HOME_JOINTS)
wait_motion_done(robot)
time.sleep(0.5)

while not robot.disable():
    time.sleep(0.01)
print("Arm disabled. Done.")
