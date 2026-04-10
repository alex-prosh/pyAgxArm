"""Sweep X, Y, Z positions and report which are reachable.

Tries each position, waits, reads actual flange pose, reports error.

Usage:
    python pyAgxArm/demos/piper/workspace_sweep.py
"""

import math
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

ROLL  = math.pi
PITCH = 0.0
YAW   = 0.0

# Grid to test — edit these
XS = [0.10, 0.15, 0.20]
YS = [0.0]
ZS = [0.20, 0.25, 0.30, 0.35, 0.40]

TOL   = 0.03   # 30mm tolerance to count as "reached"
SPEED = 20

# Pre-sweep joints: arm roughly upright with J6 at 180° to match ROLL=π,
# so no surprise rotation when the first Cartesian move is sent.
PRE_SWEEP_JOINTS = [0.0, 0.5, -1.0, 0.0, 0.5, math.pi]


def wait_motion_done(robot, timeout=20.0):
    # wait for motion to start
    time.sleep(0.4)
    for _ in range(20):
        s = robot.get_arm_status()
        if s and getattr(s.msg, "motion_status", 0) != 0:
            break
        time.sleep(0.05)
    # wait for motion to finish
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        s = robot.get_arm_status()
        if s and getattr(s.msg, "motion_status", None) == 0:
            return True
        time.sleep(0.1)
    return False


def check_pose(robot, tx, ty, tz):
    time.sleep(0.3)
    p = robot.get_flange_pose()
    if p is None:
        return None, None, None
    ax, ay, az = p.msg[0], p.msg[1], p.msg[2]
    err = math.sqrt((ax-tx)**2 + (ay-ty)**2 + (az-tz)**2)
    return ax, az, err


robot_cfg = create_agx_arm_config(robot="piper", comm="can", channel="PCAN_USBBUS1", interface="pcan")
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
robot.set_follower_mode()
time.sleep(0.1)

while not robot.enable():
    time.sleep(0.01)
robot.set_speed_percent(SPEED)

# Pre-position J6 to 180° so sweep starts without a sudden wrist rotation
print("Pre-positioning arm...")
robot.move_j(PRE_SWEEP_JOINTS)
wait_motion_done(robot)

results = []

try:
    for x in XS:
        for y in YS:
            for z in ZS:
                print(f"  X={x:.2f} Y={y:.2f} Z={z:.2f} ... ", end="", flush=True)
                robot.move_p([x, y, z, ROLL, PITCH, YAW])
                wait_motion_done(robot)
                ax, az, err = check_pose(robot, x, y, z)
                if err is None:
                    print("no feedback")
                    continue
                reached = err < TOL
                status = "OK" if reached else "MISS"
                print(f"actual_z={az:.3f} err={err*1000:.0f}mm  [{status}]")
                results.append((x, y, z, az, err, reached))

except KeyboardInterrupt:
    print("\nStopped.")

print("\n=== Summary ===")
print(f"{'X':>6} {'Y':>6} {'Z':>6} {'act_Z':>7} {'err_mm':>7} {'ok':>4}")
for x, y, z, az, err, ok in results:
    print(f"{x:6.2f} {y:6.2f} {z:6.2f} {az:7.3f} {err*1000:7.1f} {'✓' if ok else '✗'}")

while not robot.disable():
    time.sleep(0.01)
print("Done.")
