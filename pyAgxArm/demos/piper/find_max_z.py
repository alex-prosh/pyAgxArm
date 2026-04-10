"""Find max reachable Z by stepping upward until IK fails.

Starts at a center workspace position and increments Z by a small step,
stopping when the arm can't reach or times out.

Usage:
    python pyAgxArm/demos/piper/find_max_z.py
"""

import math
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

HOME_JOINTS = [
    math.radians(-0.03),   # J1
    math.radians(-0.79),   # J2
    math.radians(+1.38),   # J3
    math.radians(+6.80),   # J4
    math.radians(+24.89),  # J5
    math.radians(+19.52),  # J6
]

X = 0.15       # center-ish X
Y = 0.0        # center Y
Z_START = 0.22 # start Z (known reachable)
Z_STEP  = 0.01 # increment per step (1 cm)
ROLL    = math.pi
PITCH   = 0.0
YAW     = 0.0

def wait_motion_done(robot, timeout: float = 15.0, poll_interval: float = 0.1) -> bool:
    time.sleep(0.5)
    start_t = time.monotonic()
    while True:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            return True
        if time.monotonic() - start_t > timeout:
            return False
        time.sleep(poll_interval)

def wait_and_check(robot, target_z, timeout: float = 30.0, poll_interval: float = 0.1, tol: float = 0.03):
    """Wait for motion to settle, then check if actual Z matches target."""
    # Phase 1: wait for motion to start (status goes non-zero)
    time.sleep(0.3)
    for _ in range(20):
        status = robot.get_arm_status()
        ms = getattr(status.msg, "motion_status", None) if status else None
        if ms != 0:
            break
        time.sleep(0.05)

    # Phase 2: wait for motion to finish (status returns to zero)
    start_t = time.monotonic()
    timed_out = False
    while True:
        status = robot.get_arm_status()
        ms = getattr(status.msg, "motion_status", None) if status else None
        if ms == 0:
            break
        if time.monotonic() - start_t > timeout:
            timed_out = True
            break
        time.sleep(poll_interval)

    time.sleep(0.2)  # let pose feedback settle
    pose_msg = robot.get_flange_pose()
    if pose_msg is None:
        return "no_feedback", None
    actual_z = pose_msg.msg[2]
    error = abs(actual_z - target_z)
    print(f"target={target_z:.3f} actual={actual_z:.3f} err={error*1000:.1f}mm {'(timeout)' if timed_out else ''}", end=" ... ")
    if error > tol:
        return "unreachable", actual_z
    return "done", actual_z

robot_cfg = create_agx_arm_config(robot="piper", comm="can", channel="PCAN_USBBUS1", interface="pcan")
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
robot.set_follower_mode()
time.sleep(0.1)

while not robot.enable():
    time.sleep(0.01)
robot.set_speed_percent(15)

# Move to a known reachable starting pose before stepping
print(f"Moving to starting pose X={X} Y={Y} Z={Z_START}...")
robot.move_p([X, Y, Z_START, ROLL, PITCH, YAW])
wait_motion_done(robot)
time.sleep(0.5)

print(f"Stepping Z from {Z_START:.3f}m upward in {Z_STEP*100:.0f}cm steps at X={X}, Y={Y}")
print("─" * 50)

last_good_z = None
z = Z_START

try:
    while True:
        pose = [X, Y, z, ROLL, PITCH, YAW]
        print(f"Z = {z:.3f} m ... ", end="", flush=True)
        robot.move_p(pose)
        result, actual_z = wait_and_check(robot, z)
        print(result)

        if result == "done":
            last_good_z = z
            z = round(z + Z_STEP, 4)
        else:
            print(f"\nArm couldn't reach Z={z:.3f}m (actual={actual_z:.3f}m)")
            break
except KeyboardInterrupt:
    print("\nStopped by user.")

if last_good_z is not None:
    print(f"\n✓ Last reachable Z: {last_good_z:.3f} m")
    print(f"  Recommended Z_MAX: {last_good_z - 0.02:.3f} m  (2cm safety margin)")

print("\nLifting to safe Z...")
safe_z = last_good_z if last_good_z else Z_START
robot.set_speed_percent(15)
robot.move_p([X, Y, safe_z, ROLL, PITCH, YAW])
wait_and_check(robot, safe_z)

print("Returning to home...")
robot.set_speed_percent(30)
robot.move_j(HOME_JOINTS)
wait_motion_done(robot)
time.sleep(0.5)

while not robot.disable():
    time.sleep(0.01)
print("Done.")
