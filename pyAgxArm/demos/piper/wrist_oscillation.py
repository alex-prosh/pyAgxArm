"""Piper arm oscillation demo.

Oscillates the arm between two poses using the exact move_j pattern
from the official API docs.

Usage:
    1. Activate CAN: bash pyAgxArm/scripts/can_activate.sh can0
    2. Run: python pyAgxArm/demos/piper/wrist_oscillation.py
    3. Ctrl+C to stop — the arm will disable gracefully.
"""

import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

# Two distinct poses (from API docs / test1.py)
POSE_A = [0.0, 0.4, -0.4, 0.0, -0.4, 0.0]
POSE_B = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # zero / home
CYCLES = 3


def wait_motion_done(robot, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
    """Wait until `robot.get_arm_status().msg.motion_status == 0` or timeout."""
    time.sleep(0.5)
    start_t = time.monotonic()
    while True:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            print("motion done")
            return True
        if time.monotonic() - start_t > timeout:
            print(f"wait motion done timeout ({timeout:.1f}s)")
            return False
        time.sleep(poll_interval)


# --- Connect (matches API docs exactly) ---
robot_cfg = create_agx_arm_config(robot="piper", comm="can", channel="can0", interface="socketcan")
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()

# --- Set follower (slave) mode so arm responds to motion commands ---
robot.set_follower_mode()
time.sleep(0.1)

# --- Enable & configure ---
while not robot.enable():
    time.sleep(0.01)
robot.set_speed_percent(100)

# --- Oscillate between two poses ---
try:
    for i in range(CYCLES):
        print(f"Cycle {i + 1}/{CYCLES}: -> POSE_A")
        robot.move_j(POSE_A)
        wait_motion_done(robot)

        print(f"Cycle {i + 1}/{CYCLES}: -> POSE_B (home)")
        robot.move_j(POSE_B)
        wait_motion_done(robot)
except KeyboardInterrupt:
    print("\nInterrupted — stopping...")

# --- Disable ---
print("Returning to zero...")
robot.move_j([0.0] * 6)
wait_motion_done(robot)

while not robot.disable():
    time.sleep(0.01)
print("Arm disabled. Done.")
