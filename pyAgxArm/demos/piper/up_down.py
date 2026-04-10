"""Piper arm up-down oscillation demo.

Moves the arm between a raised pose and home using move_j.

Usage:
    1. Run: python pyAgxArm/demos/piper/up_down.py
    2. Ctrl+C to stop — the arm will disable gracefully.
"""

import math
import threading
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory

ZERO   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]            # all-zeros between movements
HOME   = [0.0, 0.0, 0.0, 0.001, 0.575, 0.0]        # neutral starting pose
HOME_JOINTS = [                                      # safe resting pose for RTH
    math.radians(-0.03),
    math.radians(-0.79),
    math.radians(+1.38),
    math.radians(+6.80),
    math.radians(+24.89),
    math.radians(+19.52),
]
POSE_A = [0.0, 0.4, -0.4, 0.0, -0.4, 0.0]          # raised position
CYCLES = 2

_stop_event = threading.Event()
_track_thread = None
_log_file = "up_down_log.csv"


def start_tracking(robot, label: str = "", interval: float = 0.2):
    global _track_thread
    _stop_event.clear()
    def _loop():
        with open(_log_file, "a") as f:
            while not _stop_event.is_set():
                msg = robot.get_joint_angles()
                if msg is not None:
                    deg = [round(math.degrees(r), 1) for r in msg.msg]
                    f.write(f"{time.time():.3f},{label},{','.join(map(str, deg))}\n")
                    f.flush()
                time.sleep(interval)
    _track_thread = threading.Thread(target=_loop, daemon=True)
    _track_thread.start()


def stop_tracking():
    global _track_thread
    _stop_event.set()
    if _track_thread is not None:
        _track_thread.join()
        _track_thread = None


def wait_motion_done(robot, timeout: float = 30.0, poll_interval: float = 0.1) -> bool:
    # Phase 1: wait for motion to start
    time.sleep(0.3)
    for _ in range(30):
        s = robot.get_arm_status()
        if s and getattr(s.msg, "motion_status", 0) != 0:
            break
        time.sleep(0.05)
    # Phase 2: wait for motion to finish
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


robot_cfg = create_agx_arm_config(
    robot="piper", comm="can", channel="PCAN_USBBUS1", interface="pcan",
    joint_limits={"joint5": [-0.5, 0.6]},
)
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

robot.set_follower_mode()
time.sleep(0.1)

while not robot.enable():
    time.sleep(0.01)
robot.set_speed_percent(30)

print("Moving to home...")
print(f"Logging to {_log_file}")
start_tracking(robot, label="home")
robot.move_j(HOME)
wait_motion_done(robot)
stop_tracking()

try:
    for i in range(CYCLES):
        print(f"Cycle {i + 1}/{CYCLES}: -> up (POSE_A), open gripper")
        start_tracking(robot, label=f"c{i+1}_up")
        robot.move_j(POSE_A)
        end_effector.move_gripper(0.07)
        wait_motion_done(robot)
        stop_tracking()

        print(f"Cycle {i + 1}/{CYCLES}: -> zero, close gripper")
        start_tracking(robot, label=f"c{i+1}_zero")
        robot.move_j(ZERO)
        end_effector.move_gripper(0)
        wait_motion_done(robot)
        stop_tracking()
except KeyboardInterrupt:
    stop_tracking()
    print("\nInterrupted — stopping...")

print("Returning to home...")
start_tracking(robot, label="return_home")
robot.move_j(HOME_JOINTS)
wait_motion_done(robot)
stop_tracking()

while not robot.disable():
    time.sleep(0.01)
print("Arm disabled. Done.")
