"""Piper gripper calibration script.

Procedure:
    1. Print full gripper status (width, force, driver flags, teaching params)
    2. Enable motor, open gripper fully to verify motion is working
    3. Disable motor so fingers can be moved freely by hand
    4. Physically squeeze fingers fully closed by hand, then calibrate zero
    5. Re-enable motor with move_gripper(0.0) and verify width reads ~0
    6. Test open (70 mm) with higher force to confirm range

Usage:
    python pyAgxArm/demos/piper/calibrate_gripper.py
"""

import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory

robot_cfg = create_agx_arm_config(
    robot="piper", comm="can", channel="PCAN_USBBUS1", interface="pcan"
)
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
robot.set_follower_mode()
time.sleep(0.1)

while not robot.enable():
    time.sleep(0.01)

gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
time.sleep(0.5)


def print_status(label=""):
    status = gripper.get_gripper_status()
    if status:
        foc = status.msg.foc_status
        print(f"  {'[' + label + '] ' if label else ''}width={status.msg.width*1000:.1f}mm  "
              f"force={status.msg.force:.2f}N  "
              f"enabled={foc.driver_enable_status}  "
              f"homing={foc.homing_status}  "
              f"sensor_ok={foc.sensor_status}  "
              f"error={foc.driver_error_status}")
    else:
        print(f"  [{label}] Warning: no gripper status received")


# --- Step 1: full diagnostic readout ---
print("=== Gripper Calibration ===\n")
print("Initial gripper status:")
print_status("initial")

param = gripper.get_gripper_teaching_pendant_param()
if param:
    print(f"  Teaching params: max_range={param.msg.max_range_config*1000:.0f}mm  "
          f"range%={param.msg.teaching_range_per}  "
          f"friction={param.msg.teaching_friction}")
else:
    print("  Teaching params: not available (firmware may not support this query)")
print()

# --- Step 2: verify motor responds by opening gripper ---
input(
    "Step 1: Press Enter to send OPEN command (70 mm, force=2.0 N).\n"
    "  Watch the gripper — if the fingers spread apart, the motor is working.\n"
    "  If nothing happens, the motor may be stuck or needs re-power.\n"
    "> "
)
gripper.move_gripper(0.07, force=2.0)
time.sleep(2.5)
print_status("after open")
print()

# --- Step 3: disable motor for hand movement ---
input(
    "Step 2: Press Enter to DISABLE the gripper motor.\n"
    "  After this you can freely move the fingers by hand.\n"
    "  You will use this to move them fully closed for zero calibration.\n"
    "> "
)
gripper.disable_gripper()
time.sleep(0.3)
print_status("after disable")
print()

# --- Step 4: calibrate zero with fingers physically closed ---
input(
    "Step 3: SQUEEZE THE FINGERS FULLY CLOSED BY HAND right now.\n"
    "  Hold them closed firmly, then press Enter.\n"
    "  The script will record this position as width = 0 mm.\n"
    "> "
)
ok = gripper.calibrate_gripper()
time.sleep(0.5)
if ok:
    print("  calibrate_gripper: ACK received — zero set successfully.")
else:
    print("  calibrate_gripper: no ACK (firmware may still have set zero — continuing).")
print_status("after calibrate")
print()

# --- Step 5: re-enable by sending a move command to width=0 ---
input(
    "Step 4: Press Enter to RE-ENABLE the motor.\n"
    "  Sends move_gripper(0.0) which re-enables the driver and holds closed.\n"
    "> "
)
gripper.move_gripper(0.0, force=1.0)
time.sleep(1.0)
print_status("after re-enable")
print()

# --- Step 6: test open ---
input(
    "Step 5: Press Enter to test OPEN to 70 mm.\n"
    "  Watch the fingers — they should spread apart to about 70 mm.\n"
    "> "
)
gripper.move_gripper(0.07, force=2.0)
time.sleep(2.5)
print_status("after open test")
print()

# --- Step 7: test close ---
input(
    "Step 6: Press Enter to test CLOSE to 0 mm.\n"
    "  Watch the fingers — they should close together.\n"
    "> "
)
gripper.move_gripper(0.0, force=1.0)
time.sleep(2.5)
print_status("after close test")

print("\nCalibration complete.")
while not robot.disable():
    time.sleep(0.01)
print("Arm disabled.")
