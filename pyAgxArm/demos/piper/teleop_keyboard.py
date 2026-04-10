"""Piper arm keyboard teleoperation.

Controls:
    W / S       +X / -X  (forward / back)
    A / D       +Y / -Y  (left / right)
    R / F       +Z / -Z  (up / down)
    Q / E       +Yaw / -Yaw
    [ / ]       open / close gripper
    ESC         quit (returns home and disables)

Usage:
    python pyAgxArm/demos/piper/teleop_keyboard.py
"""

import math
import time

from pynput import keyboard

from pyAgxArm import create_agx_arm_config, AgxArmFactory

# ---------------------------------------------------------------------------
# Workspace limits — keep in sync with piper_env.py
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.10, 0.30
Y_MIN, Y_MAX = -0.20, 0.20
Z_MIN         = 0.15          # no upper Z limit
YAW_MIN, YAW_MAX = -math.pi / 2, math.pi / 2   # ±90° around 0, well within firmware [-π, π]
GRIP_MIN, GRIP_MAX = 0.0, 0.07  # metres

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
POS_STEP  = 0.005   # metres per tick
YAW_STEP  = 0.05    # radians per tick
GRIP_STEP    = 0.01   # metres per key press
GRIP_FORCE   = 0.3    # Newtons (range 0.0–3.0)
GRIP_COOLDOWN = 0.8   # seconds to wait after sending before accepting next command
TICK_DT       = 0.05    # seconds per tick (20 Hz)
SPEED         = 25      # arm speed %

FIXED_ROLL  = math.pi
FIXED_PITCH = 0.0

HOME_JOINTS = [
    math.radians(-0.03),
    math.radians(-0.79),
    math.radians(+1.38),
    math.radians(+6.80),
    math.radians(+24.89),
    math.radians(+19.52),
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
# [x, y, z, yaw, gripper]
pos = [
    (X_MIN + X_MAX) / 2,
    (Y_MIN + Y_MAX) / 2,
    Z_MIN + 0.05,            # start just above the floor
    0.0,                     # neutral yaw
    (GRIP_MIN + GRIP_MAX) / 2,
]

pressed = set()
running = True
grip_seq = 0
last_grip_t = 0.0


def on_press(key):
    global running, grip_seq, last_grip_t
    try:
        c = key.char.lower()
        if c == '[':
            if '[' not in pressed and time.monotonic() - last_grip_t >= GRIP_COOLDOWN:
                pos[4] = min(pos[4] + GRIP_STEP, GRIP_MAX)
                gripper.move_gripper(pos[4], force=GRIP_FORCE)
                last_grip_t = time.monotonic()
                grip_seq += 1
                print(f"\n  [{grip_seq}] OPEN  → {pos[4]*1000:.1f}mm")
            pressed.add('[')
        elif c == ']':
            if ']' not in pressed and time.monotonic() - last_grip_t >= GRIP_COOLDOWN:
                pos[4] = max(pos[4] - GRIP_STEP, GRIP_MIN)
                gripper.move_gripper(pos[4], force=GRIP_FORCE)
                last_grip_t = time.monotonic()
                grip_seq += 1
                print(f"\n  [{grip_seq}] CLOSE → {pos[4]*1000:.1f}mm")
            pressed.add(']')
        else:
            pressed.add(c)
    except AttributeError:
        if key == keyboard.Key.esc:
            running = False
        else:
            pressed.add(key)


def on_release(key):
    try:
        pressed.discard(key.char)
        pressed.discard(key.char.lower())
    except AttributeError:
        pressed.discard(key)


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

gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

print("Moving to start position...")
robot.move_p([pos[0], pos[1], pos[2], FIXED_ROLL, FIXED_PITCH, math.pi])
gripper.move_gripper(pos[4], force=GRIP_FORCE)
time.sleep(1.5)

print("\nControls:")
print("  W / S      +X / -X  (forward / back)")
print("  A / D      +Y / -Y  (left / right)")
print("  R / F      +Z / -Z  (up / down)")
print("  Q / E      +Yaw / -Yaw")
print("  [          open gripper")
print("  ]          close gripper")
print("  ESC        quit\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while running and listener.is_alive():
        arm_moved = False

        if 'w' in pressed: pos[0] = min(pos[0] + POS_STEP, X_MAX);   arm_moved = True
        if 's' in pressed: pos[0] = max(pos[0] - POS_STEP, X_MIN);   arm_moved = True
        if 'a' in pressed: pos[1] = min(pos[1] + POS_STEP, Y_MAX);   arm_moved = True
        if 'd' in pressed: pos[1] = max(pos[1] - POS_STEP, Y_MIN);   arm_moved = True
        if 'r' in pressed: pos[2] += POS_STEP;                        arm_moved = True
        if 'f' in pressed: pos[2] = max(pos[2] - POS_STEP, Z_MIN);   arm_moved = True
        if 'q' in pressed: pos[3] = min(pos[3] + YAW_STEP, YAW_MAX); arm_moved = True
        if 'e' in pressed: pos[3] = max(pos[3] - YAW_STEP, YAW_MIN); arm_moved = True

        if arm_moved:
            yaw_abs = math.pi + pos[3]
            if yaw_abs > math.pi:
                yaw_abs -= 2 * math.pi
            robot.move_p([pos[0], pos[1], pos[2], FIXED_ROLL, FIXED_PITCH, yaw_abs])
            print(
                f"  X={pos[0]:.3f}  Y={pos[1]:.3f}  Z={pos[2]:.3f}"
                f"  Yaw={math.degrees(pos[3]):+.1f}°  Grip={pos[4]*1000:.1f}mm",
                end='\r',
            )

        time.sleep(TICK_DT)

except KeyboardInterrupt:
    pass

# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------
print("\nReturning home...")
robot.move_j(HOME_JOINTS)
time.sleep(3.0)
while not robot.disable():
    time.sleep(0.01)
print("Done.")
