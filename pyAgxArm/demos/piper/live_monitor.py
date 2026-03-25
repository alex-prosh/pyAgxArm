"""Piper arm live monitor.

Continuously displays joint angles, end-effector pose, and arm status
in a curses-based terminal UI. Read-only — no motion commands are sent.

Usage:
    1. Activate CAN: bash pyAgxArm/scripts/can_activate.sh can0
    2. Run: python pyAgxArm/demos/piper/live_monitor.py
    3. Press 'q' to quit.
"""

import curses
import math
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory

# Status code → human-readable string
ARM_STATUS = {
    0x00: "Normal",
    0x01: "E-Stop",
    0x02: "No Solution",
    0x03: "Singularity",
    0x04: "Pos Limit",
    0x05: "Joint Comm Err",
    0x06: "Brake Locked",
    0x07: "Collision",
    0x08: "Overspeed",
    0x09: "Joint Err",
    0x0A: "Other Err",
    0x0B: "Teach Record",
    0x0C: "Teach Execute",
    0x0D: "Teach Pause",
    0x0E: "NTC Overtemp",
    0x0F: "Resistor Overtemp",
    0xFF: "Unknown",
}

CTRL_MODE = {
    0x00: "STANDBY",
    0x01: "CAN",
    0x02: "TEACHING",
    0x03: "ETHERNET",
    0x04: "WIFI",
    0x05: "REMOTE",
    0x06: "LINKAGE_TEACH",
    0x07: "OFFLINE_TRAJ",
    0xFF: "UNKNOWN",
}

MOTION_STATUS = {
    0x00: "Reached",
    0x01: "Failed",
    0xFF: "Unknown",
}

W = 46  # box width (inner)


def hline(ch_l, ch_m, ch_r):
    return ch_l + ch_m * W + ch_r


def draw(stdscr, robot):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(20)  # ~50 Hz poll

    while True:
        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break

        # --- Fetch data ---
        joints_msg = robot.get_joint_angles()
        pose_msg = robot.get_flange_pose()
        status_msg = robot.get_arm_status()

        # Joint angles
        if joints_msg is not None:
            jdeg = [math.degrees(r) for r in joints_msg.msg]
            jhz = f"{joints_msg.hz:.0f}" if joints_msg.hz else "—"
        else:
            jdeg = [float("nan")] * 6
            jhz = "—"

        # End-effector pose
        if pose_msg is not None:
            p = pose_msg.msg  # [x, y, z, roll, pitch, yaw]
            pos = p[:3]
            ori = [math.degrees(r) for r in p[3:]]
            phz = f"{pose_msg.hz:.0f}" if pose_msg.hz else "—"
        else:
            pos = [float("nan")] * 3
            ori = [float("nan")] * 3
            phz = "—"

        # Arm status
        if status_msg is not None:
            sm = status_msg.msg
            as_val = sm.arm_status.value if hasattr(sm.arm_status, 'value') else sm.arm_status
            cm_val = sm.ctrl_mode.value if hasattr(sm.ctrl_mode, 'value') else sm.ctrl_mode
            ms_val = sm.motion_status.value if hasattr(sm.motion_status, 'value') else sm.motion_status
            arm_st = ARM_STATUS.get(as_val, str(sm.arm_status))
            ctrl_m = CTRL_MODE.get(cm_val, str(sm.ctrl_mode))
            mot_st = MOTION_STATUS.get(ms_val, str(sm.motion_status))
        else:
            arm_st = ctrl_m = mot_st = "N/A"

        # --- Draw ---
        stdscr.erase()
        row = 0

        def put(text):
            nonlocal row
            try:
                stdscr.addstr(row, 0, text)
            except curses.error:
                pass
            row += 1

        def row_line(content):
            """Wrap content in box borders, padded to width W."""
            put(f"║{content:<{W}}║")

        put(hline("╔", "═", "╗"))
        row_line(f"{'Piper Arm Live Monitor':^{W}}")
        put(hline("╠", "═", "╣"))

        # Joint angles
        row_line(f"  Joint Angles (deg)        Hz: {jhz:<5}")
        for i in range(3):
            j1, j2 = i, i + 3
            row_line(f"  J{j1+1}: {jdeg[j1]:+8.2f}    J{j2+1}: {jdeg[j2]:+8.2f}")

        put(hline("╠", "═", "╣"))

        # End-effector
        labels_pos = ["X", "Y", "Z"]
        labels_ori = ["Roll", "Pitch", "Yaw"]
        row_line(f"  End Effector (flange)     Hz: {phz:<5}")
        for i in range(3):
            row_line(
                f"  {labels_pos[i]}: {pos[i]:+6.3f} m"
                f"   {labels_ori[i]+':':<7s} {ori[i]:+7.2f}\u00b0"
            )

        put(hline("╠", "═", "╣"))

        # Status
        row_line(f"  Status: {arm_st}  Mode: {ctrl_m}")
        row_line(f"  Motion: {mot_st}")

        put(hline("╚", "═", "╝"))
        put("  Press 'q' to quit")

        stdscr.refresh()


def main():
    robot_cfg = create_agx_arm_config(
        robot="piper", comm="can", channel="can0", interface="socketcan"
    )
    robot = AgxArmFactory.create_arm(robot_cfg)
    robot.connect()
    robot.set_follower_mode()
    time.sleep(0.1)

    try:
        curses.wrapper(lambda stdscr: draw(stdscr, robot))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
