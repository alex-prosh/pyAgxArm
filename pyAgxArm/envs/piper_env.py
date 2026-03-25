"""Gymnasium environment wrapping the physical Piper arm.

Provides a standard ``gymnasium.Env`` interface matching ``tactile_envs``
(``ArmEnv`` / ``InsertionEnv``) so that policies trained in simulation can be
deployed on the real arm with no code changes at the callsite.

Usage::

    import gymnasium as gym
    import pyAgxArm.envs  # triggers registration

    env = gym.make("pyAgxArm/Piper-v0", speed_percent=25)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0.1)
    env.close()
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pyAgxArm import AgxArmFactory, create_agx_arm_config
from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Home joint pose (radians) — arm's natural resting position
HOME_JOINTS = [
    math.radians(-0.03),   # J1
    math.radians(-0.79),   # J2
    math.radians(+1.38),   # J3
    math.radians(+6.80),   # J4
    math.radians(+24.89),  # J5
    math.radians(+19.52),  # J6
]

# Piper joint limits (radians) from constants.py
_PIPER_JOINT_LIMITS = ROBOT_JOINT_LIMIT_PRESET_RAD["piper"]
JOINT_LOW = np.array([v[0] for v in _PIPER_JOINT_LIMITS.values()], dtype=np.float64)
JOINT_HIGH = np.array([v[1] for v in _PIPER_JOINT_LIMITS.values()], dtype=np.float64)

# Fixed orientation when no_rotation=True
FIXED_ROLL = math.pi
FIXED_PITCH = 0.0
FIXED_YAW = 0.0


def _wait_motion_done(
    robot, timeout: float = 15.0, poll_interval: float = 0.1
) -> bool:
    """Block until the arm reports motion_status == 0, or timeout."""
    time.sleep(0.5)
    t0 = time.monotonic()
    while True:
        status = robot.get_arm_status()
        if status is not None and getattr(status.msg, "motion_status", None) == 0:
            return True
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(poll_interval)


def convert_observation_to_space(observation):
    """Build a Dict observation space from a sample observation (matches tactile_envs)."""
    space = spaces.Dict(spaces={})
    for key, val in observation.items():
        if key == "image":
            space.spaces[key] = spaces.Box(
                low=0, high=255, shape=val.shape, dtype=np.uint8,
            )
        elif key in ("tactile", "state"):
            space.spaces[key] = spaces.Box(
                low=-float("inf"), high=float("inf"),
                shape=val.shape, dtype=np.float64,
            )
    return space


class PiperEnv(gym.Env):
    """Gymnasium wrapper for the physical Piper arm.

    The action / observation interface mirrors ``tactile_envs.envs.arm.ArmEnv``
    so that trained policies transfer directly.

    Parameters
    ----------
    action_mode : str
        ``'cartesian'`` (move_p / move_l) or ``'joint'`` (move_j).
    motion_type : str
        ``'p'`` (point-to-point) or ``'l'`` (linear).  Only used when
        ``action_mode='cartesian'``.
    no_rotation : bool
        If True the wrist yaw (or full orientation) DOF is removed from the
        action space and orientation is fixed at (roll=pi, pitch=0, yaw=0).
    no_gripping : bool
        If True the gripper DOF is removed from the action space.
    full_rotation : bool
        If True **and** ``no_rotation=False``, expose 3 rotation DOFs
        (roll, pitch, yaw) instead of only yaw.  Matches ``ArmEnv``
        ``full_rotation`` flag.
    grip_range : float
        Fraction of the gripper range to use as minimum opening.
        ``0.0`` → full range ``[0, 0.035]``.  Matches ``ArmEnv.grip_range``.
    state_type : str
        ``'privileged'`` returns ``{'state': ...}``; future options may add
        ``'vision'``, ``'touch'``, ``'vision_and_touch'``.
    max_delta : float | None
        If set, clamp the Cartesian position change between consecutive steps.
    step_dt : float | None
        Fixed step duration in seconds.  ``0.05`` gives ~20 Hz continuous
        control (the arm chases each new target without waiting to arrive).
        ``None`` falls back to blocking until motion is done.
    speed_percent : int
        Arm speed 0–100.
    channel : str
        CAN channel (e.g. ``'can0'``).
    env_id : int
        Identifier carried in ``info['id']`` (matches ``ArmEnv``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        action_mode: str = "cartesian",
        motion_type: str = "p",
        no_rotation: bool = True,
        no_gripping: bool = True,
        full_rotation: bool = False,
        grip_range: float = 0.0,
        state_type: str = "privileged",
        max_delta: Optional[float] = None,
        step_dt: Optional[float] = 0.05,
        speed_percent: int = 25,
        channel: str = "can0",
        env_id: int = -1,
        **kwargs: Any,
    ):
        super().__init__()

        assert action_mode in ("cartesian", "joint")
        assert motion_type in ("p", "l")

        self._action_mode = action_mode
        self._motion_type = motion_type
        self.with_rotation = not no_rotation
        self.adaptive_gripping = not no_gripping
        self.full_rotation = full_rotation
        self.max_delta = max_delta
        self._step_dt = step_dt
        self.state_type = state_type
        self._speed_percent = speed_percent
        self._channel = channel
        self.id = env_id

        # --- Action scale (matches ArmEnv / InsertionEnv layout) -----------
        # Each row is [low, high].  We build the full array then mask out
        # disabled DOFs, exactly as tactile_envs does.
        grip_min = grip_range * (0.035 - 0.003) + 0.003

        if action_mode == "cartesian":
            if full_rotation:
                # x, y, z, roll, pitch, yaw, gripper
                self.action_scale = np.array([
                    [0.08,     0.25],    # x
                    [-0.25,    0.25],    # y
                    [0.075,    0.20],    # z
                    [-np.pi,   np.pi],   # roll
                    [-np.pi,   np.pi],   # pitch
                    [-np.pi,   np.pi],   # yaw
                    [grip_min, 0.035],   # gripper
                ])
                self.action_mask = np.ones(7, dtype=bool)
                if no_rotation:
                    self.action_mask[3] = False
                    self.action_mask[4] = False
                    self.action_mask[5] = False
                if no_gripping:
                    self.action_mask[6] = False
            else:
                # x, y, z, yaw, gripper
                self.action_scale = np.array([
                    [0.08,     0.25],    # x
                    [-0.25,    0.25],    # y
                    [0.075,    0.20],    # z
                    [-np.pi,   np.pi],   # yaw
                    [grip_min, 0.035],   # gripper
                ])
                self.action_mask = np.ones(5, dtype=bool)
                if no_rotation:
                    self.action_mask[3] = False
                if no_gripping:
                    self.action_mask[4] = False

            self.action_scale = self.action_scale[self.action_mask]
        else:
            # Joint mode: 6 joints (+ gripper)
            joint_scale = np.stack([JOINT_LOW, JOINT_HIGH], axis=1)
            grip_row = np.array([[grip_min, 0.035]])
            if no_gripping:
                self.action_scale = joint_scale
            else:
                self.action_scale = np.vstack([joint_scale, grip_row])
            self.action_mask = np.ones(len(self.action_scale), dtype=bool)

        self.ndof_u = int(self.action_mask.sum()) if action_mode == "cartesian" else len(self.action_scale)
        self.action_space = spaces.Box(
            low=np.full(self.ndof_u, -1.0),
            high=np.full(self.ndof_u, 1.0),
            dtype=np.float32,
        )

        # --- Observation space (built from a sample, like tactile_envs) ----
        self.curr_obs = self._build_zero_obs()
        self.observation_space = convert_observation_to_space(self.curr_obs)

        # --- Connect to robot ---------------------------------------------
        cfg = create_agx_arm_config(
            robot="piper", comm="can", channel=channel, interface="socketcan",
        )
        self._robot = AgxArmFactory.create_arm(cfg)
        self._robot.connect()

        # Must call set_follower_mode before enable (see memory)
        self._robot.set_follower_mode()
        time.sleep(0.1)

        while not self._robot.enable():
            time.sleep(0.01)
        self._robot.set_speed_percent(speed_percent)

        self._step_count = 0
        self.prev_action_xyz: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        # Return to home
        self._robot.move_j(HOME_JOINTS)
        _wait_motion_done(self._robot)

        self._step_count = 0
        self.prev_action_xyz = None

        obs = self._read_obs()
        info = {
            "id": np.array([self.id]),
            "is_success": int(False),
        }
        return self._get_obs(), info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Unnormalize: [-1, 1] -> [low, high]  (same formula as tactile_envs)
        action_unnorm = (
            (action + 1.0) / 2.0
            * (self.action_scale[:, 1] - self.action_scale[:, 0])
            + self.action_scale[:, 0]
        )

        if self._action_mode == "cartesian":
            # Optionally clamp position delta
            if self.max_delta is not None and self.prev_action_xyz is not None:
                action_unnorm[:3] = np.clip(
                    action_unnorm[:3],
                    self.prev_action_xyz - self.max_delta,
                    self.prev_action_xyz + self.max_delta,
                )
            self.prev_action_xyz = action_unnorm[:3].copy()

            x, y, z = float(action_unnorm[0]), float(action_unnorm[1]), float(action_unnorm[2])
            idx = 3

            if self.full_rotation and self.with_rotation:
                roll = float(action_unnorm[idx]); idx += 1
                pitch = float(action_unnorm[idx]); idx += 1
                yaw = float(action_unnorm[idx]); idx += 1
            elif self.with_rotation:
                roll, pitch = FIXED_ROLL, FIXED_PITCH
                yaw = float(action_unnorm[idx]); idx += 1
            else:
                roll, pitch, yaw = FIXED_ROLL, FIXED_PITCH, FIXED_YAW

            pose = [x, y, z, roll, pitch, yaw]

            if self._motion_type == "l":
                self._robot.move_l(pose)
            else:
                self._robot.move_p(pose)
        else:
            # Joint mode
            joints = [float(v) for v in action_unnorm[:6]]
            self._robot.move_j(joints)

        if self._step_dt is not None:
            time.sleep(self._step_dt)
        else:
            _wait_motion_done(self._robot)

        self._step_count += 1
        obs = self._read_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "id": np.array([self.id]),
            "is_success": int(False),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        try:
            self._robot.move_j(HOME_JOINTS)
            _wait_motion_done(self._robot)
            time.sleep(0.5)
            while not self._robot.disable():
                time.sleep(0.01)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_zero_obs(self) -> Dict[str, np.ndarray]:
        """Return a zero-filled observation dict matching ``state_type``."""
        if self.state_type == "privileged":
            # joint_angles(6) + ee_pose(6) + gripper_width(1) = 13
            return {"state": np.zeros(13, dtype=np.float64)}
        else:
            raise ValueError(f"Unsupported state_type: {self.state_type!r}")

    def _read_obs(self) -> Dict[str, np.ndarray]:
        """Read sensors and update ``self.curr_obs`` in place."""
        if self.state_type == "privileged":
            # Joint angles
            ja_msg = self._robot.get_joint_angles()
            if ja_msg is not None:
                joint_angles = np.array(ja_msg.msg, dtype=np.float64)
            else:
                joint_angles = np.zeros(6, dtype=np.float64)

            # End-effector pose
            fp_msg = self._robot.get_flange_pose()
            if fp_msg is not None:
                ee_pose = np.array(fp_msg.msg, dtype=np.float64)
            else:
                ee_pose = np.zeros(6, dtype=np.float64)

            # Gripper width — placeholder (no gripper feedback on piper CAN driver)
            gripper_width = np.zeros(1, dtype=np.float64)

            self.curr_obs = {
                "state": np.concatenate([joint_angles, ee_pose, gripper_width]),
            }
        return self.curr_obs

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Return the current observation (matches tactile_envs pattern)."""
        return self.curr_obs
