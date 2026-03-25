"""Test PiperEnv with random actions, point targets, or trajectory replay.

Usage:
    1. Activate CAN: sudo bash pyAgxArm/scripts/ubuntu/can_activate.sh can0
    2. Run:
       # Random policy
       python pyAgxArm/demos/piper/test_env.py --random

       # Move to a target position (3 DOF)
       python pyAgxArm/demos/piper/test_env.py --test-point --target-x 0.15 --target-z 0.12

       # Hold at target indefinitely
       python pyAgxArm/demos/piper/test_env.py --test-point --target-x 0.15 --hold

       # Replay a trajectory (auto-detects DOF from ACTIONS list)
       python pyAgxArm/demos/piper/test_env.py --test-trajectory path/to/trajectory.py

    3. Ctrl+C to stop — the arm will return home and disable gracefully.
"""

import argparse
import importlib.util
import time

import gymnasium as gym
import numpy as np

import pyAgxArm.envs  # triggers registration


class RandomPolicy:
    """Random policy for testing."""
    def __init__(self, action_space):
        self.action_space = action_space

    def sample_actions(self, obs, temperature=0.0):
        return self.action_space.sample()


def pos_to_action(pos, scale):
    """Convert a position in meters to normalized [-1, 1] action.

    Parameters
    ----------
    pos : float
        Position in meters.
    scale : array-like, shape (2,)
        [low, high] bounds for this DOF.
    """
    return 2 * (pos - scale[0]) / (scale[1] - scale[0]) - 1


def load_trajectory(filepath):
    """Dynamically import a trajectory file and return its ACTIONS list."""
    spec = importlib.util.spec_from_file_location("trajectory_module", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "ACTIONS"):
        raise ValueError(f"Trajectory file {filepath} must define an ACTIONS list")
    return [np.array(a) for a in mod.ACTIONS]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PiperEnv")
    # Mode selection
    parser.add_argument("--random", action="store_true", help="Use random policy (default)")
    parser.add_argument("--test-point", action="store_true", help="Move to specified target position")
    parser.add_argument("--test-trajectory", type=str, default=None, metavar="FILE",
                        help="Replay action list from Python file (expects ACTIONS list)")

    # Shared args
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--speed", type=int, default=25, help="Arm speed percent 0-100")
    parser.add_argument("--scale", type=float, default=0.1,
                        help="Scale random actions by this factor (0.1 = small motions)")
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel")
    parser.add_argument("--action-mode", type=str, default="cartesian",
                        choices=["cartesian", "joint"], help="Action mode")
    parser.add_argument("--max-delta", type=float, default=None,
                        help="Max position change per step (meters)")
    parser.add_argument("--step-dt", type=float, default=0.05,
                        help="Fixed step duration in seconds (0.05 = 20Hz). Use 0 to wait for motion done.")

    # Point target args
    parser.add_argument("--target-x", type=float, default=0.15, help="Target x position (meters)")
    parser.add_argument("--target-y", type=float, default=0.0, help="Target y position (meters)")
    parser.add_argument("--target-z", type=float, default=0.12, help="Target z position (meters)")
    parser.add_argument("--hold", action="store_true",
                        help="Stay at target indefinitely (Ctrl+C to stop)")

    args = parser.parse_args()

    step_dt = args.step_dt if args.step_dt > 0 else None

    # ------------------------------------------------------------------
    # Determine env kwargs based on mode
    # ------------------------------------------------------------------
    env_kwargs = dict(
        speed_percent=args.speed,
        channel=args.channel,
        action_mode=args.action_mode,
        max_delta=args.max_delta,
        step_dt=step_dt,
    )

    if args.test_trajectory:
        raw_actions = load_trajectory(args.test_trajectory)
        raw_ndof = len(raw_actions[0])
        # Keep x, y, z, yaw (4 DOF). Drop gripper if present.
        actions = [a[:4] for a in raw_actions]
        env_kwargs["no_rotation"] = False
        env_kwargs["no_gripping"] = True
        env_kwargs["step_dt"] = 0.02  # 50 Hz
        print(f"\n=== TRAJECTORY REPLAY MODE ===")
        print(f"Loaded {len(actions)} actions from {args.test_trajectory}")
        print(f"Original {raw_ndof} DOFs → truncated to 4 (x, y, z, yaw)")
        print(f"Step dt: 0.02s (50 Hz)")
        env_kwargs["max_episode_steps"] = len(actions)
    elif args.test_point:
        # Point mode always uses 3 DOF (default: no_rotation=True, no_gripping=True)
        env_kwargs["max_episode_steps"] = args.steps
    else:
        # Random mode
        env_kwargs["max_episode_steps"] = args.steps

    env = gym.make("pyAgxArm/Piper-v0", **env_kwargs)

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action scale:\n{env.unwrapped.action_scale}")

    try:
        # ==============================================================
        # --test-point mode
        # ==============================================================
        if args.test_point:
            print(f"\n=== POINT TARGET MODE ===")
            print(f"Target: x={args.target_x}, y={args.target_y}, z={args.target_z}")

            obs, info = env.reset(seed=0)
            action_scale = env.unwrapped.action_scale  # shape (ndof, 2)

            target_action = np.array([
                np.clip(pos_to_action(args.target_x, action_scale[0]), -1, 1),
                np.clip(pos_to_action(args.target_y, action_scale[1]), -1, 1),
                np.clip(pos_to_action(args.target_z, action_scale[2]), -1, 1),
            ])
            print(f"Normalized action: {target_action}")

            ee_pos = obs["state"][6:9]
            print(f"Initial EE: [{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]")

            step_idx = 0
            if args.hold:
                print("HOLD mode — looping until Ctrl+C")
                while True:
                    obs, reward, terminated, truncated, info = env.step(target_action)
                    ee_pos = obs["state"][6:9]
                    print(
                        f"  step {step_idx:4d}: "
                        f"ee=[{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]"
                    )
                    step_idx += 1
            else:
                for step_idx in range(args.steps):
                    obs, reward, terminated, truncated, info = env.step(target_action)
                    ee_pos = obs["state"][6:9]
                    print(
                        f"  step {step_idx:3d}: "
                        f"ee=[{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]"
                    )
                    if terminated or truncated:
                        break

        # ==============================================================
        # --test-trajectory mode
        # ==============================================================
        elif args.test_trajectory:
            obs, info = env.reset(seed=0)
            ee_pos = obs["state"][6:9]
            print(f"Initial EE: [{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]")

            tic = time.time()
            for i, action in enumerate(actions):
                obs, reward, terminated, truncated, info = env.step(action)
                ee_pos = obs["state"][6:9]
                print(
                    f"  step {i:4d}/{len(actions)}: "
                    f"ee=[{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}]  "
                    f"action={np.array2string(action, precision=3, suppress_small=True)}"
                )
                if terminated or truncated:
                    print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
                    break

            elapsed = time.time() - tic
            n = i + 1
            print(f"\nReplayed {n} steps in {elapsed:.1f}s ({n / elapsed:.2f} Hz)")

        # ==============================================================
        # Random policy (default)
        # ==============================================================
        else:
            policy = RandomPolicy(env.action_space)

            for ep in range(args.episodes):
                seed = np.random.randint(0, 1000)
                print(f"\n=== Episode {ep} (seed={seed}) ===")
                obs, info = env.reset(seed=seed)
                print(f"Reset obs['state']: {obs['state']}")

                tic = time.time()
                for i in range(args.steps):
                    action = policy.sample_actions(obs) * args.scale
                    obs, reward, terminated, truncated, info = env.step(action)

                    ee_pos = obs["state"][6:9]
                    joints_deg = np.degrees(obs["state"][:6])
                    print(
                        f"  step {i:3d}: "
                        f"ee=[{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}] "
                        f"J1={joints_deg[0]:+6.1f} J2={joints_deg[1]:+6.1f} J3={joints_deg[2]:+6.1f}"
                    )

                    if terminated or truncated:
                        print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
                        break

                elapsed = time.time() - tic
                print(f"  {i + 1} steps in {elapsed:.1f}s ({(i + 1) / elapsed:.2f} Hz)")

    except KeyboardInterrupt:
        print("\nInterrupted — stopping...")

    env.close()
    print("Done.")
