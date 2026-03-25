from gymnasium.envs.registration import register

register(
    id="pyAgxArm/Piper-v0",
    entry_point="pyAgxArm.envs.piper_env:PiperEnv",
)
