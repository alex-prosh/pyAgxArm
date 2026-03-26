"""Piper arm IK reachability map.

Sweeps a 3D Cartesian grid, moves the arm to each point, checks
reachability via proprioceptive feedback, logs to CSV, and shows a plot.

Usage:
    python pyAgxArm/demos/piper/ik_reachability_map.py
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from pyAgxArm import create_agx_arm_config, AgxArmFactory

# ---------------------------------------------------------------------------
# Configuration — edit these before each run
# ---------------------------------------------------------------------------
X_MIN, X_MAX, X_STEP = 0.05, 0.35, 0.05   # meters
Y_MIN, Y_MAX, Y_STEP = -0.25, 0.25, 0.05
Z_MIN, Z_MAX, Z_STEP = 0.20, 0.40, 0.05
ROLL, PITCH, YAW     = math.pi, 0.0, 0.0  # flange pointing down
TOL      = 0.03   # metres — counts as "reached"
SPEED    = 15     # % speed
LOG_FILE = "reachability.csv"

PRE_SWEEP_JOINTS = [0.0, 0.5, -1.0, 0.0, 0.5, math.pi]
HOME_JOINTS = [
    math.radians(-0.03),
    math.radians(-0.79),
    math.radians(+1.38),
    math.radians(+6.80),
    math.radians(+24.89),
    math.radians(+19.52),
]

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
xs = np.round(np.arange(X_MIN, X_MAX + X_STEP / 2, X_STEP), 4)
ys = np.round(np.arange(Y_MIN, Y_MAX + Y_STEP / 2, Y_STEP), 4)
zs = np.round(np.arange(Z_MIN, Z_MAX + Z_STEP / 2, Z_STEP), 4)
