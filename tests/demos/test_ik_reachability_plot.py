import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def make_grid(lookup, fixed_axis, fixed_val, ax1_vals, ax2_vals):
    """Build 2D array for a slice. Returns array[ax2_idx, ax1_idx]."""
    arr = np.full((len(ax2_vals), len(ax1_vals)), np.nan)
    for i2, v2 in enumerate(ax2_vals):
        for i1, v1 in enumerate(ax1_vals):
            if fixed_axis == 'y':
                key = (round(v1, 4), round(fixed_val, 4), round(v2, 4))
            elif fixed_axis == 'z':
                key = (round(v1, 4), round(v2, 4), round(fixed_val, 4))
            else:
                key = (round(fixed_val, 4), round(v1, 4), round(v2, 4))
            if key in lookup:
                arr[i2, i1] = 1.0 if lookup[key] else 0.0
    return arr


def test_grid_xz_slice():
    xs = np.array([0.10, 0.15, 0.20])
    ys = np.array([0.00])
    zs = np.array([0.20, 0.25])
    lookup = {
        (0.10, 0.00, 0.20): True,
        (0.15, 0.00, 0.20): True,
        (0.20, 0.00, 0.20): False,
        (0.10, 0.00, 0.25): True,
        (0.15, 0.00, 0.25): False,
        (0.20, 0.00, 0.25): False,
    }
    mid_y = ys[len(ys) // 2]
    arr = make_grid(lookup, 'y', mid_y, xs, zs)
    assert arr.shape == (2, 3)
    assert arr[0, 0] == 1.0  # (x=0.10, z=0.20) reachable
    assert arr[0, 2] == 0.0  # (x=0.20, z=0.20) unreachable
    assert arr[1, 1] == 0.0  # (x=0.15, z=0.25) unreachable


def test_grid_untested_cells_are_nan():
    xs = np.array([0.10, 0.15])
    ys = np.array([0.00])
    zs = np.array([0.20, 0.25])
    lookup = {(0.10, 0.00, 0.20): True}   # only one point tested
    mid_y = ys[0]
    arr = make_grid(lookup, 'y', mid_y, xs, zs)
    assert arr[0, 0] == 1.0
    assert np.isnan(arr[0, 1])
    assert np.isnan(arr[1, 0])
    assert np.isnan(arr[1, 1])
