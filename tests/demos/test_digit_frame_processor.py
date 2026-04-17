# tests/demos/test_digit_frame_processor.py
import numpy as np
import pytest

from pyAgxArm.demos.piper.digit.tactile_heatmap import FrameProcessor


def _gray_frame(h=480, w=640, val=100):
    """Return a BGR frame where all pixels have equal R=G=B=val."""
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_baseline_not_ready_initially():
    fp = FrameProcessor()
    assert not fp.baseline_ready


def test_baseline_ready_after_n_frames():
    fp = FrameProcessor(baseline_frames=5)
    for _ in range(5):
        done = fp.add_baseline_frame(_gray_frame(val=128))
    assert done is True
    assert fp.baseline_ready


def test_baseline_not_ready_before_n_frames():
    fp = FrameProcessor(baseline_frames=5)
    for _ in range(4):
        fp.add_baseline_frame(_gray_frame(val=128))
    assert not fp.baseline_ready


def test_reset_baseline_clears_state():
    fp = FrameProcessor(baseline_frames=5)
    for _ in range(5):
        fp.add_baseline_frame(_gray_frame(val=128))
    assert fp.baseline_ready
    fp.reset_baseline()
    assert not fp.baseline_ready


def test_compute_heatmap_zero_diff_is_dark():
    """A frame identical to baseline should produce a near-black heatmap (low diff)."""
    fp = FrameProcessor(baseline_frames=3, blur=False)
    frame = _gray_frame(val=100)
    for _ in range(3):
        fp.add_baseline_frame(frame)
    heatmap = fp.compute_heatmap(frame)
    assert heatmap.shape == frame.shape
    assert heatmap.mean() < 10  # near zero diff → near-black in COLORMAP_HOT


def test_compute_heatmap_large_diff_is_bright():
    """A frame very different from baseline should produce a bright heatmap."""
    fp = FrameProcessor(baseline_frames=3, blur=False)
    for _ in range(3):
        fp.add_baseline_frame(_gray_frame(val=0))
    bright_frame = _gray_frame(val=255)
    heatmap = fp.compute_heatmap(bright_frame)
    assert heatmap.mean() > 100  # large diff → bright heatmap


def test_heatmap_shape_matches_input():
    fp = FrameProcessor(baseline_frames=2, blur=False)
    frame = _gray_frame(h=240, w=320, val=50)
    for _ in range(2):
        fp.add_baseline_frame(frame)
    heatmap = fp.compute_heatmap(frame)
    assert heatmap.shape == (240, 320, 3)


def test_toggle_blur_changes_state():
    fp = FrameProcessor(blur=True)
    assert fp.blur_enabled is True
    state = fp.toggle_blur()
    assert state is False
    assert fp.blur_enabled is False
    state = fp.toggle_blur()
    assert state is True


def test_blur_on_vs_off_differs():
    """Heatmap with blur enabled should differ from blur disabled for a noisy diff."""
    fp_blur = FrameProcessor(baseline_frames=2, blur=True)
    fp_no = FrameProcessor(baseline_frames=2, blur=False)
    baseline = _gray_frame(val=80)
    noisy = _gray_frame(val=80)
    # Inject noise
    rng = np.random.default_rng(42)
    noisy[:, :, :] = rng.integers(0, 255, noisy.shape, dtype=np.uint8)
    for _ in range(2):
        fp_blur.add_baseline_frame(baseline)
        fp_no.add_baseline_frame(baseline)
    h_blur = fp_blur.compute_heatmap(noisy).astype(float)
    h_no = fp_no.compute_heatmap(noisy).astype(float)
    assert not np.allclose(h_blur, h_no)


def test_compute_heatmap_raises_before_baseline_ready():
    """Calling compute_heatmap before baseline is captured should raise."""
    fp = FrameProcessor(baseline_frames=5)
    with pytest.raises((TypeError, ValueError)):
        fp.compute_heatmap(_gray_frame(val=100))
