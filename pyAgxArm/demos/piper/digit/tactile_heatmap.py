# pyAgxArm/demos/piper/digit/tactile_heatmap.py
import cv2
import numpy as np


class FrameProcessor:
    def __init__(self, blur: bool = True, baseline_frames: int = 30):
        self._blur = blur
        self._n = baseline_frames
        self._accum: np.ndarray | None = None
        self._count = 0
        self._baseline: np.ndarray | None = None  # float32 grayscale

    # ------------------------------------------------------------------ baseline

    def add_baseline_frame(self, frame: np.ndarray) -> bool:
        """Accumulate one frame toward baseline. Returns True when baseline is ready."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._accum is None:
            self._accum = np.zeros_like(gray)
        self._accum += gray
        self._count += 1
        if self._count >= self._n:
            self._baseline = self._accum / self._n
            return True
        return False

    def reset_baseline(self) -> None:
        self._accum = None
        self._count = 0
        self._baseline = None

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None

    # ------------------------------------------------------------------ controls

    def toggle_blur(self) -> bool:
        self._blur = not self._blur
        return self._blur

    @property
    def blur_enabled(self) -> bool:
        return self._blur

    # ------------------------------------------------------------------ compute

    def compute_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Return a BGR heatmap of contact diff. Requires baseline_ready == True."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(gray - self._baseline)
        if self._blur:
            diff = cv2.GaussianBlur(diff, (15, 15), 0)
        # Normalize to 0-255 using the full range, not MINMAX (which fails for uniform data)
        max_diff = np.max(diff)
        if max_diff > 0:
            norm = (diff / max_diff * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(diff, dtype=np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
