# pyAgxArm/demos/piper/digit/tactile_heatmap.py
import sys

import cv2
import numpy as np

from pyAgxArm.utiles.fps import FPSManager

_BASELINE_N = 30
_WINDOW = "DIGIT Tactile Heatmap  |  raw (left)  contact heatmap (right)"


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


def _overlay(img: np.ndarray, fps: float, blur: bool, capturing: bool) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if capturing:
        cv2.putText(img, "Capturing baseline...", (10, 30), font, 0.7, (0, 255, 255), 2)
        return
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), font, 0.7, (255, 255, 255), 2)
    blur_label = "Blur: ON" if blur else "Blur: OFF"
    cv2.putText(img, blur_label, (10, 60), font, 0.7, (0, 255, 0) if blur else (0, 100, 255), 2)
    cv2.putText(img, "b=recapture  g=blur  q=quit", (10, img.shape[0] - 10),
                font, 0.5, (200, 200, 200), 1)


def main() -> None:
    from digit_interface import Digit, DigitHandler

    digits = DigitHandler.find_digits()
    if not digits:
        print("No DIGIT sensor found. Check USB connection.")
        sys.exit(1)

    sensor = Digit(digits[0].serial)
    sensor.connect()
    print(f"Connected to DIGIT {digits[0].serial}")

    fps_mgr = FPSManager(start_realtime_fps=True)
    fps_mgr.add_variable("digit")
    fps_mgr.start()

    proc = FrameProcessor(blur=True, baseline_frames=_BASELINE_N)
    capturing = True
    print(f"Capturing baseline ({_BASELINE_N} frames)...")

    try:
        while True:
            frame = sensor.get_frame()
            if frame is None:
                continue

            if capturing:
                done = proc.add_baseline_frame(frame)
                display = np.hstack([frame, frame])
                _overlay(display, 0.0, proc.blur_enabled, capturing=True)
                cv2.imshow(_WINDOW, display)
                if done:
                    capturing = False
                    print("Baseline ready.")
            else:
                fps_mgr.increment("digit")
                heatmap = proc.compute_heatmap(frame)
                display = np.hstack([frame, heatmap])
                current_fps = fps_mgr.get_real_time_fps("digit", window=1.0)
                _overlay(display, current_fps, proc.blur_enabled, capturing=False)
                cv2.imshow(_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("g"):
                state = proc.toggle_blur()
                print(f"Gaussian blur: {'ON' if state else 'OFF'}")
            elif key == ord("b"):
                proc.reset_baseline()
                capturing = True
                print(f"Recapturing baseline ({_BASELINE_N} frames)...")

    finally:
        fps_mgr.stop()
        sensor.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
