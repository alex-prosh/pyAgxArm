# DIGIT Tactile Sensor Contact Heatmap Visualizer

**Date:** 2026-04-17
**Status:** Approved

## Overview

A real-time visualization tool for the Meta DIGIT tactile sensor (pad version). Displays a live contact heatmap derived from grayscale frame differencing against a captured baseline. Intended for testing and verifying sensor function when integrated with the Piper arm.

## File Layout

```
pyAgxArm/demos/piper/digit/
├── __init__.py
└── tactile_heatmap.py
```

Single script, consistent with the pattern of other demos in this project.

## Architecture & Data Flow

```
DIGIT sensor (USB-3)
    │
    ▼
digit-interface SDK  →  raw BGR frame (640×480 @ 30fps)
    │
    ▼
FrameProcessor
  - holds baseline (averaged over first 30 frames, or re-captured on keypress)
  - grayscale diff: abs(current_gray - baseline_gray)
  - optional Gaussian blur (15x15 kernel, sigma=0), toggle with 'g'
  - normalize + COLORMAP_JET  →  heatmap frame
    │
    ▼
DisplayWindow (OpenCV)
  side-by-side: [Raw Feed | Contact Heatmap]
  overlay text: FPS, blur on/off, key hints
    │
    ▼
Keyboard controls (cv2.waitKey)
  'b' — recapture baseline (averages 30 frames live)
  'g' — toggle Gaussian smoothing
  'q' — quit
```

## Key Implementation Details

**Baseline capture:**
- On startup: average first 30 frames (grayscale float32) for a stable reference
- On `'b'` keypress: collect 30 fresh frames and replace stored baseline
- Baseline stored as float32 grayscale array

**Heatmap computation:**
- Convert current frame to grayscale float32
- `diff = abs(current_gray - baseline_gray)`
- If blur enabled: `cv2.GaussianBlur(diff, (15, 15), 0)`
- Normalize to 0–255 uint8, apply `cv2.COLORMAP_HOT` (black=no contact, bright=contact)

**Display:**
- `np.hstack([raw_bgr, heatmap_bgr])` → single `cv2.imshow` window
- `cv2.putText` overlays: FPS, `Blur: ON/OFF`, key hint bar
- FPS via rolling average using `pyAgxArm/utiles/fps.py`

**Entry point:**
- Script calls `DigitHandler.list()` to enumerate connected sensors, picks the first one
- Exits with a clear error message if no sensor is found
- Graceful cleanup on `'q'` or window close

## Dependencies

Not in `pyproject.toml` (installed separately by user):
- `digit-interface` (`pip install digit-interface`)
- `opencv-python`
- `numpy`
