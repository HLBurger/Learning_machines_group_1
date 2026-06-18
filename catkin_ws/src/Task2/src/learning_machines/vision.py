"""
vision.py — green food blob detection for Task 2.

Walls are white, food is green — no wall/object classification needed.
Just find the green blob and return 3 normalised features.

Features returned:
    obj_visible : float  0 or 1
    obj_dx      : float [-1, 1]  horizontal offset (0 = centered)
    obj_size    : float [0, 1]   blob area / frame area (larger = closer)
"""

import cv2
import numpy as np
from .constants_sac import (
    GREEN_LOWER_SIM,
    GREEN_UPPER_SIM,
    GREEN_LOWER_HW,
    GREEN_UPPER_HW,
    MIN_BLOB_AREA_FRAC,
)


def analyse_frame(frame_bgr: np.ndarray, hardware: bool = False) -> dict:
    """
    Detect green food blob and return normalised features.

    Parameters
    ----------
    frame_bgr : np.ndarray — BGR image from rob.read_image_front()
    hardware  : bool       — True when on real Robobo (wider HSV range)

    Returns
    -------
    dict with keys:
        obj_visible : bool  — green blob detected
        obj_dx      : float — horizontal offset [-1, 1], 0 = centered
        obj_size    : float — blob area / frame area [0, 1], larger = closer
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {"obj_visible": False, "obj_dx": 0.0, "obj_size": 0.0}

    h, w   = frame_bgr.shape[:2]
    area   = float(h * w)

    # HSV mask
    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(GREEN_LOWER_HW if hardware else GREEN_LOWER_SIM, dtype=np.uint8)
    upper = np.array(GREEN_UPPER_HW if hardware else GREEN_UPPER_SIM, dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # clean noise
    k    = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        blob_area = cv2.contourArea(c)
        if blob_area / area < MIN_BLOB_AREA_FRAC:
            continue
        if best is None or blob_area > best[0]:
            M  = cv2.moments(c)
            cx = M["m10"] / M["m00"] if M["m00"] != 0 else w / 2
            best = (blob_area, cx)

    if best is None:
        return {"obj_visible": False, "obj_dx": 0.0, "obj_size": 0.0}

    blob_area, cx = best
    return {
        "obj_visible": True,
        "obj_dx"     : float((cx - w / 2.0) / (w / 2.0)),  # [-1, 1]
        "obj_size"   : float(min(blob_area / area, 1.0)),   # [0, 1]
    }


def vision_features(frame_bgr: np.ndarray, hardware: bool = False) -> np.ndarray:
    """
    Return vision as a numpy array for the SAC state vector.
    Shape: (3,) — [obj_visible, obj_dx, obj_size]
    """
    feats = analyse_frame(frame_bgr, hardware=hardware)
    return np.array([
        1.0 if feats["obj_visible"] else 0.0,
        feats["obj_dx"],
        feats["obj_size"],
    ], dtype=np.float32)


def no_vision() -> np.ndarray:
    """Zero vision features when camera unavailable."""
    return np.zeros(3, dtype=np.float32)


def classify_collision(
    frame_bgr,
    irs_front: list,
    ir_collision_threshold: float,
) -> str:
    """
    Return "wall", "object", or "none" based on IR + camera.

    Logic:
      - High IR + green blob visible → object (food)
      - High IR + no green          → wall
      - Low IR                      → none
    """
    if frame_bgr is None:
        front_max = max(irs_front) if irs_front else 0
        return "wall" if front_max > ir_collision_threshold else "none"

    front_max = max(irs_front) if irs_front else 0
    high_ir   = front_max > ir_collision_threshold

    if not high_ir:
        return "none"

    feats = analyse_frame(frame_bgr)
    return "object" if feats["obj_visible"] else "wall"