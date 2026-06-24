"""
vision.py — red object and green goal detection for Task 3 pushing.

The robot must push a bright red object into a green goal area.
Walls are white — no wall/object confusion in HSV space.

Features returned by analyse_frame():
    red_visible  : float  0 or 1   — red object detected above min area
    red_dx       : float [-1, 1]   — horizontal offset of red blob (0 = centred)
    red_size     : float [0, 1]    — red blob area / frame area (larger = closer)
    goal_visible : float  0 or 1   — green goal area detected above min area
    goal_dx      : float [-1, 1]   — horizontal offset of green goal (0 = centred)
    goal_size    : float [0, 1]    — green goal area / frame area
    goal_reached : bool            — True when red object centre pixel falls
                                     inside the green goal mask
"""

import cv2
import numpy as np

from .constants_sac import (
    RED1_LOWER_SIM, RED1_UPPER_SIM,
    RED2_LOWER_SIM, RED2_UPPER_SIM,
    RED1_LOWER_HW,  RED1_UPPER_HW,
    RED2_LOWER_HW,  RED2_UPPER_HW,
    GREEN_LOWER_SIM, GREEN_UPPER_SIM,
    GREEN_LOWER_HW,  GREEN_UPPER_HW,
    MIN_RED_AREA_FRAC,
    MIN_GOAL_AREA_FRAC,
    GOAL_REACHED_DILATE_ITERS
)


def analyse_frame(
    frame_bgr: np.ndarray,
    hardware: bool = False,
) -> dict:
    """
    Detect the red push object and the green goal area in a single BGR frame.

    Parameters
    ----------
    frame_bgr : np.ndarray — raw BGR image from the front camera
    hardware  : bool       — use hardware HSV ranges when True,
                             simulation ranges when False

    Returns
    -------
    {
        red_visible  : bool,
        red_dx       : float,   # [-1, 1], 0 = centred
        red_size     : float,   # [0, 1], fraction of frame area
        goal_visible : bool,
        goal_dx      : float,   # [-1, 1], 0 = centred
        goal_size    : float,   # [0, 1], fraction of frame area
        goal_reached : bool,    # red mask overlaps (dilated) green mask
    }
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "red_visible"  : False, "red_dx"  : 0.0, "red_size"  : 0.0,
            "goal_visible" : False, "goal_dx" : 0.0, "goal_size" : 0.0,
            "goal_reached" : False,
        }

    h, w        = frame_bgr.shape[:2]
    frame_area  = float(h * w)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # ── Select HSV ranges ──────────────────────────────────────────────
    if hardware:
        r1_lower = np.array(RED1_LOWER_HW,  dtype=np.uint8)
        r1_upper = np.array(RED1_UPPER_HW,  dtype=np.uint8)
        r2_lower = np.array(RED2_LOWER_HW,  dtype=np.uint8)
        r2_upper = np.array(RED2_UPPER_HW,  dtype=np.uint8)
        g_lower  = np.array(GREEN_LOWER_HW, dtype=np.uint8)
        g_upper  = np.array(GREEN_UPPER_HW, dtype=np.uint8)
    else:
        r1_lower = np.array(RED1_LOWER_SIM,  dtype=np.uint8)
        r1_upper = np.array(RED1_UPPER_SIM,  dtype=np.uint8)
        r2_lower = np.array(RED2_LOWER_SIM,  dtype=np.uint8)
        r2_upper = np.array(RED2_UPPER_SIM,  dtype=np.uint8)
        g_lower  = np.array(GREEN_LOWER_SIM, dtype=np.uint8)
        g_upper  = np.array(GREEN_UPPER_SIM, dtype=np.uint8)

    # ── Build masks ────────────────────────────────────────────────────
    # Red wraps around H=0/180, so two inRange calls are OR-ed.
    red_mask = (
        cv2.inRange(hsv, r1_lower, r1_upper)
        | cv2.inRange(hsv, r2_lower, r2_upper)
    )
    green_mask = cv2.inRange(hsv, g_lower, g_upper)

    # Morphological open+close removes small noise and fills small holes.
    kernel = np.ones((5, 5), np.uint8)
    red_mask   = cv2.morphologyEx(red_mask,   cv2.MORPH_OPEN,  kernel)
    red_mask   = cv2.morphologyEx(red_mask,   cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # ── Detect largest red object ──────────────────────────────────────
    red_visible = False
    red_dx      = 0.0
    red_size    = 0.0

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c         = max(contours, key=cv2.contourArea)
        blob_area = cv2.contourArea(c)
        if blob_area / frame_area >= MIN_RED_AREA_FRAC:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                red_visible = True
                red_dx      = float((cx - w / 2.0) / (w / 2.0))
                red_size    = float(min(blob_area / frame_area, 1.0))

    # ── Detect largest green goal area ────────────────────────────────
    goal_visible = False
    goal_dx      = 0.0
    goal_size    = 0.0

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c         = max(contours, key=cv2.contourArea)
        blob_area = cv2.contourArea(c)
        if blob_area / frame_area >= MIN_GOAL_AREA_FRAC:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                goal_visible = True
                goal_dx      = float((cx - w / 2.0) / (w / 2.0))
                goal_size    = float(min(blob_area / frame_area, 1.0))

    # ── Goal-reached: dilated red mask overlaps green mask ────────────
    # Dilating the red mask lets us catch a thin green fringe/edge that's
    # still visible around the red object even when the object is mostly
    # or fully sitting on top of (occluding) the green tile.
    goal_reached = False
    if red_visible and goal_visible:
        red_dilated = cv2.dilate(red_mask, kernel, iterations=GOAL_REACHED_DILATE_ITERS)
        overlap     = cv2.bitwise_and(red_dilated, green_mask)
        goal_reached = cv2.countNonZero(overlap) > 0

    return {
        "red_visible"  : red_visible,
        "red_dx"       : red_dx,
        "red_size"     : red_size,
        "goal_visible" : goal_visible,
        "goal_dx"      : goal_dx,
        "goal_size"    : goal_size,
        "goal_reached" : goal_reached,
    }


def vision_features(
    frame_bgr: np.ndarray,
    hardware: bool = False,
) -> np.ndarray:
    """
    Return a flat float32 array of 6 normalised vision features:
        [red_visible, red_dx, red_size, goal_visible, goal_dx, goal_size]
    Suitable for direct concatenation into the SAC state vector.
    """
    feats = analyse_frame(frame_bgr, hardware=hardware)
    return np.array([
        1.0 if feats["red_visible"]  else 0.0,
        feats["red_dx"],
        feats["red_size"],
        1.0 if feats["goal_visible"] else 0.0,
        feats["goal_dx"],
        feats["goal_size"],
    ], dtype=np.float32)


def no_vision() -> np.ndarray:
    """Return a zero vision feature vector (all features absent)."""
    return np.zeros(6, dtype=np.float32)


def classify_collision(
    frame_bgr,
    irs_front: list,
    ir_collision_threshold: float,
) -> str:
    """
    Determine the type of obstacle triggering the front IR sensors.

    Returns one of:
        "wall"   — high IR, no red object visible in camera
        "object" — high IR, red object visible in camera (robot touching object)
        "none"   — IR below threshold

    This distinction is used in the reward function to avoid penalising
    intentional contact with the red push object as a wall collision.
    """
    if frame_bgr is None:
        front_max = max(irs_front) if irs_front else 0
        return "wall" if front_max > ir_collision_threshold else "none"

    front_max = max(irs_front) if irs_front else 0
    if front_max <= ir_collision_threshold:
        return "none"

    feats = analyse_frame(frame_bgr)
    return "object" if feats["red_visible"] else "wall"
