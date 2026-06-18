import numpy as np

import cv2

GREEN_LOWER = np.array([35, 60, 40])
GREEN_UPPER = np.array([85, 255, 255])

MIN_BLOB_AREA_FRAC = 0.003   # ignore tiny green specks/noise (<0.3% of frame)
WALL_WIDTH_FRAC = 0.75       # blob wider than this fraction of frame width -> wall-like
TOP_TOUCH_MARGIN_PX = 5      # how close to the top row counts as "touching"
BOTTOM_TOUCH_MARGIN = 0.85   # blob bottom reacher this fraction of frame  height
WALL_ASPECT_RATIO = 4.0      # width/height > this > wall-like strip

def _green_mask(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    # Clean up small noise / fill small holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def analyse_frame(frame_bgr: np.ndarray) -> dict:
    """
    Analyse one camera frame and return a compact feature dict.

    Returns
    -------
    dict with keys:
        object_visible   : bool  — an object-like green blob is in view
        object_dx        : float — horizontal offset of the largest object
                                    blob's centroid from frame center,
                                    normalised to [-1, 1] (0 = centered)
        object_size      : float — area of the largest object blob,
                                    normalised to [0, 1] by frame area
                                    (bigger = closer)
        wall_visible      : bool  — a wall-like green blob is in view
        wall_frac         : float — fraction of frame width covered by
                                    the widest wall-like blob, [0, 1]
    """
    h, w = frame_bgr.shape[:2]
    frame_area = float(h * w)
    mask = _green_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_object = None   # (area, cx, cy, x, y, bw, bh)
    best_wall = None      # (width_frac, x, y, bw, bh)

    for c in contours:
        area = cv2.contourArea(c)
        if area / frame_area < MIN_BLOB_AREA_FRAC:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        touches_top = y <= TOP_TOUCH_MARGIN_PX
        touches_bottom = (y + bh) >= (h * BOTTOM_TOUCH_MARGIN)
        width_frac = bw / float(w)
        aspect_ratio = bw / float(bh) if bh > 0 else 0

        is_wall_like = (
            (width_frac >= WALL_WIDTH_FRAC or aspect_ratio >= WALL_ASPECT_RATIO)
            and (touches_top or touches_bottom)
        )

        if is_wall_like:
            if best_wall is None or width_frac > best_wall[0]:
                best_wall = (width_frac, x, y, bw, bh)
        else:
            if best_object is None or area > best_object[0]:
                M  = cv2.moments(c)
                cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + bw / 2
                best_object = (area, cx, y, x, y, bw, bh)

    if best_object is not None:
        area, cx, *_ = best_object
        object_visible = True
        object_dx = float((cx - w / 2.0) / (w / 2.0))   # -1..1
        object_size = float(area / frame_area)            # 0..1
    else:
        object_visible = False
        object_dx = 0.0
        object_size = 0.0

    if best_wall is not None:
        wall_visible = True
        wall_frac = float(best_wall[0])
    else:
        wall_visible = False
        wall_frac = 0.0

    return {
        "object_visible": object_visible,
        "object_dx": object_dx,
        "object_size": object_size,
        "wall_visible": wall_visible,
        "wall_frac": wall_frac,
    }


# ------------------------------------------------------------------
# Thresholds for the IR × green decision matrix
# ------------------------------------------------------------------
IR_HIGH_FRAC      = 0.5   # normalised front-IR median above this → "high IR"
GREEN_HIGH_FRAC   = 0.3  # green pixel fraction of frame above this → "high green"


def _green_score(frame_bgr: np.ndarray) -> float:
    """
    Return the fraction of frame pixels that are green, in [0, 1].
    Uses the same HSV mask as the rest of the vision pipeline so the
    threshold is consistent.
    """
    mask = _green_mask(frame_bgr)
    return float(np.count_nonzero(mask)) / mask.size


def classify_collision(
    frame_bgr: np.ndarray,
    irs_front: "list[float] | float",
    ir_collision_threshold: float,
) -> str:
    """
    Classify what the robot is colliding with (or not) using a combined
    IR × vision decision matrix.

    Decision table (median of front IR sensors vs. green pixel fraction):

        IR    | Green  | Interpretation
        ------+--------+---------------
        High  | High   | wall (green wall close up) — shape heuristics used
        High  | Low    | wall (non-green wall)
        Low   | High   | object / food (green object not yet touching)
        Low   | Low    | none

    Parameters
    ----------
    frame_bgr             : BGR camera frame
    irs_front             : iterable of raw front IR readings (length ≥ 1),
                            or a single float for backwards compatibility.
                            Values are expected in [0, 255].
    ir_collision_threshold: raw IR value above which a reading is "high"
    """
    # ---- normalised IR signal (median of front sensors, 0‥1) ----------
    if isinstance(irs_front, (int, float)):
        ir_vals = [float(irs_front)]
    else:
        ir_vals = [float(v) for v in irs_front]

    ir_median_norm = float(np.median(ir_vals)) / 255.0
    high_ir    = ir_median_norm >= (ir_collision_threshold / 255.0)

    # ---- vision signal ------------------------------------------------
    green_frac = _green_score(frame_bgr)
    high_green = green_frac >= GREEN_HIGH_FRAC

    # ---- 2 × 2 decision table ----------------------------------------
    if not high_ir and not high_green:
        return "none"

    if not high_ir and high_green:
        # Green visible but sensors not triggered → food/object approaching
        return "object"

    if high_ir and not high_green:
        # Close obstacle but no green signal → non-green wall
        return "wall"

    # high_ir AND high_green: use shape heuristics to tell wall from object
    feats = analyse_frame(frame_bgr)

    if feats["wall_visible"] and feats["wall_frac"] >= WALL_WIDTH_FRAC:
        return "wall"

    if feats["object_visible"] and feats["object_size"] > 0.05 and abs(feats["object_dx"]) < 0.5:
        return "object"

    # Fallback: large IR + large green but shape is ambiguous → treat as wall
    return "wall"
