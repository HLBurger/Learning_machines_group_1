# vision.py
import cv2
import numpy as np


def detect_color_features(image, color_name):
    """
    Returns:
        seen: 0.0 or 1.0
        area: fraction of image covered by the color
        center_x: -1.0 left, 0.0 center, +1.0 right
    """

    if image is None:
        return [0.0, 0.0, 0.0]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color_name == "red":
        lower1 = np.array([0, 80, 80])
        upper1 = np.array([10, 255, 255])

        lower2 = np.array([170, 80, 80])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    elif color_name == "green":
        lower = np.array([40, 60, 60])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif color_name == "white":
        # White has low saturation and high brightness.
        lower = np.array([0, 0, 180])
        upper = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower, upper)

    else:
        raise ValueError(f"Unknown color: {color_name}")

    red_or_color_pixels = cv2.countNonZero(mask)
    height, width = mask.shape
    total_pixels = height * width

    area = red_or_color_pixels / total_pixels
    seen = 1.0 if area > 0.01 else 0.0

    if seen == 0.0:
        return [0.0, 0.0, 0.0]

    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        center_x = 0.0
    else:
        cx = moments["m10"] / moments["m00"]
        center_x = (cx / width) * 2.0 - 1.0

    return [seen, area, center_x]


def build_state(rob):
    """
    State:
        8 IR sensors
        3 red features
        3 green features
        3 white features
    Total: 17 values
    """

    irs = rob.read_irs()

    try:
        image = rob.read_image_front()

        red_features = detect_color_features(image, "red")
        green_features = detect_color_features(image, "green")
        white_features = detect_color_features(image, "white")

    except Exception:
        red_features = [0.0, 0.0, 0.0]
        green_features = [0.0, 0.0, 0.0]
        white_features = [0.0, 0.0, 0.0]

    return list(irs) + red_features + green_features + white_features