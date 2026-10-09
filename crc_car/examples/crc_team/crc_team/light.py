"""Traffic light colour by HSV thresholds: a simple, fast detector for the START light.
Real LED lamps look different from the simulator (brighter, often white in the middle,
coloured halo), so tune the ranges on frames from your car with tools/tune_hsv.py.
For the light in the middle of the track, a YOLO class ("red_light", "green_light")
is more robust: red STOP signs and other red objects can fool a colour detector.
"""
import cv2
import numpy as np

# Hue in OpenCV is 0..179. Each colour: list of (low HSV, high HSV) ranges.
DEFAULT_RANGES = {
    'red': [((0, 120, 150), (8, 255, 255)), ((170, 120, 150), (179, 255, 255))],
    'yellow': [((15, 120, 150), (35, 255, 255))],
    'green': [((45, 80, 120), (95, 255, 255))],
}


class TrafficLight:
    def __init__(self, roi=(0.0, 0.0, 1.0, 0.6), min_area=20, max_area=5000,
                 min_fill=0.5, ranges=None):
        self.roi = roi                  # x0, y0, x1, y1 as fractions of the image
        self.min_area = min_area        # lamp size in pixels (640x480 image)
        self.max_area = max_area
        self.min_fill = min_fill        # blob area / bounding-circle area: lamps are round
        self.ranges = ranges or DEFAULT_RANGES
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, rgb):
        """Returns (colour, area, (cx, cy)) of the biggest lamp-like blob, or (None, 0, None).
        `rgb` is an RGB image (as published by the car and the simulator)."""
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = (int(self.roi[0] * w), int(self.roi[1] * h),
                          int(self.roi[2] * w), int(self.roi[3] * h))
        hsv = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
        best = (None, 0, None)
        for colour, ranges in self.ranges.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = m if mask is None else mask | m
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not self.min_area <= area <= self.max_area:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(c)
                if area / (np.pi * r * r + 1e-6) < self.min_fill:
                    continue
                if area > best[1]:
                    best = (colour, area, (cx + x0, cy + y0))
        return best
