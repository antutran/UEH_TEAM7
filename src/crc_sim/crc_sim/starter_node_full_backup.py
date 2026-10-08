#!/usr/bin/env python3

"""
UEH CRC 2026 Simulation Round

V4.8
CONNECTED RIGHT-BOUNDARY POLYLINE
+ DELAYED PREVIEW
+ ADAPTIVE LANE THICKNESS
+ NEAREST-CARDINAL IMU LOST RECOVERY

Main behaviour
--------------
NORMAL:
    camera -> lane vectors -> thickness gate
    -> connected right-boundary polyline
    -> lateral + heading + delayed preview control

SHORT LOST:
    preserve last good steering / last good IMU heading

LONG LOST:
    DO NOT STOP merely because vision has been lost for a long time
    -> normally use IMU to return toward the cardinal heading latched at first LOST
       (0, 90, 180 or 270 degrees), except while the persistent SEEK BOTH
       window is active
    -> creep straight slowly
    -> keep seeking the currently tracked boundary first
    -> after 15 s inside LONG LOST, start SEEK BOTH for both outer boundaries
    -> once SEEK BOTH starts, remember that mode for 15 s even if a lane is
       reacquired in the meantime
    -> if the lane is lost again during that 15 s window, hold the last valid
       lane yaw for 3 s, then return to SEEK BOTH instead of a cardinal anchor
    -> while that window is active, LONG LOST also stays in SEEK BOTH
    -> only after the 15 s SEEK BOTH window expires may LOST/LONG LOST return
       to the nearest 0/90/180/270-degree IMU anchor
    -> if both sides are valid at the same time, prefer the right boundary
    -> if SEEK BOTH remains unable to lock either side for 8 s, abort SEEK BOTH,
       rotate LEFT by 90 degrees using IMU yaw, then SEEK RIGHT only
    -> if only LEFT is recovered, follow it as a temporary +200 px proxy
       while continuously searching for RIGHT; switch back once RIGHT is stable

PERCEPTION:
    low-rate camera pipeline watches STOP/RAMP/UNEVEN/HW/BUS/TUNNEL/CROSSWALK + traffic light
    -> BUS is detected immediately; TUNNEL requires 1.0 s; neither changes motion
    -> CROSSWALK uses the starter-node blue-triangle detector
    -> CROSSWALK must remain detected continuously for 1.2 s before it is accepted
    -> after confirmed CROSSWALK disappears: follow lane 3.0 s, then watch the pink pedestrian
    -> if pink person + front LiDAR obstacle agree: stop until clear
    -> once clear: follow lane at 0.20 m/s for 1.5 s to leave the zebra quickly
    -> HW_ENTRY must remain detected continuously for 0.6 s before it enables obstacle bypass; HW_EXIT disables it
    -> ALL camera detection is paused during obstacle bypass and resumes afterwards
    -> traffic-light RED/YELLOW/GREEN require confidence > 0.86 continuously 0.8 s
    -> RED/YELLOW stop and freeze controller state; confirmed GREEN releases
    -> RAMP sign boosts lane-following to 0.25 m/s for 2.0 s, after safety gates

SAFETY / OBSTACLE POLICY:
    obstacle = STOP/WAIT by default; only HW_ENTRY..HW_EXIT permits bypass.
    During the armed crosswalk sequence, pedestrian handling owns the front-obstacle decision.

Allowed sensors:
    /camera/image_raw
    /scan
    /odom
    /imu
"""

import math
import signal
import time

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan


# =====================================================================
# Utilities
# =====================================================================

def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_angle(angle):

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def angle_diff_deg(a, b):

    d = abs(a - b) % 180.0

    if d > 90.0:
        d = 180.0 - d

    return d


def point_segment_distance(
    px,
    py,
    ax,
    ay,
    bx,
    by
):

    vx = bx - ax
    vy = by - ay

    wx = px - ax
    wy = py - ay

    vv = (
        vx * vx
        +
        vy * vy
    )

    if vv < 1e-9:

        return math.hypot(
            px - ax,
            py - ay
        )

    t = (
        wx * vx
        +
        wy * vy
    ) / vv

    t = clamp(
        t,
        0.0,
        1.0
    )

    qx = (
        ax
        +
        t * vx
    )

    qy = (
        ay
        +
        t * vy
    )

    return math.hypot(
        px - qx,
        py - qy
    )


def ccw(
    ax,
    ay,
    bx,
    by,
    cx,
    cy
):

    return (
        (cy - ay)
        * (bx - ax)
        >
        (by - ay)
        * (cx - ax)
    )


def segments_intersect(a, b):

    ax = a['xf']
    ay = a['yf']

    bx = a['xn']
    by = a['yn']

    cx = b['xf']
    cy = b['yf']

    dx = b['xn']
    dy = b['yn']

    return (
        ccw(
            ax,
            ay,
            cx,
            cy,
            dx,
            dy
        )
        !=
        ccw(
            bx,
            by,
            cx,
            cy,
            dx,
            dy
        )
        and
        ccw(
            ax,
            ay,
            bx,
            by,
            cx,
            cy
        )
        !=
        ccw(
            ax,
            ay,
            bx,
            by,
            dx,
            dy
        )
    )


def find_external_contours(mask):
    out = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return out[0] if len(out) == 2 else out[1]


def count_curve_extrema(binary_dark):
    """Very small heuristic for ramp-vs-uneven.
    For each x column, sample the median y of dark symbol pixels.  A single
    hump usually changes vertical direction fewer times than an uneven-road
    double-wave symbol.
    """
    h, w = binary_dark.shape[:2]
    if h < 10 or w < 10:
        return 0
    xs = []
    ys = []
    for x in range(w):
        yy = np.flatnonzero(binary_dark[:, x] > 0)
        if len(yy) >= 1:
            xs.append(x)
            ys.append(float(np.median(yy)))
    if len(ys) < max(8, w // 5):
        return 0
    y = np.asarray(ys, dtype=np.float32)
    # Smooth enough to suppress letter/pixel noise.
    k = max(3, int(round(len(y) * 0.09)))
    if k % 2 == 0:
        k += 1
    k = min(k, 11)
    if k >= 3 and len(y) >= k:
        y = cv2.GaussianBlur(y.reshape(1, -1), (k, 1), 0).reshape(-1)
    d = np.diff(y)
    if len(d) < 3:
        return 0
    # Ignore tiny slope jitter.
    d[np.abs(d) < 0.35] = 0.0
    signs = []
    for value in d:
        s = 1 if value > 0 else (-1 if value < 0 else 0)
        if s == 0:
            continue
        if not signs or signs[-1] != s:
            signs.append(s)
    return max(0, len(signs) - 1)


class SignPerception:
    """Camera-only sign and traffic-light perception.

    Motion/control reactions remain in Starter.  This detector watches the
    right-side roadside ROI for STOP / RAMP / UNEVEN / HIGHWAY / BUS / TUNNEL /
    CROSSWALK signs, can optionally watch the pink pedestrian, and independently
    finds the illuminated traffic-light lamp.
    """

    SIGN_NAMES = (
        'bus', 'crosswalk', 'hw_entry', 'hw_exit',
        'ramp', 'stop', 'tunnel', 'uneven',
    )

    def __init__(self, node):
        self.get_logger = node.get_logger
        defaults = {
            'min_area': 120,
            'max_area_ratio': 0.16,
            'light_min_area': 12.0,
            'light_max_area': 950.0,
            'bus_candidate_min_area': 60,
            'tunnel_confirm_sec': 1.0,
            'bus_min_width_px': 18,
            'bus_min_height_px': 18,
            'pedestrian_min_area': 90.0,
            'ped_zone_left': 0.33,
            'ped_zone_right': 0.67,
            'ped_zone_top': 0.25,
            'ped_zone_bottom': 0.86,
            'blue_h_low': 96,
            'blue_h_high': 134,
            'blue_s_min': 80,
            'blue_v_min': 90,
            'yellow_s_min': 32,
            'yellow_v_min': 95,
            'green_s_min': 24,
            'green_v_min': 82,
            'uneven_extrema_threshold': 5,
            'nms_iou_threshold': 0.28,
            'nms_overlap_threshold': 0.58,
            'sign_min_conf': 0.54,
            'sign_min_width_px': 36,
            'sign_min_height_px': 36,
            'sign_right_start_ratio': 0.55,
            'sign_top_ratio': 0.06,
            'sign_bottom_ratio': 0.88,
            'sign_release_frames': 8,
            'show_sign_roi': True,
        }
        for name, default in defaults.items():
            node.declare_parameter(name, default)
            setattr(self, name, node.get_parameter(name).value)

        self.sign_encounter_active = False
        self.sign_missing_frames = 0
        self.sign_event_id = 0
        self.last_sign_event_label = None
        self.last_sign_event_box = None
        self.last_detections = []
        self.last_light_state = 'UNKNOWN'
        self.last_light_box = None
        self.last_pedestrians = []
        self.pedestrian_in_zone = False
        self.pedestrian_detection_enabled = False

        # BUS is exposed immediately once the tuned geometry + BUS-specific
        # size gate accept it. TUNNEL remains conservative and must survive its
        # own continuous confirmation window before being exposed.
        self.tunnel_candidate_since = None
        self.tunnel_candidate_box = None
        self.tunnel_candidate_confidence = 0.0

        self.last_observation = {}

    def color_masks(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 75, 55), (12, 255, 255))
        red2 = cv2.inRange(hsv, (168, 75, 55), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)

        # Bright-blue information signs.  The saturation/value floor rejects
        # most of the darker parked robot while retaining BUS/TUNNEL plates.
        blue = cv2.inRange(
            hsv,
            (self.blue_h_low, self.blue_s_min, self.blue_v_min),
            (self.blue_h_high, 255, 255),
        )

        # Highway entry/exit plates are green.  HW_EXIT is separated later by
        # detecting the red diagonal slash through the green panel.
        green = cv2.inRange(hsv, (35, 55, 40), (92, 255, 255))

        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)
        result = {}
        for name, mask in (('red', red), ('blue', blue), ('green', green)):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
            result[name] = mask
        return result

    @staticmethod
    def box_iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def box_overlap_smaller(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        smaller = min(max(1, aw * ah), max(1, bw * bh))
        return inter / float(smaller)

    def suppress_overlaps(self, detections):
        """Class-agnostic NMS: one overlapping object survives."""
        ranked = sorted(
            detections,
            key=lambda d: (
                float(d.get('confidence', 0.0)),
                d['box'][2] * d['box'][3],
            ),
            reverse=True,
        )
        kept = []
        for det in ranked:
            duplicate = False
            for winner in kept:
                iou = self.box_iou(det['box'], winner['box'])
                contained = self.box_overlap_smaller(det['box'], winner['box'])
                if (iou >= self.nms_iou_threshold or
                        contained >= self.nms_overlap_threshold):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    @staticmethod
    def expand_box(box, image_shape, frac=0.10):
        x, y, w, h = box
        H, W = image_shape[:2]
        px = int(round(w * frac))
        py = int(round(h * frac))
        x0 = clamp(x - px, 0, W - 1)
        y0 = clamp(y - py, 0, H - 1)
        x1 = clamp(x + w + px, 1, W)
        y1 = clamp(y + h + py, 1, H)
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    def candidate_boxes(self, image, masks):
        H, W = image.shape[:2]
        image_area = float(H * W)
        raw = []
        for family, mask in masks.items():
            for contour in find_external_contours(mask):
                area = cv2.contourArea(contour)
                min_area = (min(self.min_area, self.bus_candidate_min_area)
                            if family == 'blue' else self.min_area)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                box_area = float(w * h)
                if box_area <= 0 or box_area > self.max_area_ratio * image_area:
                    continue
                aspect = w / float(max(h, 1))
                if aspect < 0.45 or aspect > 1.85:
                    continue
                if min(w, h) < 10:
                    continue
                roi = mask[y:y + h, x:x + w]
                occupancy = cv2.countNonZero(roi) / max(1.0, box_area)
                if occupancy < 0.035:
                    continue
                raw.append({
                    'family': family,
                    'box': (x, y, w, h),
                    'area': area,
                    'occupancy': occupancy,
                    'contour': contour,
                })
        raw.sort(key=lambda d: d['area'], reverse=True)
        return raw[:20]

    def select_one_right_sign(self, detections, image_shape):
        """Keep at most one sufficiently large sign in the right-side ROI."""
        H, W = image_shape[:2]
        valid = []
        for det in detections:
            x, y, w, h = det['box']
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            if float(det.get('confidence', 0.0)) < self.sign_min_conf:
                continue
            # BUS keeps the far-distance size gate from Code 2.
            if det.get('label') == 'bus':
                min_w, min_h = self.bus_min_width_px, self.bus_min_height_px
            else:
                min_w, min_h = self.sign_min_width_px, self.sign_min_height_px
            if w < min_w or h < min_h:
                continue
            if cx < self.sign_right_start_ratio * W:
                continue
            if cy < self.sign_top_ratio * H or cy > self.sign_bottom_ratio * H:
                continue
            valid.append(det)
        if not valid:
            return []
        best = max(
            valid,
            key=lambda d: (
                float(d.get('confidence', 0.0)),
                d['box'][2] * d['box'][3],
            ),
        )
        return [best]

    def update_sign_encounter(self, detections):
        """Return True once per physical sign encounter."""
        if detections:
            self.sign_missing_frames = 0
            if not self.sign_encounter_active:
                self.sign_encounter_active = True
                self.sign_event_id += 1
                self.last_sign_event_label = detections[0]['label']
                self.last_sign_event_box = detections[0]['box']
                self.get_logger().info(
                    f'SIGN #{self.sign_event_id}: {self.last_sign_event_label} '
                    f'conf={detections[0]["confidence"]:.2f}')
                return True
            return False

        if self.sign_encounter_active:
            self.sign_missing_frames += 1
            if self.sign_missing_frames >= max(1, self.sign_release_frames):
                self.sign_encounter_active = False
                self.sign_missing_frames = 0
                self.last_sign_event_label = None
                self.last_sign_event_box = None
        return False

    @staticmethod
    def white_mask(crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (0, 0, 145), (180, 95, 255))

    @staticmethod
    def dark_mask(crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, (0, 0, 0), (180, 150, 115))

    @staticmethod
    def red_ratio(crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, (0, 75, 55), (12, 255, 255))
        r2 = cv2.inRange(hsv, (168, 75, 55), (180, 255, 255))
        red = cv2.bitwise_or(r1, r2)
        return cv2.countNonZero(red) / float(max(1, red.size))

    @staticmethod
    def normalize_crop(crop, size=128):
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return None
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = crop
        return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)

    def classify_blue(self, crop):
        """Separate CROSSWALK / BUS / TUNNEL from bright-blue signs.

        CROSSWALK is detected first from its large white triangular field.
        BUS and TUNNEL are then analysed on a deeper central crop so the outer
        white sign frame does not dominate the pictogram geometry.

        BUS cues:
          - several enclosed dark holes/windows inside the white vehicle glyph,
          - strong horizontal structure,
          - centre is not hollow relative to the side pillars.

        TUNNEL cues:
          - very few enclosed holes,
          - white side pillars stronger than the centre, producing an arch/hollow
            signature.

        Ambiguous blue signs remain blue_info instead of being forced into BUS
        or TUNNEL.
        """
        norm = self.normalize_crop(crop)
        if norm is None:
            return 'blue_info', 0.0, 'invalid crop'

        n = norm.shape[0]

        # -------------------------------------------------------------
        # CROSSWALK on a shallower crop: keep the large white triangle.
        # -------------------------------------------------------------
        tri_pad = int(0.12 * n)
        tri_inner = norm[tri_pad:n - tri_pad, tri_pad:n - tri_pad]
        tri_white = self.white_mask(tri_inner)
        tri_white = cv2.morphologyEx(
            tri_white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        tri_white = cv2.morphologyEx(
            tri_white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        tri_area_total = float(max(1, tri_white.shape[0] * tri_white.shape[1]))
        tri_infos = []
        for c in find_external_contours(tri_white):
            area = float(cv2.contourArea(c))
            if area < 0.0035 * tri_area_total:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.035 * peri, True) if peri > 0 else []
            x, y, w, h = cv2.boundingRect(c)
            tri_infos.append((area, len(approx), w, h, c))
        tri_infos.sort(reverse=True, key=lambda item: item[0])

        for area, vertices, w, h, _ in tri_infos[:4]:
            if vertices in (3, 4) and area / tri_area_total > 0.065:
                fill = area / float(max(1, w * h))
                if vertices == 3 or fill < 0.72:
                    return 'crosswalk', 0.78, f'white triangle v={vertices}'

        # -------------------------------------------------------------
        # BUS / TUNNEL: crop further inward to remove the white plate frame.
        # The old 12% crop often saw the square frame as one giant component,
        # which made a real BUS look square and therefore ambiguous.
        # -------------------------------------------------------------
        pad = int(0.20 * n)
        inner = norm[pad:n - pad, pad:n - pad]
        white = self.white_mask(inner)
        white = cv2.morphologyEx(
            white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        white = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        area_total = float(max(1, white.shape[0] * white.shape[1]))
        infos = []
        for c in find_external_contours(white):
            area = float(cv2.contourArea(c))
            if area < 0.0035 * area_total:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.035 * peri, True) if peri > 0 else []
            x, y, w, h = cv2.boundingRect(c)
            infos.append((area, len(approx), w, h, c))
        infos.sort(reverse=True, key=lambda item: item[0])

        if not infos:
            return 'blue_info', 0.0, 'no inner white pictogram'

        ys, xs = np.nonzero(white)
        if len(xs) < 8:
            return 'blue_info', 0.0, 'too few white pixels'

        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        symbol_w = max(1, bx1 - bx0)
        symbol_h = max(1, by1 - by0)
        symbol_aspect = symbol_w / float(symbol_h)

        _area, _vertices, lw, lh, _ = infos[0]
        largest_aspect = lw / float(max(1, lh))
        component_count = len(infos)

        # Analyse only the pictogram bounding box, not the empty crop margin.
        symbol = white[by0:by1, bx0:bx1]
        binary = (symbol > 0).astype(np.float32)
        fill_ratio = float(binary.mean()) if binary.size else 0.0
        row_peak = float(binary.mean(axis=1).max()) if binary.size else 0.0
        col_peak = float(binary.mean(axis=0).max()) if binary.size else 0.0
        horizontal_strength = row_peak / max(1e-6, col_peak)

        H, W = binary.shape
        left = binary[:, :max(1, int(0.28 * W))]
        center = binary[:, int(0.34 * W):max(int(0.66 * W), int(0.34 * W) + 1)]
        right = binary[:, int(0.72 * W):]
        left_density = float(left.mean()) if left.size else 0.0
        right_density = float(right.mean()) if right.size else 0.0
        side_density = 0.5 * (left_density + right_density)
        center_density = float(center.mean()) if center.size else 0.0
        arch_hollow_ratio = side_density / max(1e-4, center_density)

        # BUS windows/wheel cut-outs appear as enclosed dark holes inside the
        # connected white vehicle glyph. Count only meaningful holes so pixel
        # noise does not become a BUS vote.
        hole_count = 0
        hole_areas = []
        hole_contours, hierarchy = cv2.findContours(
            symbol, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for idx, c in enumerate(hole_contours):
                if hierarchy[0][idx][3] < 0:
                    continue
                hole_area = float(cv2.contourArea(c))
                if hole_area >= 0.0035 * float(max(1, symbol.size)):
                    hole_count += 1
                    hole_areas.append(hole_area)

        metrics = (
            f'ar={symbol_aspect:.2f} lar={largest_aspect:.2f} '
            f'comp={component_count} holes={hole_count} '
            f'fill={fill_ratio:.2f} horiz={horizontal_strength:.2f} '
            f'hollow={arch_hollow_ratio:.2f}'
        )

        # BUS: windows/holes are the strongest cue. The second clause catches
        # slightly blurred BUS symbols whose windows partly merge.
        bus_geometry = (
            (hole_count >= 2 and
             horizontal_strength >= 1.02 and
             arch_hollow_ratio <= 1.12)
            or
            (horizontal_strength >= 1.14 and
             fill_ratio >= 0.38 and
             arch_hollow_ratio <= 1.00 and
             symbol_aspect >= 0.92)
        )

        # TUNNEL: keep this deliberately separate from BUS. A tunnel should
        # have a hollow centre/arch and very few enclosed window-like holes.
        tunnel_geometry = (
            hole_count <= 1 and
            arch_hollow_ratio >= 1.13 and
            side_density > center_density and
            horizontal_strength <= 1.35
        )

        if bus_geometry and not tunnel_geometry:
            confidence = clamp(
                0.68 + 0.035 * min(hole_count, 4) +
                0.04 * min(max(horizontal_strength - 1.0, 0.0), 1.0),
                0.0, 0.88)
            return 'bus', confidence, f'bus windows/vehicle {metrics}'

        if tunnel_geometry and not bus_geometry:
            confidence = clamp(
                0.68 + 0.10 * min(max(arch_hollow_ratio - 1.0, 0.0), 1.0),
                0.0, 0.88)
            return 'tunnel', confidence, f'tunnel arch {metrics}'

        # Conservative fallbacks: still do not force every blue sign into one
        # of the two similar classes.
        if hole_count >= 3 and arch_hollow_ratio < 1.18:
            return 'bus', 0.68, f'bus multi-hole fallback {metrics}'

        if (hole_count <= 1 and arch_hollow_ratio >= 1.25 and
                center_density < side_density):
            return 'tunnel', 0.68, f'tunnel hollow fallback {metrics}'

        return 'blue_info', 0.0, f'ambiguous blue {metrics}'

    def classify_red(self, crop, contour):
        norm = self.normalize_crop(crop)
        if norm is None:
            return 'unknown', 0.0, 'invalid crop'

        rr = self.red_ratio(norm)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.035 * peri, True) if peri > 0 else []
        vertices = len(approx)

        # STOP: red-filled octagonal plate. Warning signs are red borders around
        # a mostly white triangular field.
        if (6 <= vertices <= 10 and rr >= 0.15) or rr >= 0.30:
            return (
                'stop',
                0.90 if 7 <= vertices <= 9 else 0.80,
                f'red filled v={vertices} rr={rr:.2f}',
            )

        n = norm.shape[0]
        p = int(0.22 * n)
        inner = norm[p:n - p, p:n - p]
        dark = self.dark_mask(inner)
        dark = cv2.morphologyEx(
            dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        extrema = count_curve_extrema(dark)

        if extrema >= self.uneven_extrema_threshold:
            return 'ramp', 0.62, f'wave extrema={extrema}'
        return 'uneven', 0.62, f'wave extrema={extrema}'

    def classify_green(self, crop):
        # HW_EXIT has the red diagonal slash; HW_ENTRY does not.
        rr = self.red_ratio(crop)
        if rr >= 0.012:
            return 'hw_exit', 0.86, f'red slash rr={rr:.3f}'
        return 'hw_entry', 0.80, f'no red slash rr={rr:.3f}'

    def classify(self, image, candidate):
        x, y, w, h = self.expand_box(candidate['box'], image.shape, frac=0.10)
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return 'unknown', 0.0, 'empty', (x, y, w, h)

        family = candidate['family']
        if family == 'red':
            label, conf, why = self.classify_red(crop, candidate['contour'])
        elif family == 'green':
            label, conf, why = self.classify_green(crop)
        elif family == 'blue':
            label, conf, why = self.classify_blue(crop)
        else:
            label, conf, why = 'unknown', 0.0, 'unknown family'
        return label, conf, why, (x, y, w, h)

    def traffic_light_masks(self, image):
        """HSV masks for illuminated lamp colours, kept from starter_node."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 120, 110), (11, 255, 255)),
            cv2.inRange(hsv, (169, 120, 110), (180, 255, 255)),
        )
        yellow = cv2.inRange(
            hsv,
            (15, self.yellow_s_min, self.yellow_v_min),
            (43, 255, 255),
        )
        green = cv2.inRange(
            hsv,
            (35, self.green_s_min, self.green_v_min),
            (100, 255, 255),
        )
        kernel = np.ones((3, 3), np.uint8)
        return {
            'RED': cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel),
            'YELLOW': cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel),
            'GREEN': cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel),
        }

    def detect_traffic_light(self, image):
        """Find the best small bright roughly-circular lamp candidate.

        This intentionally preserves the starter_node detector: circularity plus
        dark housing around the coloured blob is used to reject coloured signs.
        """
        H, W = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 150, 100))
        best = None

        for state, mask in self.traffic_light_masks(image).items():
            for c in find_external_contours(mask):
                area = float(cv2.contourArea(c))
                if area < self.light_min_area or area > self.light_max_area:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                if w < 4 or h < 4:
                    continue
                aspect = w / float(max(h, 1))
                if aspect < 0.55 or aspect > 1.75:
                    continue

                peri = cv2.arcLength(c, True)
                if peri <= 1e-6:
                    continue
                circularity = 4.0 * math.pi * area / (peri * peri)
                if circularity < 0.38:
                    continue

                pad_x = max(6, int(round(w * 1.5)))
                pad_y = max(8, int(round(h * 2.0)))
                x0 = max(0, x - pad_x)
                y0 = max(0, y - pad_y)
                x1 = min(W, x + w + pad_x)
                y1 = min(H, y + h + pad_y)
                roi_dark = dark[y0:y1, x0:x1]
                dark_ratio = cv2.countNonZero(roi_dark) / float(max(1, roi_dark.size))
                if dark_ratio < 0.02:
                    continue

                score = circularity + 1.4 * dark_ratio + min(area / 250.0, 0.6)
                if y + h * 0.5 < H * 0.72:
                    score += 0.12

                if best is None or score > best['score']:
                    best = {
                        'state': state,
                        'box': (x, y, w, h),
                        'score': float(score),
                    }

        if best is None:
            return 'UNKNOWN', None, 0.0

        confidence = clamp(0.35 + 0.28 * best['score'], 0.0, 0.99)
        return best['state'], best['box'], confidence

    def reset_tunnel_confirmation(self):
        self.tunnel_candidate_since = None
        self.tunnel_candidate_box = None
        self.tunnel_candidate_confidence = 0.0

    def confirm_tunnel_for_output(self, signs, now):
        """BUS is immediate; hide only TUNNEL until it is continuous long enough."""
        if not signs:
            self.reset_tunnel_confirmation()
            return []

        det = signs[0]
        label = det.get('label')

        # BUS and every non-tunnel sign pass through immediately. The BUS-specific
        # far-distance size gate already ran inside select_one_right_sign().
        if label != 'tunnel':
            self.reset_tunnel_confirmation()
            return signs

        if self.tunnel_candidate_since is None:
            self.tunnel_candidate_since = now
            self.tunnel_candidate_box = det.get('box')
            self.tunnel_candidate_confidence = float(det.get('confidence', 0.0))
            return []

        self.tunnel_candidate_box = det.get('box')
        self.tunnel_candidate_confidence = float(det.get('confidence', 0.0))

        if now - self.tunnel_candidate_since < self.tunnel_confirm_sec:
            return []

        return signs

    @staticmethod
    def pedestrian_mask(image):
        """Mask the simulator pedestrian's pink/magenta body."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Same broad ranges used by starter_node.py: lighting tolerant while
        # staying away from the red STOP-sign family.
        m1 = cv2.inRange(hsv, (135, 65, 55), (168, 255, 255))
        m2 = cv2.inRange(hsv, (168, 55, 70), (178, 220, 255))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def detect_pedestrians(self, image):
        """Detect upright pink people and mark only the forward camera zone."""
        H, W = image.shape[:2]
        mask = self.pedestrian_mask(image)
        detections = []
        for c in find_external_contours(mask):
            area = float(cv2.contourArea(c))
            if area < self.pedestrian_min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 6 or h < 10:
                continue
            if h / float(max(w, 1)) < 0.85:
                continue
            fill = area / float(max(1, w * h))
            if fill < 0.12:
                continue
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            in_zone = (
                self.ped_zone_left * W <= cx <= self.ped_zone_right * W and
                self.ped_zone_top * H <= cy <= self.ped_zone_bottom * H)
            confidence = clamp(
                0.45 + min(area / 900.0, 0.25) + min(fill, 0.25),
                0.0, 0.95)
            detections.append({
                'kind': 'pedestrian',
                'label': 'PERSON',
                'box': (x, y, w, h),
                'area': area,
                'confidence': float(confidence),
                'in_zone': bool(in_zone),
                'centre': (cx, cy),
            })
        return detections

    def reset_transient_state(self):
        """Clear visible/candidate detections when camera perception is paused."""
        self.reset_tunnel_confirmation()
        self.sign_encounter_active = False
        self.sign_missing_frames = 0
        self.last_sign_event_label = None
        self.last_sign_event_box = None
        self.last_detections = []
        self.last_light_state = 'UNKNOWN'
        self.last_light_box = None
        self.last_pedestrians = []
        self.pedestrian_in_zone = False
        self.pedestrian_detection_enabled = False
        self.last_observation = {
            'sign': None,
            'sign_event_id': self.sign_event_id,
            'new_sign_event': False,
            'light': {'state': 'UNKNOWN', 'box': None, 'confidence': 0.0},
            'pedestrians': [],
            'pedestrian_in_zone': False,
        }

    def process(self, image, detect_pedestrians=False):
        """Update sign/light observations and optionally the pink pedestrian."""
        now = time.monotonic()
        raw_signs = []
        candidates = self.candidate_boxes(image, self.color_masks(image))
        for candidate in candidates:
            label, confidence, reason, box = self.classify(image, candidate)
            if label not in self.SIGN_NAMES:
                continue
            if candidate['area'] < self.min_area and label not in ('bus', 'tunnel'):
                continue  # Original area threshold for CROSSWALK, etc.
            raw_signs.append({
                'kind': 'sign',
                'label': label,
                'confidence': float(confidence),
                'reason': reason,
                'box': box,
            })

        selected = self.select_one_right_sign(
            self.suppress_overlaps(raw_signs), image.shape)
        # BUS and TUNNEL are detection only. BUS is immediate; TUNNEL
        # requires 1.0 s of uninterrupted observation.
        signs = self.confirm_tunnel_for_output(selected, now)
        new_sign_event = self.update_sign_encounter(signs)

        light_state, light_box, light_confidence = self.detect_traffic_light(image)
        if light_state != self.last_light_state:
            self.get_logger().info(
                f'LIGHT observed: {light_state} conf={light_confidence:.2f}')

        self.pedestrian_detection_enabled = bool(detect_pedestrians)
        pedestrians = (
            self.detect_pedestrians(image) if self.pedestrian_detection_enabled else [])
        person_in_zone = any(p.get('in_zone', False) for p in pedestrians)
        if (self.pedestrian_detection_enabled and
                person_in_zone != self.pedestrian_in_zone):
            self.get_logger().info(
                f'PEDESTRIAN observed in forward zone: {person_in_zone}')

        self.last_detections = signs
        self.last_light_state = light_state
        self.last_light_box = light_box
        self.last_pedestrians = pedestrians
        self.pedestrian_in_zone = bool(person_in_zone)
        self.last_observation = {
            'sign': (
                {
                    'label': signs[0]['label'],
                    'confidence': signs[0]['confidence'],
                    'box': signs[0]['box'],
                }
                if signs else None
            ),
            'sign_event_id': self.sign_event_id,
            'new_sign_event': bool(new_sign_event),
            'light': {
                'state': light_state,
                'box': light_box,
                'confidence': float(light_confidence),
            },
            'pedestrians': [
                {
                    'box': ped['box'],
                    'confidence': ped['confidence'],
                    'in_zone': ped['in_zone'],
                }
                for ped in pedestrians
            ],
            'pedestrian_in_zone': bool(person_in_zone),
        }
        return self.last_observation

    def render_overlay(self, image):
        annotated = image.copy()
        height, width = annotated.shape[:2]

        if self.show_sign_roi:
            x0 = int(self.sign_right_start_ratio * width)
            y0 = int(self.sign_top_ratio * height)
            y1 = int(self.sign_bottom_ratio * height)
            cv2.rectangle(
                annotated, (x0, y0), (width - 1, y1), (120, 120, 120), 1)

        sign_label = (
            self.last_detections[0]['label'] if self.last_detections else 'none')
        light_conf = float(
            self.last_observation.get('light', {}).get('confidence', 0.0))

        if self.pedestrian_detection_enabled:
            cv2.rectangle(
                annotated,
                (int(self.ped_zone_left * width), int(self.ped_zone_top * height)),
                (int(self.ped_zone_right * width), int(self.ped_zone_bottom * height)),
                (255, 0, 255), 2)

        panel_top = max(0, height - 79)
        cv2.rectangle(
            annotated, (0, panel_top), (min(width, 410), height), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            f'SIGN: {sign_label}  event #{self.sign_event_id}',
            (8, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f'LIGHT RAW: {self.last_light_state}  conf={light_conf:.2f}',
            (8, height - 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
        ped_label = (
            f'ON  IN ZONE={"YES" if self.pedestrian_in_zone else "NO"}'
            if self.pedestrian_detection_enabled else 'OFF')
        cv2.putText(
            annotated, f'PED RAW: {ped_label}', (8, height - 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)

        display = list(self.last_detections)
        if self.last_light_box is not None:
            display.append({
                'kind': 'light',
                'label': self.last_light_state,
                'confidence': light_conf,
                'box': self.last_light_box,
            })
        display.extend(self.last_pedestrians)

        for item in self.suppress_overlaps(display):
            x, y, w, h = item['box']
            kind = item.get('kind', 'sign')
            color = (
                (0, 255, 255) if kind == 'sign' else
                (255, 255, 255) if kind == 'light' else
                (255, 0, 255))
            label = f'{item["label"]} {item["confidence"]:.2f}'
            if kind == 'pedestrian' and item.get('in_zone'):
                label += ' IN ZONE'
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated, label, (x, max(18, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        return annotated


# =====================================================================
# Main node
# =====================================================================

class Starter(Node):

    def __init__(self):

        super().__init__(
            'crc_starter'
        )

        # =============================================================
        # BASIC CONTROL
        # =============================================================

        self.declare_parameter(
            'lane_only_mode',
            False
        )
        _lane_only_val = self.get_parameter('lane_only_mode').value
        if isinstance(_lane_only_val, str):
            self.lane_only_mode = _lane_only_val.lower() in ('true', '1', 'yes')
        else:
            self.lane_only_mode = bool(_lane_only_val)

        self.declare_parameter(
            'lane_debug_fps',
            4.0
        )
        self.lane_debug_fps = float(
            self.get_parameter('lane_debug_fps').value
        )

        self.declare_parameter(
            'max_speed',
            0.1
        )

        self.declare_parameter(
            'max_turn',
            0.85
        )

        self.declare_parameter(
            'rate',
            20.0
        )

        self.declare_parameter(
            'target_shift_px',
            0.0
        )

        self.declare_parameter(
            'kp_lateral',
            0.58
        )

        self.declare_parameter(
            'kp_heading',
            0.28
        )

        self.declare_parameter(
            'kp_preview',
            0.30
        )

        self.declare_parameter(
            'steer_alpha',
            0.24
        )

        self.declare_parameter(
            'preview_deadband_px',
            18.0
        )

        self.declare_parameter(
            'preview_turn_limit',
            0.32
        )

        # =============================================================
        # CAMERA / MASK
        # =============================================================

        self.declare_parameter(
            'white_v_min',
            120
        )

        self.declare_parameter(
            'white_s_max',
            120
        )

        self.declare_parameter(
            'roi_top_ratio',
            0.43
        )

        self.declare_parameter(
            'near_y_ratio',
            0.84
        )

        self.declare_parameter(
            'mid_y_ratio',
            0.69
        )

        self.declare_parameter(
            'lookahead_y_ratio',
            0.72
        )

        self.declare_parameter(
            'far_y_ratio',
            0.50
        )

        # =============================================================
        # HOUGH / VECTOR
        # =============================================================

        self.declare_parameter(
            'hough_threshold',
            22
        )

        self.declare_parameter(
            'min_line_length',
            28
        )

        self.declare_parameter(
            'left_min_line_length',
            15
        )

        self.declare_parameter(
            'max_line_gap',
            18
        )

        self.declare_parameter(
            'min_vertical_span_px',
            14.0
        )

        self.declare_parameter(
            'max_lane_tilt_deg',
            78.0
        )

        self.declare_parameter(
            'seed_max_y_extrap_px',
            95.0
        )

        # =============================================================
        # LANE THICKNESS
        # =============================================================

        self.declare_parameter(
            'min_lane_thickness_px',
            8.0
        )

        self.declare_parameter(
            'max_lane_thickness_px',
            40.0
        )

        self.declare_parameter(
            'thickness_ratio_min',
            0.3
        )

        self.declare_parameter(
            'thickness_ratio_max',
            2.8
        )

        self.declare_parameter(
            'thickness_alpha',
            0.10
        )

        self.declare_parameter(
            'thickness_scan_px',
            28
        )

        self.declare_parameter(
            'thickness_samples',
            7
        )

        self.declare_parameter(
            'thickness_update_ratio_min',
            0.70
        )

        self.declare_parameter(
            'thickness_update_ratio_max',
            1.35
        )

        # =============================================================
        # CONNECTED VECTOR GRAPH
        # =============================================================

        self.declare_parameter(
            'chain_connect_px',
            48.0
        )

        self.declare_parameter(
            'chain_max_turn_deg',
            76.0
        )

        self.declare_parameter(
            'chain_min_progress_px',
            5.0
        )

        self.declare_parameter(
            'chain_max_segments',
            7
        )

        self.declare_parameter(
            'sample_max_y_gap_px',
            25.0
        )

        # =============================================================
        # TEMPORAL TRACKING
        # =============================================================

        self.declare_parameter(
            'acquire_frames',
            4
        )

        self.declare_parameter(
            'tracking_corridor_px',
            82.0
        )

        self.declare_parameter(
            'tracking_angle_deg',
            58.0
        )

        self.declare_parameter(
            'max_near_jump_px',
            58.0
        )

        self.declare_parameter(
            'max_path_angle_jump_deg',
            62.0
        )

        self.declare_parameter(
            'lane_alpha',
            0.40
        )

        self.declare_parameter(
            'angle_alpha',
            0.38
        )

        # =============================================================
        # TRANSVERSE / CROSSWALK
        # =============================================================

        self.declare_parameter(
            'transverse_relative_angle_deg',
            48.0
        )

        self.declare_parameter(
            'transverse_min_length',
            48.0
        )

        self.declare_parameter(
            'transverse_distance_px',
            34.0
        )

        self.declare_parameter(
            'transverse_hold_frames',
            8
        )

        # =============================================================
        # LOST RECOVERY
        # =============================================================

        self.declare_parameter(
            'short_lost_frames',
            3
        )

        # Hold the yaw of the last valid lane for this many seconds before
        # allowing LOST recovery to rotate toward the nearest cardinal anchor.
        # Applies identically after tracking either the right or left boundary.
        self.declare_parameter(
            'lost_anchor_delay_sec',
            3.0
        )

        self.declare_parameter(
            'curve_hold_frames',
            10
        )

        self.declare_parameter(
            'curve_turn_threshold',
            0.16
        )

        self.declare_parameter(
            'lost_heading_kp',
            1.30
        )

        self.declare_parameter(
            'lost_turn_limit',
            0.48
        )

        self.declare_parameter(
            'lost_creep_speed',
            0.028
        )

        self.declare_parameter(
            'lost_rotate_tolerance_deg',
            8.0
        )

        # After this many lost frames:
        # switch permanently to "IMU straight + seek".
        #
        # IMPORTANT:
        # no longer means STOP.
        self.declare_parameter(
            'long_lost_frames',
            90
        )

        # Keep the existing recovery behaviour for this many seconds
        # after entering LONG LOST.  After the delay, recovery may use
        # either outer boundary; the right side has priority if both exist.
        self.declare_parameter(
            'long_lost_dual_seek_delay_sec',
            15.0
        )

        # Once SEEK BOTH is entered, keep that recovery mode alive across
        # temporary reacquisition/loss cycles for this many seconds.
        # During the window, a new LOST still gets the normal 3 s yaw hold;
        # after that hold it goes back to SEEK BOTH instead of a cardinal anchor.
        self.declare_parameter(
            'seek_both_window_sec',
            15.0
        )

        # If SEEK BOTH cannot establish a stable lock on either boundary for
        # this long, rotate left by seek_both_left_turn_deg and then recover
        # using the RIGHT boundary only.
        self.declare_parameter(
            'seek_both_fail_timeout_sec',
            8.0
        )

        self.declare_parameter(
            'seek_both_left_turn_deg',
            90.0
        )

        # If LONG LOST recovers the left boundary first, use it only as a
        # temporary proxy for the missing right boundary.  A +200 px virtual
        # shift means the left boundary is controlled against
        # (right_target_x_near - 200 px).
        self.declare_parameter(
            'left_recovery_offset_px',
            230.0
        )

        # Long-lost straight speed.
        self.declare_parameter(
            'long_lost_speed',
            0.030
        )

        # If yaw error is larger than this,
        # move very slowly while correcting.
        self.declare_parameter(
            'long_lost_yaw_tolerance_deg',
            10.0
        )

        self.declare_parameter(
            'long_lost_correct_speed',
            0.015
        )

        self.declare_parameter(
            'long_lost_turn_limit',
            0.30
        )

        self.declare_parameter(
            'reacquire_frames',
            3
        )

        # =============================================================
        # LIDAR / RAMP
        # =============================================================

        self.declare_parameter(
            'stop_distance',
            0.3
        )

        self.declare_parameter(
            'ramp_stop_distance',
            0.3
        )

        self.declare_parameter(
            'front_sector_deg',
            10.0
        )

        self.declare_parameter(
            'front_percentile',
            25.0
        )

        self.declare_parameter(
            'obstacle_confirm_frames',
            5
        )

        # Right-lane obstacle bypass: make a rectangular detour to the LEFT,
        # pass the obstacle, return to the original lane, then seek RIGHT again.
        self.declare_parameter(
            'obstacle_bypass_lateral_m',
            0.42
        )

        self.declare_parameter(
            'obstacle_bypass_forward_m',
            0.75
        )

        self.declare_parameter(
            'obstacle_bypass_speed',
            0.080
        )

        self.declare_parameter(
            'obstacle_bypass_turn_kp',
            1.40
        )

        self.declare_parameter(
            'obstacle_bypass_turn_limit',
            0.45
        )

        self.declare_parameter(
            'obstacle_bypass_turn_tolerance_deg',
            6.0
        )

        self.declare_parameter(
            'ramp_pitch_deg',
            3.0
        )

        # =============================================================
        # OPTIONAL RIGHT WALL
        # =============================================================

        self.declare_parameter(
            'right_wall_gain',
            0.0
        )

        self.declare_parameter(
            'right_wall_safe_distance',
            0.20
        )

        self.declare_parameter(
            'right_wall_angle_deg',
            -72.0
        )

        self.declare_parameter(
            'right_wall_sector_deg',
            18.0
        )

        # =============================================================
        # READ PARAMETERS
        # =============================================================

        gp = lambda name: self.get_parameter(
            name
        ).value

        self.max_speed = float(
            gp('max_speed')
        )

        self.max_turn = float(
            gp('max_turn')
        )

        self.rate = float(
            gp('rate')
        )

        self.target_shift_px = float(
            gp('target_shift_px')
        )

        self.kp_lateral = float(
            gp('kp_lateral')
        )

        self.kp_heading = float(
            gp('kp_heading')
        )

        self.kp_preview = float(
            gp('kp_preview')
        )

        self.steer_alpha = float(
            gp('steer_alpha')
        )

        self.preview_deadband_px = float(
            gp('preview_deadband_px')
        )

        self.preview_turn_limit = float(
            gp('preview_turn_limit')
        )

        self.white_v_min = int(
            gp('white_v_min')
        )

        self.white_s_max = int(
            gp('white_s_max')
        )

        self.roi_top_ratio = float(
            gp('roi_top_ratio')
        )

        self.near_y_ratio = float(
            gp('near_y_ratio')
        )

        self.mid_y_ratio = float(
            gp('mid_y_ratio')
        )

        self.lookahead_y_ratio = float(
            gp('lookahead_y_ratio')
        )

        self.far_y_ratio = float(
            gp('far_y_ratio')
        )

        self.hough_threshold = int(
            gp('hough_threshold')
        )

        self.min_line_length = int(
            gp('min_line_length')
        )

        self.left_min_line_length = int(
            gp('left_min_line_length')
        )

        self.max_line_gap = int(
            gp('max_line_gap')
        )

        self.min_vertical_span_px = float(
            gp('min_vertical_span_px')
        )

        self.max_lane_tilt_deg = float(
            gp('max_lane_tilt_deg')
        )

        self.seed_max_y_extrap_px = float(
            gp('seed_max_y_extrap_px')
        )

        self.min_lane_thickness_px = float(
            gp('min_lane_thickness_px')
        )

        self.max_lane_thickness_px = float(
            gp('max_lane_thickness_px')
        )

        self.thickness_ratio_min = float(
            gp('thickness_ratio_min')
        )

        self.thickness_ratio_max = float(
            gp('thickness_ratio_max')
        )

        self.thickness_alpha = float(
            gp('thickness_alpha')
        )

        self.thickness_scan_px = int(
            gp('thickness_scan_px')
        )

        self.thickness_samples = int(
            gp('thickness_samples')
        )

        self.thickness_update_ratio_min = float(
            gp('thickness_update_ratio_min')
        )

        self.thickness_update_ratio_max = float(
            gp('thickness_update_ratio_max')
        )

        self.chain_connect_px = float(
            gp('chain_connect_px')
        )

        self.chain_max_turn_deg = float(
            gp('chain_max_turn_deg')
        )

        self.chain_min_progress_px = float(
            gp('chain_min_progress_px')
        )

        self.chain_max_segments = int(
            gp('chain_max_segments')
        )

        self.sample_max_y_gap_px = float(
            gp('sample_max_y_gap_px')
        )

        self.acquire_frames = int(
            gp('acquire_frames')
        )

        self.tracking_corridor_px = float(
            gp('tracking_corridor_px')
        )

        self.tracking_angle_deg = float(
            gp('tracking_angle_deg')
        )

        self.max_near_jump_px = float(
            gp('max_near_jump_px')
        )

        self.max_path_angle_jump_deg = float(
            gp('max_path_angle_jump_deg')
        )

        self.lane_alpha = float(
            gp('lane_alpha')
        )

        self.angle_alpha = float(
            gp('angle_alpha')
        )

        self.transverse_relative_angle_deg = float(
            gp('transverse_relative_angle_deg')
        )

        self.transverse_min_length = float(
            gp('transverse_min_length')
        )

        self.transverse_distance_px = float(
            gp('transverse_distance_px')
        )

        self.transverse_hold_frames = int(
            gp('transverse_hold_frames')
        )

        self.short_lost_frames = int(
            gp('short_lost_frames')
        )

        self.lost_anchor_delay_sec = float(
            gp('lost_anchor_delay_sec')
        )

        self.curve_hold_frames = int(
            gp('curve_hold_frames')
        )

        self.curve_turn_threshold = float(
            gp('curve_turn_threshold')
        )

        self.lost_heading_kp = float(
            gp('lost_heading_kp')
        )

        self.lost_turn_limit = float(
            gp('lost_turn_limit')
        )

        self.lost_creep_speed = float(
            gp('lost_creep_speed')
        )

        self.lost_rotate_tolerance_deg = float(
            gp('lost_rotate_tolerance_deg')
        )

        self.long_lost_frames = int(
            gp('long_lost_frames')
        )

        self.long_lost_dual_seek_delay_sec = float(
            gp('long_lost_dual_seek_delay_sec')
        )

        self.seek_both_window_sec = float(
            gp('seek_both_window_sec')
        )

        self.seek_both_fail_timeout_sec = float(
            gp('seek_both_fail_timeout_sec')
        )

        self.seek_both_left_turn_deg = float(
            gp('seek_both_left_turn_deg')
        )

        self.left_recovery_offset_px = float(
            gp('left_recovery_offset_px')
        )

        self.long_lost_speed = float(
            gp('long_lost_speed')
        )

        self.long_lost_yaw_tolerance_deg = float(
            gp('long_lost_yaw_tolerance_deg')
        )

        self.long_lost_correct_speed = float(
            gp('long_lost_correct_speed')
        )

        self.long_lost_turn_limit = float(
            gp('long_lost_turn_limit')
        )

        self.reacquire_frames = int(
            gp('reacquire_frames')
        )

        self.stop_distance = float(
            gp('stop_distance')
        )

        self.ramp_stop_distance = float(
            gp('ramp_stop_distance')
        )

        self.front_sector_deg = float(
            gp('front_sector_deg')
        )

        self.front_percentile = float(
            gp('front_percentile')
        )

        self.obstacle_confirm_frames = int(
            gp('obstacle_confirm_frames')
        )

        self.obstacle_bypass_lateral_m = float(
            gp('obstacle_bypass_lateral_m')
        )

        self.obstacle_bypass_forward_m = float(
            gp('obstacle_bypass_forward_m')
        )

        self.obstacle_bypass_speed = float(
            gp('obstacle_bypass_speed')
        )

        self.obstacle_bypass_turn_kp = float(
            gp('obstacle_bypass_turn_kp')
        )

        self.obstacle_bypass_turn_limit = float(
            gp('obstacle_bypass_turn_limit')
        )

        self.obstacle_bypass_turn_tolerance_deg = float(
            gp('obstacle_bypass_turn_tolerance_deg')
        )

        self.ramp_pitch_deg = float(
            gp('ramp_pitch_deg')
        )

        self.right_wall_gain = float(
            gp('right_wall_gain')
        )

        self.right_wall_safe_distance = float(
            gp('right_wall_safe_distance')
        )

        self.right_wall_angle_deg = float(
            gp('right_wall_angle_deg')
        )

        self.right_wall_sector_deg = float(
            gp('right_wall_sector_deg')
        )

        # =============================================================
        # SENSOR STATE
        # =============================================================

        self.image = None
        self.scan = None

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        self.imu_yaw = None
        self.imu_pitch = 0.0

        # =============================================================
        # TRACK STATE
        # =============================================================

        self.lane_locked = False

        self.track_x_near = None
        self.track_x_mid = None
        self.track_x_look = None
        self.track_x_far = None

        self.track_path_angle_deg = None

        self.reference_path_angle_deg = None
        self.reference_preview_dx = None

        # Keep a mirrored calibration for the optional left-boundary mode.
        # The active reference_* variables remain the ones used by the
        # original controller; these stored copies are only swapped in when
        # long-lost recovery intentionally changes boundary side.
        self.right_reference_path_angle_deg = None
        self.left_reference_path_angle_deg = None
        self.right_reference_preview_dx = None
        self.left_reference_preview_dx = None

        self.target_x_near = None

        # The original controller is calibrated on the right boundary.
        # Keep that target unchanged and create its mirrored counterpart
        # only for the new long-lost left-boundary recovery path.
        self.right_target_x_near = None
        self.left_target_x_near = None
        self.tracked_lane_side = 'right'

        self.reference_lane_thickness = None

        self.last_good_yaw = None
        self.last_good_turn = 0.0

        self.last_chain = []

        # =============================================================
        # ACQUIRE / RECOVERY
        # =============================================================

        self.acquire_samples = []
        self.acquire_thickness_samples = []

        self.recovery_samples = []
        self.recovery_lane_side = None

        # Used only while a recovered left boundary is the temporary fallback.
        # The controller keeps following that left boundary while independently
        # accumulating a stable right-boundary reacquisition.
        self.right_return_samples = []

        self.lost_count = 0
        self.lost_cardinal_target = None

        # SEEK BOTH persistence state.
        # The window is intentionally independent from lost_count because
        # lost_count is reset as soon as a lane is reacquired.
        self.seek_both_started_at = None
        self.seek_both_until = None

        # Prevent the same uninterrupted LOST episode from immediately
        # starting a second SEEK BOTH window as soon as the first one expires.
        # It is re-armed only after a valid lane is tracked outside the window.
        self.seek_both_retrigger_blocked = False

        # Timeout state for the new fallback:
        # SEEK BOTH for 8 s without a stable lane lock -> LEFT 90 -> SEEK RIGHT.
        # The no-lock timer only runs while dual-side perception is actually
        # active; it is reset by any successful lane reacquisition.
        self.seek_both_no_lock_since = None
        self.seek_right_after_left_turn_active = False
        self.seek_right_turn_target_yaw = None
        self.seek_right_turn_complete = False

        self.transverse_counter = 0

        self.obstacle_count = 0

        # Obstacle bypass state machine.  It is entered only while RIGHT lane
        # tracking is active.  Geometry: LEFT 90 -> lateral move -> RIGHT 90
        # -> pass -> RIGHT 90 -> lateral return -> LEFT 90 -> SEEK RIGHT.
        self.obstacle_bypass_active = False
        self.obstacle_bypass_phase = None
        self.obstacle_bypass_reference_yaw = None
        self.obstacle_bypass_target_yaw = None
        self.obstacle_bypass_origin = None

        # Obstacle policy gate.  Before HW_ENTRY and after HW_EXIT, a front
        # obstacle is a pure stop-and-wait event.  Only inside the highway
        # section may the existing rectangular bypass state machine start.
        self.highway_mode = False
        self.highway_last_event_id = -1

        self.last_turn = 0.0

        self.last_log = {}

        # =============================================================
        # ROS
        # =============================================================

        self.bridge = CvBridge()

        self.pub_cmd = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.pub_debug = self.create_publisher(
            Image,
            '/lane_debug/image',
            10
        )

        self.pub_detection_debug = self.create_publisher(
            Image,
            '/detection_debug/image',
            10
        )

        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.on_image,
            qos_profile_sensor_data
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.on_odom,
            10
        )

        self.create_subscription(
            Imu,
            '/imu',
            self.on_imu,
            qos_profile_sensor_data
        )

        self.create_timer(
            1.0 / self.rate,
            self.tick
        )

        self.get_logger().info(
            (
                'V4.8 CARDINAL IMU LOST ready | '
                f'speed={self.max_speed:.3f} | '
                f'longLost={self.long_lost_frames} frames | '
                f'longSpeed={self.long_lost_speed:.3f}'
            )
        )

        # Separate camera pipelines:
        #   /lane_debug/image      = lane/controller only
        #   /detection_debug/image = STOP/RAMP/UNEVEN detector only
        self.last_camera_frame_time = 0.0
        self.last_camera_error = None
        self.next_camera_wait_log = 0.0
        self.next_perception_error_log = 0.0
        self.next_publish_error_log = 0.0
        self.next_detection_publish_error_log = 0.0
        self.last_lane_debug_publish_time = 0.0
        self.lane_status_active = False
        self.lane_debug_ready_logged = False
        self.detection_debug_ready_logged = False

        self.detection_image = None
        self.detection_header = None
        self.camera_frame_seq = 0
        self.last_detection_frame_seq = -1
        self.last_perception_update_time = 0.0

        self.sign_perception = SignPerception(self)
        self.declare_parameter('perception_rate_hz', 5.0)
        perception_hz = max(0.2, float(self.get_parameter('perception_rate_hz').value))
        self.perception_period_sec = 1.0 / perception_hz

        # STOP reaction: continuous detector observations for 1.5 s -> immediate
        # stop.  During the hold, controller/lane state is frozen, not reset.
        self.declare_parameter('stop_sign_confirm_sec', 1.0)
        self.declare_parameter('stop_sign_hold_sec', 2.0)
        self.stop_sign_confirm_sec = float(
            self.get_parameter('stop_sign_confirm_sec').value)
        self.stop_sign_hold_sec = float(
            self.get_parameter('stop_sign_hold_sec').value)
        self.stop_sign_seen_since = None
        self.stop_sign_seen_event_id = None
        self.stop_sign_handled_event_id = -1
        self.stop_freeze_active = False
        self.stop_freeze_until = 0.0
        self.stop_freeze_started_ros = None

        # Traffic light reaction: the SAME colour must remain above the strict
        # confidence threshold on fresh detector frames for 0.8 s.
        self.declare_parameter('traffic_light_min_conf', 0.86)
        self.declare_parameter('traffic_light_confirm_sec', 0.8)
        self.declare_parameter('green_light_speed', 0.20)
        self.traffic_light_min_conf = float(
            self.get_parameter('traffic_light_min_conf').value)
        self.traffic_light_confirm_sec = float(
            self.get_parameter('traffic_light_confirm_sec').value)
        self.green_light_speed = float(
            self.get_parameter('green_light_speed').value)
        self.traffic_light_candidate = None
        self.traffic_light_candidate_since = None
        self.traffic_light_confirmed_state = 'UNKNOWN'
        self.traffic_light_confirmed_confidence = 0.0
        self.traffic_light_stop_active = False
        self.traffic_light_completed = False
        self.traffic_light_stop_started_at = None

        # Highway-entry confirmation: a green object may momentarily be confused
        # with HW_ENTRY (for example a close traffic light).  Bypass permission is
        # therefore armed only after HW_ENTRY survives on consecutive fresh
        # perception frames for the full confirmation window.
        self.declare_parameter('highway_entry_confirm_sec', 0.20)
        self.highway_entry_confirm_sec = float(
            self.get_parameter('highway_entry_confirm_sec').value)
        self.highway_entry_candidate_since = None
        self.highway_entry_candidate_event_id = None
        self.highway_entry_candidate_confidence = 0.0

        # RAMP boost: a fresh accepted ramp sign arms one 2 s speed window.
        self.declare_parameter('ramp_boost_speed', 0.25)
        self.declare_parameter('ramp_boost_sec', 2.0)
        self.ramp_boost_speed = float(self.get_parameter('ramp_boost_speed').value)
        self.ramp_boost_sec = float(self.get_parameter('ramp_boost_sec').value)
        self.ramp_boost_until = 0.0
        self.ramp_boost_handled_event_id = -1

        # CROSSWALK reaction requested for the pedestrian-warning sign:
        # sign disappears -> 3.0 s normal lane follow -> camera pink-person
        # check corroborated by front LiDAR -> when clear, 1.5 s lane-following
        # boost at 0.20 m/s when front LiDAR is clear.
        self.declare_parameter('crosswalk_sign_min_conf', 0.62)
        self.declare_parameter('crosswalk_confirm_sec', 0.50)
        self.declare_parameter('crosswalk_after_sign_delay_sec', 3.0)
        self.declare_parameter('crosswalk_boost_sec', 1.5)
        self.declare_parameter('crosswalk_boost_speed', 0.20)
        self.crosswalk_sign_min_conf = float(
            self.get_parameter('crosswalk_sign_min_conf').value)
        self.crosswalk_confirm_sec = float(
            self.get_parameter('crosswalk_confirm_sec').value)
        self.crosswalk_after_sign_delay_sec = float(
            self.get_parameter('crosswalk_after_sign_delay_sec').value)
        self.crosswalk_boost_sec = float(
            self.get_parameter('crosswalk_boost_sec').value)
        self.crosswalk_boost_speed = float(
            self.get_parameter('crosswalk_boost_speed').value)
        self.crosswalk_candidate_since = None
        self.crosswalk_candidate_event_id = None
        self.crosswalk_phase = 'IDLE'
        self.crosswalk_sign_visible = False
        self.crosswalk_event_id = -1
        self.crosswalk_handled_event_id = -1
        self.crosswalk_delay_until = 0.0
        self.crosswalk_boost_until = 0.0
        self.crosswalk_watch_sample_ready = False
        self.crosswalk_last_person_sample_time = 0.0

        # Camera perception is fully suspended while the open-loop highway
        # obstacle bypass owns motion. Temporal confirmations are reset so no
        # TUNNEL/light/STOP timer can bridge across the bypass interval.
        self.detection_paused_for_bypass = False
        self.detection_pause_started = None

        if not self.lane_only_mode:
            self.create_timer(self.perception_period_sec, self.detection_tick)
        self.create_timer(1.0, self.detect_status_tick)

        if self.lane_only_mode:
            self.get_logger().warn(
                '======================================================\n'
                '*** LANE ONLY MODE ACTIVE ***\n'
                '- Sign & traffic light detection: DISABLED\n'
                '- Pedestrian / crosswalk vision: DISABLED\n'
                '- Obstacle bypass manoeuvre: DISABLED (Safe stop kept)\n'
                '- Detection debug image: DISABLED\n'
                f'- Lane debug image: THROTTLED to ~{self.lane_debug_fps:.1f} FPS\n'
                '- Full LiDAR obstacle safety: ACTIVE\n'
                '======================================================'
            )
        else:
            self.get_logger().info(
                f'Perception at {perception_hz:.1f} Hz | '
                'STOP/RAMP/UNEVEN/HW/BUS/TUNNEL/CROSSWALK + traffic light | right sign ROI | '
                'lane=/lane_debug/image | detection=/detection_debug/image')
            self.get_logger().info(
                f'LIGHT confirm: conf > {self.traffic_light_min_conf:.2f} for '
                f'{self.traffic_light_confirm_sec:.1f}s | RED/YELLOW=STOP, '
                f'GREEN={self.green_light_speed:.2f} m/s while green is still visible')
            self.get_logger().info(
                'BUS immediate (detect only) | TUNNEL 1.0 s (detect only) | detection PAUSED during bypass')
            self.get_logger().info(
                f'CROSSWALK: continuous {self.crosswalk_confirm_sec:.1f}s confirm -> '
                f'sign-lost +{self.crosswalk_after_sign_delay_sec:.1f}s -> '
                f'pink-person + LiDAR gate -> {self.crosswalk_boost_speed:.2f} m/s for '
                f'{self.crosswalk_boost_sec:.1f}s')
            self.get_logger().info(
                f'RAMP={self.ramp_boost_speed:.2f}m/s for {self.ramp_boost_sec:.1f}s | '
                f'BYPASS={self.obstacle_bypass_speed:.2f}m/s | '
                f'SEEK BOTH fails after {self.seek_both_fail_timeout_sec:.1f}s')
        self.get_logger().info(
            f'OBSTACLE policy: STOP by default -> HW_ENTRY continuous '
            f'{self.highway_entry_confirm_sec:.1f}s enables bypass -> '
            'HW_EXIT restores stop-and-wait')

    # =================================================================
    # Callbacks
    # =================================================================

    def on_image(self, msg):

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception as exc:
            now = time.monotonic()
            self.last_camera_error = str(exc)
            if now >= self.next_camera_wait_log:
                self.get_logger().warn(f'image conversion failed: {exc}')
                self.next_camera_wait_log = now + 5.0
            return

        # Lane control and sign detection receive independent snapshots of the
        # same legal robot camera.  Detection never paints on the lane frame.
        self.image = image
        if not self.lane_only_mode:
            self.detection_image = image.copy()
            self.detection_header = msg.header
        self.camera_frame_seq += 1

        now = time.monotonic()
        self.last_camera_frame_time = now
        self.last_camera_error = None

        # If control exits before its normal debug publish, keep lane debug alive
        # with a raw lane-camera preview only (never a detection overlay).
        if not self.lane_only_mode:
            if (self.lane_status_active or
                    now - self.last_lane_debug_publish_time > 0.3):
                self.publish_lane_debug_frame(image.copy(), msg.header)
        else:
            lane_debug_interval = 1.0 / max(0.5, self.lane_debug_fps)
            if self.lane_status_active and (now - self.last_lane_debug_publish_time >= lane_debug_interval):
                self.publish_lane_debug_frame(image.copy(), msg.header)

    def pause_detection_for_bypass(self, now):
        """Suspend all camera perception while the open-loop bypass is active."""
        if not self.detection_paused_for_bypass:
            self.detection_paused_for_bypass = True
            self.detection_pause_started = now
            self.sign_perception.reset_transient_state()
            self.stop_sign_seen_since = None
            self.stop_sign_seen_event_id = None
            self.traffic_light_candidate = None
            self.traffic_light_candidate_since = None
            self.reset_crosswalk_confirmation()
            self.get_logger().info(
                'DETECTION PAUSED: obstacle bypass owns motion')

    def resume_detection_after_bypass(self, now):
        if not self.detection_paused_for_bypass:
            return
        paused = 0.0
        if self.detection_pause_started is not None:
            paused = max(0.0, now - self.detection_pause_started)

        # Freeze crosswalk timing while perception itself is disabled.
        if self.crosswalk_phase == 'DELAY':
            self.crosswalk_delay_until += paused
        elif self.crosswalk_phase == 'BOOST':
            self.crosswalk_boost_until += paused

        self.detection_paused_for_bypass = False
        self.detection_pause_started = None
        self.get_logger().info(
            f'DETECTION RESUMED after bypass ({paused:.2f}s paused)')

    def advance_crosswalk_delay_for_detection(self, now):
        """Arm person detection only after the requested 3 s post-sign delay."""
        if (self.crosswalk_phase == 'DELAY' and
                now >= self.crosswalk_delay_until):
            self.crosswalk_phase = 'WATCH'
            self.crosswalk_watch_sample_ready = False
            self.crosswalk_last_person_sample_time = 0.0
            self.get_logger().info(
                'CROSSWALK delay done -> pink pedestrian detection ON')

    def reset_crosswalk_confirmation(self):
        """Break CROSSWALK temporal continuity on any non-crosswalk fresh frame."""
        self.crosswalk_candidate_since = None
        self.crosswalk_candidate_event_id = None

    def update_crosswalk_sign_state(self, observation, observed_at):
        """Accept CROSSWALK only after continuous detection for 1.2 s.

        Any fresh perception result that is not CROSSWALK resets the candidate
        timer.  After confirmation, the existing behaviour is preserved: wait
        until the sign disappears, follow the lane for 3 s, then enable the
        pink-pedestrian + LiDAR gate.
        """
        sign = observation.get('sign') if observation else None
        event_id = int(observation.get('sign_event_id', -1)) if observation else -1
        visible = bool(
            sign and sign.get('label') == 'crosswalk' and
            float(sign.get('confidence', 0.0)) >= self.crosswalk_sign_min_conf)

        if visible:
            # Never re-arm a physical sign that has already completed its
            # crosswalk sequence.
            if event_id == self.crosswalk_handled_event_id:
                self.reset_crosswalk_confirmation()
                return

            # Once confirmed, simply keep the sign marked visible until it
            # disappears.  The confirmation timer is no longer needed.
            if (self.crosswalk_phase == 'SIGN_VISIBLE' and
                    self.crosswalk_event_id == event_id):
                self.crosswalk_sign_visible = True
                return

            # A crosswalk sequence is already in progress, so do not create a
            # second sign candidate in DELAY/WATCH/BOOST.
            if self.crosswalk_phase != 'IDLE':
                self.reset_crosswalk_confirmation()
                return

            # First fresh sample of a new candidate starts the continuity timer.
            if (self.crosswalk_candidate_since is None or
                    self.crosswalk_candidate_event_id != event_id):
                self.crosswalk_candidate_since = observed_at
                self.crosswalk_candidate_event_id = event_id
                self.get_logger().info(
                    f'CROSSWALK candidate #{event_id} -> need '
                    f'{self.crosswalk_confirm_sec:.1f}s continuous')
                return

            elapsed = observed_at - self.crosswalk_candidate_since
            self.log_every(
                0.4,
                'crosswalk_confirm',
                f'CROSSWALK confirm {elapsed:.1f}/'
                f'{self.crosswalk_confirm_sec:.1f}s',
            )

            if elapsed < self.crosswalk_confirm_sec:
                return

            # Candidate survived continuously for the full window.
            self.reset_crosswalk_confirmation()
            self.crosswalk_phase = 'SIGN_VISIBLE'
            self.crosswalk_sign_visible = True
            self.crosswalk_event_id = event_id
            self.get_logger().info(
                f'CROSSWALK #{event_id} CONFIRMED for '
                f'{self.crosswalk_confirm_sec:.1f}s -> wait until sign disappears')
            return

        # Any fresh non-crosswalk sample breaks a not-yet-confirmed candidate.
        self.reset_crosswalk_confirmation()

        # A confirmed crosswalk sign has now disappeared: preserve the previous
        # 3 s lane-follow delay before the pedestrian detector is enabled.
        if (self.crosswalk_sign_visible and
                self.crosswalk_phase == 'SIGN_VISIBLE' and
                self.crosswalk_event_id != self.crosswalk_handled_event_id):
            self.crosswalk_sign_visible = False
            self.crosswalk_handled_event_id = self.crosswalk_event_id
            self.crosswalk_phase = 'DELAY'
            self.crosswalk_delay_until = (
                observed_at + self.crosswalk_after_sign_delay_sec)
            self.crosswalk_watch_sample_ready = False
            self.get_logger().info(
                f'CROSSWALK sign lost -> lane follow for '
                f'{self.crosswalk_after_sign_delay_sec:.1f}s before person check')
        else:
            self.crosswalk_sign_visible = False

    def crosswalk_pedestrian_detection_active(self):
        return self.crosswalk_phase in ('WATCH', 'BOOST')

    def crosswalk_person_sample_is_fresh(self):
        if not self.crosswalk_pedestrian_detection_active():
            return False
        freshness = max(0.6, 3.0 * self.perception_period_sec)
        return bool(
            self.crosswalk_watch_sample_ready and
            time.monotonic() - self.crosswalk_last_person_sample_time <= freshness)

    def crosswalk_person_obstacle_match(self, front, active_stop):
        """Require BOTH a fresh pink person and a front LiDAR obstacle."""
        return bool(
            self.crosswalk_person_sample_is_fresh() and
            self.sign_perception.pedestrian_in_zone and
            front < active_stop)

    def start_crosswalk_boost(self, now):
        self.crosswalk_phase = 'BOOST'
        self.crosswalk_boost_until = now + self.crosswalk_boost_sec
        self.get_logger().info(
            f'CROSSWALK clear -> lane-follow boost '
            f'{self.crosswalk_boost_speed:.2f} m/s for '
            f'{self.crosswalk_boost_sec:.1f}s')

    def interrupt_crosswalk_boost_for_person(self):
        if self.crosswalk_phase != 'BOOST':
            return
        self.crosswalk_phase = 'WATCH'
        self.crosswalk_boost_until = 0.0
        self.get_logger().info(
            'CROSSWALK pink person + LiDAR appeared during boost -> STOP / WATCH')

    def crosswalk_boost_is_active(self):
        if self.crosswalk_phase != 'BOOST':
            return False
        now = time.monotonic()
        if now < self.crosswalk_boost_until:
            return True
        self.crosswalk_phase = 'IDLE'
        self.crosswalk_boost_until = 0.0
        self.crosswalk_watch_sample_ready = False
        self.crosswalk_last_person_sample_time = 0.0
        self.get_logger().info('CROSSWALK boost done -> normal lane speed')
        return False

    def crosswalk_owns_front_obstacle(self):
        return self.crosswalk_phase in ('DELAY', 'WATCH', 'BOOST')

    def detection_tick(self):
        """Run camera perception on fresh frames, except during bypass."""
        if self.lane_only_mode:
            return

        if self.detection_image is None:
            return

        now = time.monotonic()

        # User-requested hard gate: no sign/light/person detection at all during
        # the highway obstacle-bypass manoeuvre. Consume frames silently so the
        # first post-bypass detection uses a genuinely fresh camera image.
        if self.obstacle_bypass_active:
            self.pause_detection_for_bypass(now)
            self.last_detection_frame_seq = self.camera_frame_seq
            return

        self.resume_detection_after_bypass(now)

        # Never count repeated processing of a frozen/stale image toward any
        # temporal confirmation window.
        if self.last_detection_frame_seq == self.camera_frame_seq:
            return

        frame = self.detection_image.copy()
        header = self.detection_header
        frame_seq = self.camera_frame_seq
        observed_at = time.monotonic()

        try:
            # If the 3 s delay has elapsed, enable the starter-node pink-person
            # detector on THIS frame before control is allowed to decide BOOST.
            self.advance_crosswalk_delay_for_detection(observed_at)
            ped_enabled = self.crosswalk_pedestrian_detection_active()
            observation = self.sign_perception.process(
                frame, detect_pedestrians=ped_enabled)
            self.last_perception_update_time = observed_at
            self.last_detection_frame_seq = frame_seq

            if ped_enabled:
                self.crosswalk_watch_sample_ready = True
                self.crosswalk_last_person_sample_time = observed_at

            self.update_stop_sign_confirmation(observation, observed_at)
            self.update_highway_mode(observation, observed_at)
            self.update_ramp_sign_boost(observation, observed_at)
            self.update_crosswalk_sign_state(observation, observed_at)
            self.update_traffic_light_confirmation(observation, observed_at)

            preview = self.sign_perception.render_overlay(frame)
            self.draw_detection_control_status(preview, observed_at)
            self.publish_detection_debug_frame(preview, header)

        except Exception as exc:
            self.last_detection_frame_seq = frame_seq
            err_now = time.monotonic()
            if err_now >= self.next_perception_error_log:
                self.get_logger().error(f'Camera sign detection failed: {exc}')
                self.next_perception_error_log = err_now + 3.0
            self.sign_perception.reset_transient_state()
            self.stop_sign_seen_since = None
            self.stop_sign_seen_event_id = None
            self.traffic_light_candidate = None
            self.traffic_light_candidate_since = None
            self.reset_crosswalk_confirmation()
            self.crosswalk_watch_sample_ready = False

    def update_stop_sign_confirmation(self, observation, observed_at):
        """Confirm STOP only from consecutive fresh detector observations."""
        sign = observation.get('sign') if observation else None
        event_id = int(observation.get('sign_event_id', -1)) if observation else -1
        visible_stop = bool(sign and sign.get('label') == 'stop')

        # The detector remains live during the hold, but the same encounter may
        # never trigger STOP twice.
        if self.stop_freeze_active:
            return

        if visible_stop and self.stop_sign_handled_event_id < 0:
            if (self.stop_sign_seen_since is None or
                    self.stop_sign_seen_event_id != event_id):
                self.stop_sign_seen_since = observed_at
                self.stop_sign_seen_event_id = event_id
                self.get_logger().info(
                    f'STOP visible -> confirm for {self.stop_sign_confirm_sec:.1f}s')
                return

            seen_time = observed_at - self.stop_sign_seen_since
            self.log_every(
                0.5,
                'stop_confirm',
                f'STOP confirm {seen_time:.1f}/{self.stop_sign_confirm_sec:.1f}s')

            if seen_time >= self.stop_sign_confirm_sec:
                self.stop_sign_handled_event_id = event_id
                self.stop_sign_seen_since = None
                self.stop_sign_seen_event_id = None
                self.begin_stop_freeze(event_id)
            return

        # Any fresh detector sample without STOP breaks continuity.
        self.stop_sign_seen_since = None
        self.stop_sign_seen_event_id = None

    def reset_highway_entry_confirmation(self):
        """Break HW_ENTRY temporal continuity on any non-entry fresh frame."""
        self.highway_entry_candidate_since = None
        self.highway_entry_candidate_event_id = None
        self.highway_entry_candidate_confidence = 0.0

    def update_highway_mode(self, observation, observed_at):
        """Require continuous HW_ENTRY for 0.6 s before enabling bypass.

        HW_EXIT keeps the previous event-based reaction.  Entry confirmation is
        deliberately frame-continuous: any fresh perception result that is not
        HW_ENTRY resets the timer, so a short close-up GREEN-light confusion
        cannot arm the obstacle bypass.
        """
        sign = observation.get('sign') if observation else None
        label = sign.get('label') if sign else None
        confidence = float(sign.get('confidence', 0.0)) if sign else 0.0
        event_id = int(observation.get('sign_event_id', -1)) if observation else -1

        # -------------------------------------------------------------
        # HW_ENTRY: must be the selected sign on every fresh perception
        # sample for the entire confirmation interval.
        # -------------------------------------------------------------
        if not self.highway_mode:
            if label != 'hw_entry':
                self.reset_highway_entry_confirmation()
                return

            if (self.highway_entry_candidate_since is None or
                    self.highway_entry_candidate_event_id != event_id):
                self.highway_entry_candidate_since = observed_at
                self.highway_entry_candidate_event_id = event_id
                self.highway_entry_candidate_confidence = confidence
                self.get_logger().info(
                    f'HW_ENTRY candidate #{event_id} conf={confidence:.2f} -> '
                    f'need {self.highway_entry_confirm_sec:.1f}s continuous')
                return

            self.highway_entry_candidate_confidence = confidence
            elapsed = observed_at - self.highway_entry_candidate_since
            self.log_every(
                0.4,
                'hw_entry_confirm',
                f'HW_ENTRY confirm {elapsed:.1f}/'
                f'{self.highway_entry_confirm_sec:.1f}s conf={confidence:.2f}',
            )

            if elapsed < self.highway_entry_confirm_sec:
                return

            self.highway_mode = True
            self.highway_last_event_id = event_id
            self.reset_highway_entry_confirmation()
            self.obstacle_count = 0
            self.get_logger().info(
                f'HW_ENTRY #{event_id} CONFIRMED for '
                f'{self.highway_entry_confirm_sec:.1f}s -> '
                'HIGHWAY MODE ON: obstacle bypass ENABLED')
            return

        # Once highway mode is active, no entry candidate should remain armed.
        self.reset_highway_entry_confirmation()

        # -------------------------------------------------------------
        # HW_EXIT: preserve the existing once-per-sign-event behaviour.
        # -------------------------------------------------------------
        if not observation or not observation.get('new_sign_event'):
            return
        if not sign or label != 'hw_exit':
            return
        if event_id == self.highway_last_event_id:
            return

        self.highway_mode = False
        self.highway_last_event_id = event_id
        self.obstacle_count = 0
        self.get_logger().info(
            f'HW_EXIT #{event_id} conf={confidence:.2f} -> '
            'HIGHWAY MODE OFF: obstacle STOP/WAIT restored')

    def update_ramp_sign_boost(self, observation, observed_at):
        """Arm the speed override once per accepted RAMP encounter."""
        sign = observation.get('sign') if observation else None
        if not sign or sign.get('label') != 'ramp':
            return
        event_id = int(observation.get('sign_event_id', -1))
        if event_id < 0 or event_id == self.ramp_boost_handled_event_id:
            return
        self.ramp_boost_handled_event_id = event_id
        self.ramp_boost_until = observed_at + self.ramp_boost_sec
        self.get_logger().info(
            f'RAMP #{event_id} -> {self.ramp_boost_speed:.2f} m/s '
            f'for {self.ramp_boost_sec:.1f}s while lane tracking')

    def ramp_boost_is_active(self):
        return time.monotonic() < self.ramp_boost_until

    def update_traffic_light_confirmation(self, observation, observed_at):
        """Confirm one lamp colour only after >threshold for a continuous window."""
        if self.traffic_light_completed or getattr(self, 'odom_x', 0.0) > 3.5:
            self.traffic_light_completed = True
            self.traffic_light_stop_active = False
            return

        if self.traffic_light_stop_active:
            if self.traffic_light_stop_started_at is None:
                self.traffic_light_stop_started_at = observed_at
            elif observed_at - self.traffic_light_stop_started_at >= 10.0:
                self.traffic_light_stop_active = False
                self.traffic_light_completed = True
                self.get_logger().info('LIGHT STOP TIMEOUT (10.0s) -> AUTO-RELEASE & RESUME CONTROLLER')
                return

        light = observation.get('light', {}) if observation else {}
        state = str(light.get('state', 'UNKNOWN'))
        confidence = float(light.get('confidence', 0.0))

        valid = (
            state in ('RED', 'YELLOW', 'GREEN')
            and confidence > self.traffic_light_min_conf
        )

        # Any fresh sample below/equal threshold, UNKNOWN, or wrong colour breaks
        # the 0.8 s continuity requirement.
        if not valid:
            self.traffic_light_candidate = None
            self.traffic_light_candidate_since = None
            return

        if self.traffic_light_candidate != state:
            self.traffic_light_candidate = state
            self.traffic_light_candidate_since = observed_at
            self.get_logger().info(
                f'LIGHT candidate {state} conf={confidence:.2f} -> '
                f'need {self.traffic_light_confirm_sec:.1f}s continuous')
            return

        if self.traffic_light_candidate_since is None:
            self.traffic_light_candidate_since = observed_at
            return

        elapsed = observed_at - self.traffic_light_candidate_since
        self.log_every(
            0.4,
            'light_confirm',
            f'LIGHT {state} confirm {elapsed:.1f}/'
            f'{self.traffic_light_confirm_sec:.1f}s conf={confidence:.2f}',
        )

        if elapsed < self.traffic_light_confirm_sec:
            return

        # Do not re-log/re-arm the same already-confirmed state.
        if self.traffic_light_confirmed_state == state:
            self.traffic_light_confirmed_confidence = confidence
            return

        self.traffic_light_confirmed_state = state
        self.traffic_light_confirmed_confidence = confidence

        if state in ('RED', 'YELLOW'):
            self.traffic_light_stop_active = True
            self.traffic_light_stop_started_at = observed_at
            self.stop()
            self.get_logger().info(
                f'LIGHT {state} CONFIRMED conf={confidence:.2f} -> STOP')
        else:
            self.traffic_light_stop_active = False
            self.traffic_light_completed = True
            self.get_logger().info(
                f'LIGHT GREEN CONFIRMED conf={confidence:.2f} -> GO / resume controller')

    def green_light_fast_is_active(self):
        """Use the 0.20 m/s lane-follow speed only for a freshly observed,
        already-confirmed GREEN lamp.  Once the lamp leaves view or confidence
        falls back to/below the strict threshold, normal lane speed resumes.
        """
        if self.traffic_light_confirmed_state != 'GREEN':
            return False

        if self.detection_paused_for_bypass:
            return False

        light = self.sign_perception.last_observation.get('light', {})
        state = str(light.get('state', 'UNKNOWN'))
        confidence = float(light.get('confidence', 0.0))
        fresh_for = time.monotonic() - self.last_perception_update_time
        freshness_limit = max(0.6, 2.5 * self.perception_period_sec)

        return bool(
            state == 'GREEN'
            and confidence > self.traffic_light_min_conf
            and fresh_for <= freshness_limit
        )

    def traffic_light_hold_is_active(self):
        return bool(self.traffic_light_stop_active)

    def begin_stop_freeze(self, event_id):
        """Stop immediately and freeze controller state for the hold interval."""
        if self.stop_freeze_active:
            return

        now = time.monotonic()
        self.stop_freeze_active = True
        self.stop_freeze_until = now + self.stop_sign_hold_sec
        self.stop_freeze_started_ros = (
            self.get_clock().now().nanoseconds * 1e-9)
        self.stop()
        self.get_logger().info(
            f'STOP #{event_id} CONFIRMED -> freeze controller state and hold '
            f'{self.stop_sign_hold_sec:.1f}s')

    def end_stop_freeze(self):
        """Resume exactly the pre-STOP controller state, including time windows."""
        if not self.stop_freeze_active:
            return

        ros_now = self.get_clock().now().nanoseconds * 1e-9
        frozen_sec = 0.0
        if self.stop_freeze_started_ros is not None:
            frozen_sec = max(0.0, ros_now - self.stop_freeze_started_ros)

        # Frame-counted LOST/REACQUIRE state never advanced during the hold.
        # Shift the few absolute ROS-time recovery windows so they are frozen too.
        for name in (
            'seek_both_started_at',
            'seek_both_until',
            'seek_both_no_lock_since',
        ):
            value = getattr(self, name, None)
            if value is not None:
                setattr(self, name, value + frozen_sec)

        self.stop_freeze_active = False
        self.stop_freeze_until = 0.0
        self.stop_freeze_started_ros = None
        self.get_logger().info(
            f'STOP HOLD DONE -> resume frozen controller state after '
            f'{frozen_sec:.2f}s pause')

    def stop_freeze_is_active(self):
        if not self.stop_freeze_active:
            return False
        if time.monotonic() >= self.stop_freeze_until:
            self.end_stop_freeze()
            return False
        return True

    def draw_detection_control_status(self, frame, now):
        if self.stop_freeze_active:
            left = max(0.0, self.stop_freeze_until - now)
            stop_label = f'STOP SIGN: HOLD {left:.1f}s'
        elif self.stop_sign_seen_since is not None:
            seen = max(0.0, now - self.stop_sign_seen_since)
            stop_label = (
                f'STOP SIGN: confirm {seen:.1f}/{self.stop_sign_confirm_sec:.1f}s')
        else:
            stop_label = 'STOP SIGN: idle'

        light = self.sign_perception.last_observation.get('light', {})
        raw_state = str(light.get('state', 'UNKNOWN'))
        raw_conf = float(light.get('confidence', 0.0))
        if (self.traffic_light_candidate is not None and
                self.traffic_light_candidate_since is not None):
            elapsed = max(0.0, now - self.traffic_light_candidate_since)
            candidate_label = (
                f'{self.traffic_light_candidate} '
                f'{elapsed:.1f}/{self.traffic_light_confirm_sec:.1f}s')
        else:
            candidate_label = 'none'

        if self.crosswalk_candidate_since is not None:
            cw_elapsed = max(0.0, now - self.crosswalk_candidate_since)
            cw_detail = (
                f'CONFIRM {cw_elapsed:.1f}/{self.crosswalk_confirm_sec:.1f}s')
        elif self.crosswalk_phase == 'DELAY':
            cw_detail = (
                f'DELAY {max(0.0, self.crosswalk_delay_until - now):.1f}s')
        elif self.crosswalk_phase == 'WATCH':
            cw_detail = (
                f'WATCH person={"YES" if self.sign_perception.pedestrian_in_zone else "NO"}')
        elif self.crosswalk_phase == 'BOOST':
            cw_detail = (
                f'BOOST {max(0.0, self.crosswalk_boost_until - now):.1f}s')
        else:
            cw_detail = self.crosswalk_phase

        if self.highway_mode:
            highway_detail = 'ON / BYPASS'
        elif self.highway_entry_candidate_since is not None:
            hw_elapsed = max(0.0, now - self.highway_entry_candidate_since)
            highway_detail = (
                f'CONFIRM {hw_elapsed:.1f}/{self.highway_entry_confirm_sec:.1f}s')
        else:
            highway_detail = 'OFF / OBSTACLE STOP'

        lines = (
            stop_label,
            f'HIGHWAY: {highway_detail}',
            f'CROSSWALK: {cw_detail}',
            f'LIGHT raw: {raw_state} conf={raw_conf:.2f}  candidate={candidate_label}',
            f'LIGHT ctrl: {self.traffic_light_confirmed_state}  '
            f'{"STOP" if self.traffic_light_stop_active else "RUN"}',
        )

        y = 22
        for label in lines:
            cv2.putText(
                frame, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 255, 255), 2, cv2.LINE_AA)
            y += 21

    def publish_lane_debug_frame(self, frame, header=None):
        if self.pub_debug.get_subscription_count() == 0:
            self.last_lane_debug_publish_time = time.monotonic()
            return True
        try:
            if self.lane_only_mode and frame.shape[1] > 480:
                frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if header is not None:
                msg.header = header
            self.pub_debug.publish(msg)
            self.last_lane_debug_publish_time = time.monotonic()
            self.lane_status_active = False
            if not self.lane_debug_ready_logged:
                self.get_logger().info('Lane debug publishing on /lane_debug/image')
                self.lane_debug_ready_logged = True
            return True
        except Exception as exc:
            now = time.monotonic()
            if now >= self.next_publish_error_log:
                self.get_logger().error(f'Cannot publish /lane_debug/image: {exc}')
                self.next_publish_error_log = now + 3.0
            return False

    def publish_detection_debug_frame(self, frame, header=None):
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            if header is not None:
                msg.header = header
            self.pub_detection_debug.publish(msg)
            if not self.detection_debug_ready_logged:
                self.get_logger().info(
                    'Detection debug publishing on /detection_debug/image')
                self.detection_debug_ready_logged = True
            return True
        except Exception as exc:
            now = time.monotonic()
            if now >= self.next_detection_publish_error_log:
                self.get_logger().error(
                    f'Cannot publish /detection_debug/image: {exc}')
                self.next_detection_publish_error_log = now + 3.0
            return False

    def detect_status_tick(self):
        """Show an explicit status image when the camera has stopped publishing."""
        now = time.monotonic()
        if now - self.last_camera_frame_time < 2.0:
            return

        status = ('CAMERA CONVERSION FAILED' if self.last_camera_error
                  else 'WAITING FOR /camera/image_raw')
        if now >= self.next_camera_wait_log:
            self.get_logger().warn(status)
            self.next_camera_wait_log = now + 5.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, status, (25, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Check simulator camera / sensor_check', (25, 244),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (230, 230, 230), 1, cv2.LINE_AA)
        if self.publish_lane_debug_frame(frame):
            self.lane_status_active = True

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.odom_x = float(p.x)
        self.odom_y = float(p.y)

        self.odom_yaw = math.atan2(
            2.0 * (
                q.w * q.z
                +
                q.x * q.y
            ),
            1.0
            -
            2.0 * (
                q.y * q.y
                +
                q.z * q.z
            )
        )

    def on_imu(self, msg):

        q = msg.orientation

        s = 2.0 * (
            q.w * q.y
            -
            q.z * q.x
        )

        s = clamp(
            s,
            -1.0,
            1.0
        )

        self.imu_pitch = math.asin(
            s
        )

        self.imu_yaw = math.atan2(
            2.0 * (
                q.w * q.z
                +
                q.x * q.y
            ),
            1.0
            -
            2.0 * (
                q.y * q.y
                +
                q.z * q.z
            )
        )

    # =================================================================
    # Basic helpers
    # =================================================================

    def current_yaw(self):

        if self.imu_yaw is not None:
            return self.imu_yaw

        return self.odom_yaw

    def nearest_cardinal_yaw(self, yaw):
        """Return the nearest 0, 90, 180 or 270 degree heading in radians."""

        candidates = (
            0.0,
            math.pi / 2.0,
            math.pi,
            -math.pi / 2.0
        )

        return min(
            candidates,
            key=lambda target: abs(wrap_angle(target - yaw))
        )

    def lost_target_yaw(self):
        """Keep the same cardinal target throughout one LOST episode."""

        if self.lost_cardinal_target is None:
            self.lost_cardinal_target = self.nearest_cardinal_yaw(
                self.current_yaw()
            )

        return self.lost_cardinal_target

    def drive(
        self,
        v,
        w,
        linear_limit=None
    ):

        # Forward-Only Differential Drive Policy:
        # Track width CRC_WHEEL_SEPARATION = 0.295m (half-track = 0.1475m).
        # In diff-drive: v_inner = v - |w| * (L/2).
        # To eliminate any backward jerking / motor reversals (v_inner < 0),
        # we dynamically elevate forward speed v so that v_inner >= min_inner_forward.
        # Steering is achieved by ACCELERATING THE OUTER WHEEL FORWARD!
        wheel_half_track = 0.1475
        min_inner_forward = 0.025  # Minimum positive forward crawl (m/s)

        if v > 0.0:
            min_v_needed = min_inner_forward + abs(w) * wheel_half_track
            v = max(v, min_v_needed)

        msg = Twist()

        max_allowed = (
            max(0.35, float(linear_limit))
            if linear_limit is not None
            else max(0.35, self.max_speed)
        )

        msg.linear.x = float(
            clamp(
                v,
                0.0 if v >= 0.0 else -0.1,
                max_allowed
            )
        )

        msg.angular.z = float(
            clamp(
                w,
                -self.max_turn,
                self.max_turn
            )
        )

        self.pub_cmd.publish(
            msg
        )

    def stop(self):

        self.pub_cmd.publish(
            Twist()
        )

    def log_every(
        self,
        seconds,
        key,
        text
    ):

        now = time.time()

        if (
            now
            -
            self.last_log.get(
                key,
                0.0
            )
            >= seconds
        ):

            self.last_log[key] = now

            self.get_logger().info(
                text
            )

    def tick(self):

        try:

            self.control()

        except Exception as exc:

            self.get_logger().error(
                f'control() raised: {exc}'
            )

            self.stop()

    # =================================================================
    # Lidar
    # =================================================================

    def range_sector_percentile(
        self,
        centre_deg,
        width_deg,
        percentile
    ):

        if (
            self.scan is None
            or
            not self.scan.ranges
        ):

            return float('inf')

        centre = math.radians(
            centre_deg
        )

        half = (
            math.radians(
                width_deg
            )
            / 2.0
        )

        values = []

        for i, r in enumerate(
            self.scan.ranges
        ):

            if not math.isfinite(r):
                continue

            if r <= self.scan.range_min:
                continue

            if (
                self.scan.range_max > 0.0
                and
                r >= self.scan.range_max
            ):
                continue

            angle = (
                self.scan.angle_min
                +
                i
                *
                self.scan.angle_increment
            )

            if (
                abs(
                    wrap_angle(
                        angle
                        -
                        centre
                    )
                )
                <= half
            ):

                values.append(
                    r
                )

        if not values:
            return float('inf')

        return float(
            np.percentile(
                np.asarray(
                    values,
                    dtype=np.float32
                ),
                percentile
            )
        )

    # =================================================================
    # White mask
    # =================================================================

    def create_white_mask(
        self,
        image
    ):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lower = np.array(
            [
                0,
                0,
                self.white_v_min
            ],
            dtype=np.uint8
        )

        upper = np.array(
            [
                180,
                self.white_s_max,
                255
            ],
            dtype=np.uint8
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones(
                (
                    3,
                    3
                ),
                np.uint8
            )
        )

        return mask

    # =================================================================
    # Segment geometry
    # =================================================================

    def x_at_y(
        self,
        seg,
        y
    ):

        dy = float(
            seg['yn']
            -
            seg['yf']
        )

        if abs(dy) < 1e-6:
            return None

        t = (
            float(y)
            -
            seg['yf']
        ) / dy

        return float(
            seg['xf']
            +
            t
            * (
                seg['xn']
                -
                seg['xf']
            )
        )

    def segment_y_gap(
        self,
        seg,
        y
    ):

        if (
            seg['yf']
            <= y
            <= seg['yn']
        ):

            return 0.0

        return min(
            abs(
                y
                -
                seg['yf']
            ),
            abs(
                y
                -
                seg['yn']
            )
        )

    # =================================================================
    # Thickness
    # =================================================================

    def measure_segment_thickness(
        self,
        mask,
        seg
    ):

        dx = float(
            seg['xn']
            -
            seg['xf']
        )

        dy = float(
            seg['yn']
            -
            seg['yf']
        )

        length = math.hypot(
            dx,
            dy
        )

        if length < 1.0:
            return None

        nx = (
            -dy
            /
            length
        )

        ny = (
            dx
            /
            length
        )

        h, w = (
            mask.shape[:2]
        )

        max_scan = (
            self.thickness_scan_px
        )

        sample_count = max(
            3,
            self.thickness_samples
        )

        sample_ts = np.linspace(
            0.18,
            0.82,
            sample_count
        )

        cx = seg['xf'] + sample_ts * dx
        cy = seg['yf'] + sample_ts * dy
        offsets = np.arange(-max_scan, max_scan + 1)

        xs_float = cx[:, None] + nx * offsets[None, :]
        ys_float = cy[:, None] + ny * offsets[None, :]

        in_bounds = (xs_float >= 0) & (xs_float < w) & (ys_float >= 0) & (ys_float < h)
        xs = np.clip(np.round(xs_float).astype(int), 0, w - 1)
        ys = np.clip(np.round(ys_float).astype(int), 0, h - 1)

        profiles = (mask[ys, xs] > 0) & in_bounds

        measured_widths = []

        for profile in profiles:

            runs = []
            run_start = None

            for i in range(
                len(profile) + 1
            ):

                active = (
                    i < len(profile)
                    and
                    profile[i]
                )

                if (
                    active
                    and
                    run_start is None
                ):

                    run_start = i

                elif (
                    not active
                    and
                    run_start is not None
                ):

                    run_end = i - 1

                    width_px = (
                        run_end
                        -
                        run_start
                        +
                        1
                    )

                    centre_index = (
                        0.5
                        * (
                            run_start
                            +
                            run_end
                        )
                    )

                    centre_offset = (
                        centre_index
                        -
                        max_scan
                    )

                    runs.append(
                        (
                            abs(
                                centre_offset
                            ),
                            width_px
                        )
                    )

                    run_start = None

            if runs:

                runs.sort(
                    key=lambda item:
                        item[0]
                )

                measured_widths.append(
                    float(
                        runs[0][1]
                    )
                )

        if not measured_widths:
            return None

        return float(
            np.median(
                np.asarray(
                    measured_widths,
                    dtype=np.float32
                )
            )
        )

    def lane_thickness_ok(
        self,
        thickness
    ):

        if thickness is None:
            return False

        if (
            thickness
            <
            self.min_lane_thickness_px
        ):
            return False

        if (
            thickness
            >
            self.max_lane_thickness_px
        ):
            return False

        if (
            self.reference_lane_thickness
            is None
        ):
            return True

        # Clamp effective reference to nominal lane thickness range [8.0, 16.0]
        ref = min(max(float(self.reference_lane_thickness), 8.0), 16.0)

        ratio = (
            thickness
            /
            max(
                ref,
                1e-6
            )
        )

        if (
            ratio
            <
            self.thickness_ratio_min
        ):
            return False

        if (
            ratio
            >
            self.thickness_ratio_max
        ):
            return False

        return True

    def chain_median_thickness(
        self,
        chain
    ):

        values = [
            seg.get(
                'thickness'
            )
            for seg in chain
            if (
                seg.get(
                    'thickness'
                )
                is not None
            )
        ]

        if not values:
            return None

        return float(
            np.median(
                np.asarray(
                    values,
                    dtype=np.float32
                )
            )
        )

    def maybe_update_thickness_reference(
        self,
        chain
    ):

        current = (
            self.chain_median_thickness(
                chain
            )
        )

        if current is None:
            return

        if (
            self.reference_lane_thickness
            is None
        ):

            self.reference_lane_thickness = (
                current
            )

            return

        ratio = (
            current
            /
            max(
                self.reference_lane_thickness,
                1e-6
            )
        )

        if not (
            self.thickness_update_ratio_min
            <= ratio
            <= self.thickness_update_ratio_max
        ):

            return

        # Ignore abnormally thick crosswalk / zebra patterns (> 20px)
        if current > 20.0:
            return

        alpha = (
            self.thickness_alpha
        )

        self.reference_lane_thickness = (
            (
                1.0
                -
                alpha
            )
            *
            self.reference_lane_thickness
            +
            alpha
            *
            current
        )
        self.reference_lane_thickness = min(
            max(float(self.reference_lane_thickness), 8.0),
            16.0
        )

    # =================================================================
    # Hough vectors
    # =================================================================

    def extract_segments(
        self,
        mask,
        near_y
    ):

        h, w = (
            mask.shape
        )

        y0 = int(
            h
            *
            self.roi_top_ratio
        )

        roi = mask[
            y0:h,
            :
        ]

        edges = cv2.Canny(
            roi,
            45,
            145
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        all_segments = []
        lane_segments = []
        left_short_lane_segments = []
        thickness_rejected = []

        if lines is None:
            lines = np.empty((0, 1, 4), dtype=np.int32)

        seg_id = 0

        for raw in lines[:, 0]:

            x1, yy1, x2, yy2 = [
                int(v)
                for v in raw
            ]

            y1 = (
                yy1
                +
                y0
            )

            y2 = (
                yy2
                +
                y0
            )

            # FAR endpoint first
            if y1 <= y2:

                xf = x1
                yf = y1

                xn = x2
                yn = y2

            else:

                xf = x2
                yf = y2

                xn = x1
                yn = y1

            dx = float(
                xn
                -
                xf
            )

            dy = float(
                yn
                -
                yf
            )

            length = math.hypot(
                dx,
                dy
            )

            if (
                length
                <
                self.min_line_length
            ):

                continue

            angle = math.degrees(
                math.atan2(
                    dx,
                    max(
                        dy,
                        1e-6
                    )
                )
            )

            seg = {
                'id':
                    seg_id,

                'xf':
                    float(
                        xf
                    ),

                'yf':
                    float(
                        yf
                    ),

                'xn':
                    float(
                        xn
                    ),

                'yn':
                    float(
                        yn
                    ),

                'length':
                    float(
                        length
                    ),

                'vertical_span':
                    abs(
                        dy
                    ),

                'angle':
                    float(
                        angle
                    ),

                'thickness':
                    None
            }

            seg_id += 1

            all_segments.append(
                seg
            )

            if (
                seg[
                    'vertical_span'
                ]
                <
                self.min_vertical_span_px
            ):

                continue

            if (
                abs(
                    seg[
                        'angle'
                    ]
                )
                >
                self.max_lane_tilt_deg
            ):

                continue

            if (
                seg[
                    'yn'
                ]
                <
                int(
                    h * 0.56
                )
            ):

                continue

            if (
                self.segment_y_gap(
                    seg,
                    near_y
                )
                >
                self.seed_max_y_extrap_px
            ):

                continue

            x_near = (
                self.x_at_y(
                    seg,
                    near_y
                )
            )

            if x_near is None:
                continue

            if x_near < 15.0 or x_near > float(w - 20):
                continue

            seg2 = dict(
                seg
            )

            seg2[
                'x_near'
            ] = (
                x_near
            )

            seg2[
                'thickness'
            ] = (
                self.measure_segment_thickness(
                    mask,
                    seg
                )
            )

            if not self.lane_thickness_ok(
                seg2[
                    'thickness'
                ]
            ):

                thickness_rejected.append(
                    seg2
                )

                continue

            lane_segments.append(
                seg2
            )

        left_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.left_min_line_length,
            maxLineGap=self.max_line_gap
        )

        if left_lines is not None:

            for raw in left_lines[:, 0]:

                x1, yy1, x2, yy2 = [
                    int(v)
                    for v in raw
                ]

                y1 = yy1 + y0
                y2 = yy2 + y0

                if y1 <= y2:
                    xf, yf, xn, yn = x1, y1, x2, y2
                else:
                    xf, yf, xn, yn = x2, y2, x1, y1

                dx = float(xn - xf)
                dy = float(yn - yf)
                length = math.hypot(dx, dy)

                if (
                    length < self.left_min_line_length
                    or
                    length >= self.min_line_length
                ):
                    continue

                angle = math.degrees(
                    math.atan2(
                        dx,
                        max(dy, 1e-6)
                    )
                )

                seg = {
                    'id': seg_id,
                    'xf': float(xf),
                    'yf': float(yf),
                    'xn': float(xn),
                    'yn': float(yn),
                    'length': float(length),
                    'vertical_span': abs(dy),
                    'angle': float(angle)
                }

                seg_id += 1

                if (
                    seg['vertical_span'] < self.min_vertical_span_px
                    or
                    abs(seg['angle']) > self.max_lane_tilt_deg
                    or
                    seg['yn'] < int(h * 0.56)
                    or
                    self.segment_y_gap(seg, near_y) > self.seed_max_y_extrap_px
                ):
                    continue

                x_near = self.x_at_y(
                    seg,
                    near_y
                )

                if (
                    x_near is None
                    or
                    x_near >= w * 0.50
                ):
                    continue

                seg['thickness'] = (
                    self.measure_segment_thickness(
                        mask,
                        seg
                    )
                )

                seg2 = dict(seg)
                seg2['x_near'] = x_near

                if not self.lane_thickness_ok(
                    seg2['thickness']
                ):
                    continue

                left_short_lane_segments.append(
                    seg2
                )

        return (
            all_segments,
            lane_segments,
            left_short_lane_segments,
            thickness_rejected
        )

    def easy_left_seek_measurement(
        self,
        mask,
        near_y,
        mid_y,
        look_y,
        far_y,
        reference=None
    ):
        """Fit one left-boundary vector across a transverse mark or dash gap.

        Used only for LEFT during the SEEK BOTH window.  A wide horizontal
        white stripe is erased before Hough.  Near-collinear pieces on either
        side of that stripe then vote for one longitudinal lane vector.
        """
        h, w = mask.shape
        y0 = int(h * self.roi_top_ratio)
        left_w = int(w * 0.50)

        if left_w < 2 or near_y == look_y:
            return None, []

        left_roi = mask[y0:h, :left_w].copy()

        # A crossing occupies a large fraction of many consecutive rows;
        # a genuine left dash stays narrow.  Remove those rows (and 2 px on
        # either side) only from this LEFT detector, never from RIGHT/NORMAL.
        wide_rows = np.count_nonzero(left_roi, axis=1) >= 0.30 * left_w
        if np.any(wide_rows):
            padded_rows = np.convolve(
                wide_rows.astype(np.uint8),
                np.ones(5, dtype=np.uint8),
                mode='same'
            ) > 0
            left_roi[padded_rows, :] = 0

        edges = cv2.Canny(left_roi, 45, 145)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=min(self.hough_threshold, 12),
            minLineLength=min(self.left_min_line_length, 12),
            maxLineGap=self.max_line_gap
        )

        if lines is None:
            return None, []

        candidates = []

        for index, raw in enumerate(lines[:, 0]):
            x1, yy1, x2, yy2 = [int(v) for v in raw]
            y1, y2 = yy1 + y0, yy2 + y0

            if y1 <= y2:
                xf, yf, xn, yn = x1, y1, x2, y2
            else:
                xf, yf, xn, yn = x2, y2, x1, y1

            dx = float(xn - xf)
            dy = float(yn - yf)
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dx, max(dy, 1e-6)))

            # Only left-half, longitudinal-looking marks.  This excludes
            # horizontal crossings and the mostly horizontal road edge.
            if (
                length < min(self.left_min_line_length, 12)
                or dy < 7.0
                or abs(angle) > 72.0
                or yn < y0 + 8
            ):
                continue

            seg = {
                'id': -(index + 1),
                'xf': float(xf),
                'yf': float(yf),
                'xn': float(xn),
                'yn': float(yn),
                'length': length,
                'vertical_span': dy,
                'angle': angle,
                'thickness': None
            }

            candidates.append(seg)

        if not candidates:
            return None, []

        ref_angle = reference.get('path_angle') if reference else None
        ref_x = reference.get('x_near') if reference else None
        best = None
        best_score = -float('inf')

        for seed in candidates:
            # Group pieces that point along the same left boundary, including
            # pieces separated vertically by the removed horizontal stripe.
            group = []
            seed_slope = (seed['xn'] - seed['xf']) / seed['vertical_span']

            for seg in candidates:
                if angle_diff_deg(seg['angle'], seed['angle']) > 17.0:
                    continue

                y_gap = max(
                    0.0,
                    max(seed['yf'], seg['yf'])
                    - min(seed['yn'], seg['yn'])
                )
                if y_gap > 0.28 * h:
                    continue

                compare_y = 0.5 * (seed['yn'] + seg['yn'])
                separation = abs(
                    self.x_at_y(seed, compare_y)
                    - self.x_at_y(seg, compare_y)
                ) / math.sqrt(1.0 + seed_slope * seed_slope)

                if separation <= max(14.0, 0.055 * w):
                    group.append(seg)

            if not group:
                continue

            top = min(seg['yf'] for seg in group)
            bottom = max(seg['yn'] for seg in group)
            span = bottom - top

            # A first lock needs a real elongated mark or multiple agreeing
            # fragments.  Once a left direction is known, one short dash can
            # still keep the vehicle on that direction.
            if (
                ref_angle is None
                and len(group) < 2
                and span < max(18.0, 0.05 * h)
            ):
                continue

            slope = float(np.median([
                (seg['xn'] - seg['xf']) / seg['vertical_span']
                for seg in group
            ]))
            fitted_angle = math.degrees(math.atan(slope))
            angle_error = (
                angle_diff_deg(fitted_angle, ref_angle)
                if ref_angle is not None else 0.0
            )

            # An apparent vector through the crossing must agree with the
            # lane already being followed.  If none agrees, normal LOST keeps
            # the last valid yaw instead of steering toward the crossing.
            if angle_error > 25.0:
                continue

            x_bottom = float(np.median([
                seg['xn'] + slope * (bottom - seg['yn'])
                for seg in group
            ]))
            proxy_y = min(float(near_y), bottom + 0.08 * h)
            x_proxy = x_bottom + slope * (proxy_y - bottom)
            x_near = float(clamp(x_proxy, 0.0, float(left_w - 1)))

            def projected_x(sample_y):
                return float(clamp(
                    x_near - slope * (near_y - sample_y),
                    0.0,
                    float(w - 1)
                ))

            x_look = projected_x(look_y)
            measurement = {
                'x_near': x_near,
                'x_mid': projected_x(mid_y),
                'x_look': x_look,
                'x_far': projected_x(far_y),
                'path_angle': math.degrees(math.atan2(
                    x_near - x_look,
                    near_y - look_y
                ))
            }

            score = (
                span
                + 0.20 * sum(seg['length'] for seg in group)
                + 0.04 * bottom
                - 1.5 * angle_error
            )
            if ref_x is not None:
                score -= 0.40 * abs(x_near - ref_x)

            if score > best_score:
                best_score = score
                for seg in group:
                    seg['x_near'] = x_near
                chain = sorted(
                    group,
                    key=lambda seg: seg['yn'],
                    reverse=True
                )[:self.chain_max_segments]
                best = (measurement, chain)

        return best if best is not None else (None, [])

    # =================================================================
    # Seed selection
    # =================================================================

    def choose_initial_seed(
        self,
        candidates,
        width,
        near_y
    ):

        valid = []

        for seg in candidates:

            # Candidate must be in the right half of the image and not pressed against the border
            if (
                seg['x_near'] < width * 0.45
                or seg['x_near'] > (width - 25)
            ):
                continue

            # Real tape line has thickness >= 8px; filter out thin single-edge mat borders
            if (
                seg.get('thickness') is not None
                and seg['thickness'] < 8.0
            ):
                continue

            # Real lane marking has notable length
            if seg['length'] < 35.0:
                continue

            y_gap = self.segment_y_gap(seg, near_y)

            # Strongly prioritize long continuous lane tape, good vertical span,
            # and proximity to the nominal right-lane line when car is centered (~520-560px),
            # penalizing distance from near_y.
            score = (
                2.0 * seg['length']
                + 1.5 * seg['vertical_span']
                - 0.4 * abs(seg['x_near'] - 530.0)
                - 0.5 * y_gap
            )

            valid.append((score, seg))

        if not valid:
            return None

        return max(valid, key=lambda item: item[0])[1]

    def choose_tracking_seed(
        self,
        candidates
    ):

        if (
            self.track_x_near
            is None
            or
            self.track_path_angle_deg
            is None
        ):

            return None

        scored = []

        for seg in candidates:

            xerr = abs(
                seg[
                    'x_near'
                ]
                -
                self.track_x_near
            )

            aerr = (
                angle_diff_deg(
                    seg[
                        'angle'
                    ],
                    self.track_path_angle_deg
                )
            )

            if (
                xerr
                >
                self.tracking_corridor_px
            ):

                continue

            if (
                aerr
                >
                self.tracking_angle_deg
            ):

                continue

            score = (
                xerr
                +
                1.1
                *
                aerr
                -
                0.02
                *
                seg[
                    'length'
                ]
            )

            scored.append(
                (
                    score,
                    seg
                )
            )

        if not scored:
            return None

        return min(
            scored,
            key=lambda item:
                item[0]
        )[1]

    def choose_recovery_seed_for_side(
        self,
        candidates,
        width,
        near_y,
        side
    ):
        """Choose one outer boundary without mixing left and right.

        The right-side score is exactly the original recovery score.
        The left-side score is its mirror image, so the existing recovery
        geometry is preserved while allowing a second side to be searched.
        """

        valid = []

        for seg in candidates:

            x_near = seg[
                'x_near'
            ]

            if side == 'right':

                if (
                    x_near
                    <
                    width * 0.50
                ):

                    continue

                side_score = (
                    x_near
                )

            elif side == 'left':

                if (
                    x_near
                    >=
                    width * 0.50
                ):

                    continue

                side_score = (
                    width
                    -
                    x_near
                )

            else:

                return None

            y_gap = (
                self.segment_y_gap(
                    seg,
                    near_y
                )
            )

            score = (
                side_score
                +
                0.05
                *
                seg[
                    'length'
                ]
                -
                0.30
                *
                y_gap
            )

            valid.append(
                (
                    score,
                    seg
                )
            )

        if not valid:
            return None

        return max(
            valid,
            key=lambda item:
                item[0]
        )[1]

    def choose_recovery_seed(
        self,
        candidates,
        width,
        near_y
    ):
        """Original right-boundary recovery selector."""

        return self.choose_recovery_seed_for_side(
            candidates,
            width,
            near_y,
            'right'
        )

    # =================================================================
    # Connected chain
    # =================================================================

    def connection_distance(
        self,
        current,
        candidate
    ):

        d1 = point_segment_distance(
            current[
                'xf'
            ],
            current[
                'yf'
            ],
            candidate[
                'xf'
            ],
            candidate[
                'yf'
            ],
            candidate[
                'xn'
            ],
            candidate[
                'yn'
            ]
        )

        d2 = point_segment_distance(
            candidate[
                'xn'
            ],
            candidate[
                'yn'
            ],
            current[
                'xf'
            ],
            current[
                'yf'
            ],
            current[
                'xn'
            ],
            current[
                'yn'
            ]
        )

        return min(
            d1,
            d2
        )

    def build_chain(
        self,
        seed,
        candidates
    ):

        if seed is None:
            return []

        chain = [
            seed
        ]

        used = {
            seed[
                'id'
            ]
        }

        for _ in range(
            self.chain_max_segments
            -
            1
        ):

            current = (
                chain[-1]
            )

            best = None
            best_score = float(
                'inf'
            )

            for candidate in candidates:

                if (
                    candidate[
                        'id'
                    ]
                    in
                    used
                ):

                    continue

                progress = (
                    current[
                        'yf'
                    ]
                    -
                    candidate[
                        'yf'
                    ]
                )

                if (
                    progress
                    <
                    self.chain_min_progress_px
                ):

                    continue

                if (
                    candidate[
                        'yn'
                    ]
                    >
                    current[
                        'yn'
                    ]
                    +
                    28.0
                ):

                    continue

                angle_change = (
                    angle_diff_deg(
                        current[
                            'angle'
                        ],
                        candidate[
                            'angle'
                        ]
                    )
                )

                if (
                    angle_change
                    >
                    self.chain_max_turn_deg
                ):

                    continue

                connect = (
                    self.connection_distance(
                        current,
                        candidate
                    )
                )

                if (
                    connect
                    >
                    self.chain_connect_px
                ):

                    continue

                thickness_penalty = 0.0

                t1 = current.get(
                    'thickness'
                )

                t2 = candidate.get(
                    'thickness'
                )

                if (
                    t1 is not None
                    and
                    t2 is not None
                ):

                    thickness_penalty = (
                        4.0
                        *
                        abs(
                            math.log(
                                max(
                                    t2,
                                    1e-3
                                )
                                /
                                max(
                                    t1,
                                    1e-3
                                )
                            )
                        )
                    )

                score = (
                    connect
                    +
                    0.65
                    *
                    angle_change
                    -
                    0.08
                    *
                    progress
                    +
                    thickness_penalty
                )

                if (
                    score
                    <
                    best_score
                ):

                    best_score = (
                        score
                    )

                    best = (
                        candidate
                    )

            if best is None:
                break

            chain.append(
                best
            )

            used.add(
                best[
                    'id'
                ]
            )

        return chain

    # =================================================================
    # Sample chain
    # =================================================================

    def sample_chain_x(
        self,
        chain,
        y
    ):

        if not chain:
            return None

        best_x = None
        best_gap = float(
            'inf'
        )

        for seg in chain:

            gap = (
                self.segment_y_gap(
                    seg,
                    y
                )
            )

            if (
                gap
                >
                self.sample_max_y_gap_px
            ):

                continue

            x = (
                self.x_at_y(
                    seg,
                    y
                )
            )

            if x is None:
                continue

            if (
                gap
                <
                best_gap
            ):

                best_gap = (
                    gap
                )

                best_x = (
                    x
                )

        return best_x

    def chain_measurement(
        self,
        chain,
        near_y,
        mid_y,
        look_y,
        far_y
    ):

        if not chain:
            return None

        x_near = (
            self.sample_chain_x(
                chain,
                near_y
            )
        )

        x_mid = (
            self.sample_chain_x(
                chain,
                mid_y
            )
        )

        x_look = (
            self.sample_chain_x(
                chain,
                look_y
            )
        )

        x_far = (
            self.sample_chain_x(
                chain,
                far_y
            )
        )

        if x_near is None:
            return None

        if x_look is None:

            if x_mid is not None:

                x_look = (
                    x_mid
                )

            elif (
                self.track_x_look
                is not None
            ):

                x_look = (
                    self.track_x_look
                )

            else:

                x_look = (
                    x_near
                )

        if x_mid is None:

            x_mid = (
                0.5
                *
                (
                    x_near
                    +
                    x_look
                )
            )

        if x_far is None:

            x_far = (
                x_look
            )

        dy = float(
            near_y
            -
            look_y
        )

        if abs(dy) < 1e-6:
            return None

        path_angle = math.degrees(
            math.atan2(
                x_near
                -
                x_look,
                dy
            )
        )

        return {
            'x_near':
                float(
                    x_near
                ),

            'x_mid':
                float(
                    x_mid
                ),

            'x_look':
                float(
                    x_look
                ),

            'x_far':
                float(
                    x_far
                ),

            'path_angle':
                float(
                    path_angle
                )
        }

    # =================================================================
    # Transverse
    # =================================================================

    def segment_distance(
        self,
        a,
        b
    ):

        if segments_intersect(
            a,
            b
        ):
            return 0.0

        return min(
            point_segment_distance(
                a['xf'],
                a['yf'],
                b['xf'],
                b['yf'],
                b['xn'],
                b['yn']
            ),

            point_segment_distance(
                a['xn'],
                a['yn'],
                b['xf'],
                b['yf'],
                b['xn'],
                b['yn']
            ),

            point_segment_distance(
                b['xf'],
                b['yf'],
                a['xf'],
                a['yf'],
                a['xn'],
                a['yn']
            ),

            point_segment_distance(
                b['xn'],
                b['yn'],
                a['xf'],
                a['yf'],
                a['xn'],
                a['yn']
            )
        )

    def detect_transverse(
        self,
        all_segments,
        chain
    ):

        if not chain:
            return []

        chain_ids = {
            seg[
                'id'
            ]
            for seg in chain
        }

        transverse = []

        for seg in all_segments:

            if (
                seg[
                    'id'
                ]
                in chain_ids
            ):

                continue

            if (
                seg[
                    'length'
                ]
                <
                self.transverse_min_length
            ):

                continue

            best_relative_angle = None
            best_distance = float(
                'inf'
            )

            for lane_seg in chain:

                relative_angle = (
                    angle_diff_deg(
                        seg[
                            'angle'
                        ],
                        lane_seg[
                            'angle'
                        ]
                    )
                )

                distance = (
                    self.segment_distance(
                        seg,
                        lane_seg
                    )
                )

                if (
                    distance
                    <
                    best_distance
                ):

                    best_distance = (
                        distance
                    )

                    best_relative_angle = (
                        relative_angle
                    )

            if (
                best_relative_angle
                is None
            ):

                continue

            if (
                best_relative_angle
                <
                self.transverse_relative_angle_deg
            ):

                continue

            if (
                best_distance
                >
                self.transverse_distance_px
            ):

                continue

            transverse.append(
                seg
            )

        return transverse

    # =================================================================
    # Temporal tracking
    # =================================================================

    def measurement_plausible(
        self,
        measurement
    ):

        if measurement is None:
            return False

        if (
            self.track_x_near
            is not None
        ):

            if (
                abs(
                    measurement[
                        'x_near'
                    ]
                    -
                    self.track_x_near
                )
                >
                self.max_near_jump_px
            ):

                return False

        if (
            self.track_path_angle_deg
            is not None
        ):

            if (
                angle_diff_deg(
                    measurement[
                        'path_angle'
                    ],
                    self.track_path_angle_deg
                )
                >
                self.max_path_angle_jump_deg
            ):

                return False

        return True

    def set_track_from_measurement(
        self,
        measurement
    ):

        self.track_x_near = (
            measurement[
                'x_near'
            ]
        )

        self.track_x_mid = (
            measurement[
                'x_mid'
            ]
        )

        self.track_x_look = (
            measurement[
                'x_look'
            ]
        )

        self.track_x_far = (
            measurement[
                'x_far'
            ]
        )

        self.track_path_angle_deg = (
            measurement[
                'path_angle'
            ]
        )

    def update_track(
        self,
        measurement
    ):

        if not self.measurement_plausible(
            measurement
        ):

            return False

        if (
            self.track_x_near
            is None
        ):

            self.set_track_from_measurement(
                measurement
            )

            return True

        alpha = (
            self.lane_alpha
        )

        angle_alpha = (
            self.angle_alpha
        )

        self.track_x_near = (
            (
                1.0
                -
                alpha
            )
            *
            self.track_x_near
            +
            alpha
            *
            measurement[
                'x_near'
            ]
        )

        self.track_x_mid = (
            (
                1.0
                -
                alpha
            )
            *
            self.track_x_mid
            +
            alpha
            *
            measurement[
                'x_mid'
            ]
        )

        self.track_x_look = (
            (
                1.0
                -
                alpha
            )
            *
            self.track_x_look
            +
            alpha
            *
            measurement[
                'x_look'
            ]
        )

        self.track_x_far = (
            (
                1.0
                -
                alpha
            )
            *
            self.track_x_far
            +
            alpha
            *
            measurement[
                'x_far'
            ]
        )

        self.track_path_angle_deg = (
            (
                1.0
                -
                angle_alpha
            )
            *
            self.track_path_angle_deg
            +
            angle_alpha
            *
            measurement[
                'path_angle'
            ]
        )

        return True

    # =================================================================
    # Preview
    # =================================================================

    def preview_effective_error(
        self,
        preview_error_px
    ):

        if (
            abs(
                preview_error_px
            )
            <=
            self.preview_deadband_px
        ):

            return 0.0

        return math.copysign(
            abs(
                preview_error_px
            )
            -
            self.preview_deadband_px,
            preview_error_px
        )

    # =================================================================
    # Lost control
    # =================================================================

    def short_lost_command(
        self
    ):

        current = (
            self.current_yaw()
        )

        # -------------------------------------------------------------
        # First LOST phase: hold LAST VALID LANE YAW for a full 3 s
        # (configurable by lost_anchor_delay_sec).
        #
        # This is side-agnostic: last_good_yaw is refreshed whenever the
        # active boundary track is valid, whether that boundary is RIGHT
        # or the temporary LEFT fallback.
        # -------------------------------------------------------------
        hold_frames = max(
            1,
            int(round(self.lost_anchor_delay_sec * self.rate))
        )

        if self.lost_count <= 4 and self.last_good_turn is not None:
            # Brief dropout grace period (e.g. dashed line gap, lighting glitch):
            # Maintain forward cruising momentum and smoothly decay steering
            glide_speed = max(0.075, self.max_speed * 0.85)
            glide_turn = self.last_good_turn * (0.90 ** self.lost_count)
            return (
                glide_speed,
                glide_turn,
                f'LOST GLIDE {self.lost_count} | MOMENTUM',
                0.0
            )

        if self.lost_count <= hold_frames:

            if self.last_good_yaw is not None:
                target = self.last_good_yaw
            else:
                target = current

            yaw_error = wrap_angle(
                target
                -
                current
            )

            # Do not immediately replay the previous lane steering command.
            # During the 3 s grace window we only hold the last valid lane yaw.
            turn = clamp(
                self.lost_heading_kp
                *
                yaw_error,
                -self.lost_turn_limit,
                self.lost_turn_limit
            )

            return (
                self.lost_creep_speed,
                turn,
                (
                    f'LOST {self.lost_count / max(self.rate, 1e-6):.1f}s '
                    f'-> HOLD LAST LANE YAW '
                    f'{math.degrees(target) % 360.0:.1f}deg'
                ),
                math.degrees(yaw_error)
            )

        # -------------------------------------------------------------
        # If SEEK BOTH was activated recently, do NOT return to a cardinal
        # anchor after the 3 s hold.  Keep the last valid lane yaw and let
        # perception search both outer boundaries until the persistent window
        # expires.  This state survives lane reacquisition because its timer is
        # independent from lost_count.
        # -------------------------------------------------------------
        if self.seek_both_window_active():

            if self.last_good_yaw is not None:
                target = self.last_good_yaw
            else:
                target = current

            yaw_error = wrap_angle(
                target
                -
                current
            )

            turn = clamp(
                self.lost_heading_kp
                *
                yaw_error,
                -self.lost_turn_limit,
                self.lost_turn_limit
            )

            # Keep creeping so the camera has a chance to recover either side.
            # If yaw has drifted badly, use the slower correction speed rather
            # than stopping and falling back to a cardinal heading.
            if (
                abs(math.degrees(yaw_error))
                >
                self.lost_rotate_tolerance_deg
            ):
                speed = self.long_lost_correct_speed
            else:
                speed = self.lost_creep_speed

            return (
                speed,
                turn,
                (
                    f'LOST > {self.lost_anchor_delay_sec:.1f}s -> '
                    f'SEEK BOTH WINDOW '
                    f'{self.seek_both_window_remaining_sec():.1f}s | '
                    f'HOLD LAST LANE YAW '
                    f'{math.degrees(target) % 360.0:.1f}deg'
                ),
                math.degrees(yaw_error)
            )

        # -------------------------------------------------------------
        # Outside the SEEK BOTH persistence window, the original recovery
        # resumes: return toward one of the four cardinal anchors.
        # -------------------------------------------------------------
        target = self.lost_target_yaw()

        yaw_error = wrap_angle(
            target
            -
            current
        )

        turn = clamp(
            self.lost_heading_kp
            *
            yaw_error,
            -self.lost_turn_limit,
            self.lost_turn_limit
        )

        if (
            abs(math.degrees(yaw_error))
            >
            self.lost_rotate_tolerance_deg
        ):
            speed = 0.0
        else:
            speed = self.lost_creep_speed

        return (
            speed,
            turn,
            (
                f'LOST > {self.lost_anchor_delay_sec:.1f}s -> IMU ANCHOR '
                f'{math.degrees(target) % 360.0:.0f} + SEEK '
                f'{self.tracked_lane_side.upper()}'
            ),
            math.degrees(yaw_error)
        )

    def long_lost_elapsed_sec(
        self
    ):
        """Seconds elapsed since the controller entered LONG LOST."""

        if (
            self.lost_count
            <=
            self.long_lost_frames
        ):

            return 0.0

        return (
            self.lost_count
            -
            self.long_lost_frames
        ) / max(
            self.rate,
            1e-6
        )

    def seek_both_window_active(
        self
    ):
        """Return True while the persistent SEEK BOTH window is alive."""

        if self.seek_both_until is None:
            return False

        now = self.get_clock().now().nanoseconds * 1e-9

        if now < self.seek_both_until:
            return True

        # Expire exactly once.  Keep seek_both_retrigger_blocked=True while
        # we are still in the same LOST episode, so the old long-lost timer
        # cannot immediately open another 15 s window.
        self.seek_both_started_at = None
        self.seek_both_until = None
        self.seek_both_no_lock_since = None

        self.get_logger().info(
            'SEEK BOTH WINDOW EXPIRED -> cardinal IMU anchors re-enabled'
        )

        return False

    def seek_both_window_remaining_sec(
        self
    ):
        """Seconds left in the persistent SEEK BOTH window."""

        if not self.seek_both_window_active():
            return 0.0

        return max(
            0.0,
            self.seek_both_until - (self.get_clock().now().nanoseconds * 1e-9)
        )

    def reset_seek_both_fail_recovery(
        self,
        clear_turn_state=False
    ):
        """Reset the 8 s no-lock timer; optionally clear LEFT-90 fallback."""

        self.seek_both_no_lock_since = None

        if clear_turn_state:
            self.seek_right_after_left_turn_active = False
            self.seek_right_turn_target_yaw = None
            self.seek_right_turn_complete = False

    def maybe_trigger_seek_right_after_failed_both(
        self,
        dual_seek_active
    ):
        """After 8 s of unsuccessful SEEK BOTH, LEFT 90 then SEEK RIGHT."""

        if self.seek_right_after_left_turn_active:
            return True

        # Count only time spent in the actual dual-side search.  In particular,
        # a fresh LOST during the persistence window still gets its normal 3 s
        # HOLD LAST LANE YAW before this timer starts again.
        if not dual_seek_active:
            self.seek_both_no_lock_since = None
            return False

        now = self.get_clock().now().nanoseconds * 1e-9

        if self.seek_both_no_lock_since is None:
            self.seek_both_no_lock_since = now
            return False

        elapsed = now - self.seek_both_no_lock_since

        if elapsed < self.seek_both_fail_timeout_sec:
            return False

        # Latch the manoeuvre exactly once.  Positive angular.z is left/CCW,
        # so +90 degrees from the current IMU yaw is the desired target.
        self.seek_right_after_left_turn_active = True
        self.seek_right_turn_target_yaw = wrap_angle(
            self.current_yaw()
            +
            math.radians(self.seek_both_left_turn_deg)
        )
        self.seek_right_turn_complete = False

        # SEEK BOTH is over now.  Keep retrigger blocked for this LOST episode
        # so the dual-side window cannot reopen while SEEK RIGHT is active.
        self.seek_both_started_at = None
        self.seek_both_until = None
        self.seek_both_retrigger_blocked = True

        self.recovery_lane_side = 'right'
        self.recovery_samples = []
        self.right_return_samples = []

        self.get_logger().info(
            (
                'SEEK BOTH TIMEOUT | '
                f'no lock for {elapsed:.1f}s -> '
                f'LEFT {self.seek_both_left_turn_deg:.0f}deg -> SEEK RIGHT | '
                f'targetYaw='
                f'{math.degrees(self.seek_right_turn_target_yaw) % 360.0:.1f}deg'
            )
        )

        return True

    def maybe_rearm_seek_both_after_valid_track(
        self
    ):
        """Allow a future LONG LOST episode to start a fresh window."""

        if (
            self.seek_both_retrigger_blocked
            and
            not self.seek_both_window_active()
        ):

            self.seek_both_retrigger_blocked = False

            self.get_logger().info(
                'SEEK BOTH re-armed after valid lane tracking'
            )

    def maybe_start_seek_both_window(
        self
    ):
        """Start SEEK BOTH once per recovery cycle after the old delay."""

        if self.seek_both_window_active():
            return True

        if self.seek_both_retrigger_blocked:
            return False

        ready = (
            self.lost_count
            >
            self.long_lost_frames
            and
            self.long_lost_elapsed_sec()
            >=
            self.long_lost_dual_seek_delay_sec
        )

        if not ready:
            return False

        now = self.get_clock().now().nanoseconds * 1e-9

        self.seek_both_started_at = now
        self.seek_both_until = (
            now
            +
            self.seek_both_window_sec
        )
        self.seek_both_retrigger_blocked = True

        self.seek_both_no_lock_since = None
        self.seek_right_after_left_turn_active = False
        self.seek_right_turn_target_yaw = None
        self.seek_right_turn_complete = False

        self.get_logger().info(
            (
                'SEEK BOTH WINDOW START | '
                f'duration={self.seek_both_window_sec:.1f}s | '
                'persists across reacquisition'
            )
        )

        return True

    def long_lost_dual_seek_active(
        self
    ):
        """Return True when recovery should currently inspect BOTH sides."""

        window_active = (
            self.maybe_start_seek_both_window()
        )

        if not window_active:
            return False

        # On every fresh loss, preserve the requested 3 s HOLD LAST LANE YAW
        # phase first.  After that grace period, the still-live window takes
        # over and the recovery search returns to BOTH sides immediately.
        hold_frames = max(
            1,
            int(round(self.lost_anchor_delay_sec * self.rate))
        )

        return (
            self.lost_count
            >
            hold_frames
        )

    def long_lost_command(
        self
    ):
        """
        Never stop only because vision has been lost for a long time.

        Hold the cardinal heading chosen at first LOST and seek the lane.
        """

        current_yaw = (
            self.current_yaw()
        )

        # -------------------------------------------------------------
        # SEEK BOTH timeout fallback:
        # rotate LEFT by the latched 90-degree target, then hold that heading
        # while perception searches RIGHT only.
        # -------------------------------------------------------------
        if self.seek_right_after_left_turn_active:

            if self.seek_right_turn_target_yaw is None:
                self.seek_right_turn_target_yaw = wrap_angle(
                    current_yaw
                    +
                    math.radians(self.seek_both_left_turn_deg)
                )

            target_yaw = self.seek_right_turn_target_yaw

            yaw_error = wrap_angle(
                target_yaw
                -
                current_yaw
            )

            yaw_error_deg = math.degrees(
                yaw_error
            )

            turn = clamp(
                self.lost_heading_kp
                *
                yaw_error,
                -self.long_lost_turn_limit,
                self.long_lost_turn_limit
            )

            if not self.seek_right_turn_complete:

                if (
                    abs(yaw_error_deg)
                    <=
                    self.lost_rotate_tolerance_deg
                ):

                    self.seek_right_turn_complete = True
                    turn = 0.0
                    speed = self.long_lost_speed

                    self.get_logger().info(
                        (
                            f'LEFT {self.seek_both_left_turn_deg:.0f}deg COMPLETE '
                            '-> SEEK RIGHT'
                        )
                    )

                else:

                    # Rotate in place so the requested 90-degree manoeuvre is
                    # deterministic and does not add forward drift.
                    speed = 0.0

                state_text = (
                    f'SEEK BOTH TIMEOUT -> LEFT '
                    f'{self.seek_both_left_turn_deg:.0f}deg | '
                    f'target={math.degrees(target_yaw) % 360.0:.1f} '
                    f'err={yaw_error_deg:+.1f}'
                )

            else:

                # The turn has completed.  Creep forward while holding the new
                # heading; the recovery perception path is forced to RIGHT.
                if (
                    abs(yaw_error_deg)
                    >
                    self.long_lost_yaw_tolerance_deg
                ):
                    speed = self.long_lost_correct_speed
                else:
                    speed = self.long_lost_speed

                state_text = (
                    f'AFTER LEFT {self.seek_both_left_turn_deg:.0f}deg '
                    f'-> SEEK RIGHT | '
                    f'target={math.degrees(target_yaw) % 360.0:.1f}'
                )

            return (
                speed,
                turn,
                state_text,
                yaw_error_deg
            )

        dual_seek_active = (
            self.long_lost_dual_seek_active()
        )

        if dual_seek_active:

            # While the persistent SEEK BOTH window is alive, do not pull the
            # robot back toward 0/90/180/270.  Continue from the most recent
            # valid lane yaw while perception searches both boundaries.
            if self.last_good_yaw is not None:
                target_yaw = self.last_good_yaw
            else:
                target_yaw = current_yaw

        else:

            target_yaw = self.lost_target_yaw()

        yaw_error = wrap_angle(
            target_yaw
            -
            current_yaw
        )

        yaw_error_deg = math.degrees(
            yaw_error
        )

        turn = (
            self.lost_heading_kp
            *
            yaw_error
        )

        turn = clamp(
            turn,
            -self.long_lost_turn_limit,
            self.long_lost_turn_limit
        )

        # If heading drifted badly, move only very slowly.
        if (
            abs(
                yaw_error_deg
            )
            >
            self.long_lost_yaw_tolerance_deg
        ):

            speed = (
                self.long_lost_correct_speed
            )

        else:

            speed = (
                self.long_lost_speed
            )

        if dual_seek_active:

            state_text = (
                f'LOST LONG -> SEEK BOTH WINDOW '
                f'{self.seek_both_window_remaining_sec():.1f}s | '
                f'HOLD LAST LANE YAW '
                f'{math.degrees(target_yaw) % 360.0:.1f}'
            )

        else:

            state_text = (
                f'LOST LONG -> IMU '
                f'{math.degrees(target_yaw) % 360.0:.0f} + '
                f'SEEK {self.tracked_lane_side.upper()}'
            )

        return (
            speed,
            turn,
            state_text,
            yaw_error_deg
        )

    # =================================================================
    # Debug
    # =================================================================

    def publish_debug(
        self,
        image,
        state,
        lane_segments,
        thickness_rejected,
        chain,
        transverse,
        measurement,
        near_y,
        mid_y,
        look_y,
        far_y,
        front,
        turn,
        preview_error=0.0,
        preview_effective=0.0
    ):

        now = time.monotonic()
        if self.lane_only_mode:
            if self.pub_debug.get_subscription_count() == 0:
                return
            lane_debug_interval = 1.0 / max(0.5, self.lane_debug_fps)
            if (now - self.last_lane_debug_publish_time) < lane_debug_interval:
                return

        debug = (
            image.copy()
        )

        _, w = (
            debug.shape[:2]
        )

        # -------------------------------------------------------------
        # Clean & Minimalist Lane HUD (High-clarity, no clutter)
        # -------------------------------------------------------------
        # 1. Tracked lane segments (chain): Bright green line
        for seg in chain:
            cv2.line(
                debug,
                (int(seg['xf']), int(seg['yf'])),
                (int(seg['xn']), int(seg['yn'])),
                (0, 255, 0),
                4,
                cv2.LINE_AA
            )

        # 2. Transverse markings (crosswalk/stop line): Red line
        for seg in transverse:
            cv2.line(
                debug,
                (int(seg['xf']), int(seg['yf'])),
                (int(seg['xn']), int(seg['yn'])),
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        # 3. Smoothed tracking trajectory: Smooth cyan curve
        points = []
        for x, y in (
            (self.track_x_far, far_y),
            (self.track_x_look, look_y),
            (self.track_x_mid, mid_y),
            (self.track_x_near, near_y),
        ):
            if x is not None:
                points.append((int(x), int(y)))

        if len(points) >= 2:
            cv2.polylines(
                debug,
                [np.asarray(points, dtype=np.int32)],
                False,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

        # 4. Target & Lookahead markers: subtle indicators
        if self.target_x_near is not None:
            cv2.circle(
                debug,
                (int(self.target_x_near), int(near_y)),
                8,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

        if measurement is not None:
            cv2.circle(
                debug,
                (int(measurement['x_look']), int(look_y)),
                6,
                (255, 0, 255),
                -1,
                cv2.LINE_AA
            )

        # 5. Clean, modern HUD badge on top-left (semi-transparent card)
        overlay = debug.copy()
        cv2.rectangle(overlay, (12, 12), (370, 82), (18, 22, 30), -1)
        cv2.addWeighted(overlay, 0.70, debug, 0.30, 0, debug)
        cv2.rectangle(debug, (12, 12), (370, 82), (70, 85, 105), 1, cv2.LINE_AA)

        side_str = getattr(self, 'tracked_lane_side', 'right').upper()
        if not getattr(self, 'lane_locked', False):
            state_str = 'INITIALIZING'
        elif getattr(self, 'lost_count', 0) > 0:
            state_str = f'SEARCHING ({self.lost_count})'
        elif abs(turn) < 0.04:
            state_str = f'CENTERED ({side_str})'
        else:
            state_str = f'TRACKING ({side_str})'

        steer_dir = 'LEFT' if turn > 0.04 else ('RIGHT' if turn < -0.04 else 'STRAIGHT')

        cv2.putText(
            debug,
            f'LANE: {state_str}',
            (22, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            debug,
            f'STEER: {turn:+.2f} rad/s ({steer_dir})',
            (22, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        e_lat = (self.track_x_near - self.target_x_near) if (self.track_x_near is not None and self.target_x_near is not None) else 0.0
        e_head = (self.track_path_angle_deg - self.reference_path_angle_deg) if (self.track_path_angle_deg is not None and self.reference_path_angle_deg is not None) else 0.0
        cv2.putText(
            debug,
            f'eLat: {e_lat:+.0f}px  |  eHead: {e_head:+.1f}deg',
            (22, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 205, 230),
            1,
            cv2.LINE_AA
        )

        self.publish_lane_debug_frame(debug)

    # =================================================================
    # RIGHT-LANE OBSTACLE BYPASS
    # =================================================================

    def obstacle_bypass_distance(self):

        if self.obstacle_bypass_origin is None:
            return 0.0

        return math.hypot(
            self.odom_x - self.obstacle_bypass_origin[0],
            self.odom_y - self.obstacle_bypass_origin[1]
        )

    def obstacle_bypass_set_phase(self, phase, target_yaw=None):

        self.obstacle_bypass_phase = phase
        self.obstacle_bypass_target_yaw = target_yaw
        self.obstacle_bypass_origin = (self.odom_x, self.odom_y)

    def start_obstacle_bypass(self):

        base = self.current_yaw()

        self.obstacle_bypass_active = True
        self.obstacle_bypass_reference_yaw = base

        # The detour supersedes LOST/SEEK-BOTH memory.  After completing it,
        # perception starts again from a clean RIGHT-boundary acquisition.
        self.seek_both_started_at = None
        self.seek_both_until = None
        self.seek_both_no_lock_since = None
        self.seek_both_retrigger_blocked = False
        self.seek_right_after_left_turn_active = False
        self.seek_right_turn_target_yaw = None
        self.seek_right_turn_complete = False

        self.obstacle_bypass_set_phase(
            'TURN_LEFT_OUT',
            wrap_angle(base + math.pi / 2.0)
        )

        self.get_logger().info(
            'OBSTACLE BYPASS START | LEFT 90 -> OUT -> RIGHT 90 -> PASS '
            '-> RIGHT 90 -> RETURN -> LEFT 90 -> SEEK RIGHT'
        )

    def finish_obstacle_bypass(self):

        self.obstacle_bypass_active = False
        self.obstacle_bypass_phase = None
        self.obstacle_bypass_target_yaw = None
        self.obstacle_bypass_origin = None
        self.obstacle_bypass_reference_yaw = None
        self.obstacle_count = 0

        # Force a fresh RIGHT-boundary acquisition rather than trusting the lane
        # state that existed before the open-loop rectangular manoeuvre.
        self.lane_locked = False
        self.tracked_lane_side = 'right'
        self.acquire_samples = []
        self.recovery_samples = []
        self.recovery_lane_side = None
        self.right_return_samples = []
        self.lost_count = 0
        self.lost_cardinal_target = None
        self.last_turn = 0.0

    def obstacle_bypass_command(self):

        if not self.obstacle_bypass_active:
            return None

        phase = self.obstacle_bypass_phase
        base = self.obstacle_bypass_reference_yaw
        current = self.current_yaw()

        if base is None:
            self.finish_obstacle_bypass()
            return (0.0, 0.0, 'OBSTACLE BYPASS ABORT')

        # -------------------------------------------------------------
        # TURN phases: rotate in place using IMU/odom yaw.
        # -------------------------------------------------------------
        if phase in (
            'TURN_LEFT_OUT',
            'TURN_RIGHT_FORWARD',
            'TURN_RIGHT_IN',
            'TURN_LEFT_RESUME'
        ):

            target = self.obstacle_bypass_target_yaw
            yaw_error = wrap_angle(target - current)
            yaw_error_deg = math.degrees(yaw_error)

            if abs(yaw_error_deg) <= self.obstacle_bypass_turn_tolerance_deg:

                if phase == 'TURN_LEFT_OUT':
                    self.obstacle_bypass_set_phase('STRAIGHT_OUT', target)
                    return (0.0, 0.0, 'OBSTACLE LEFT 90 DONE -> OUT')

                if phase == 'TURN_RIGHT_FORWARD':
                    self.obstacle_bypass_set_phase('STRAIGHT_PASS', target)
                    return (0.0, 0.0, 'OBSTACLE FORWARD ALIGNED -> PASS')

                if phase == 'TURN_RIGHT_IN':
                    self.obstacle_bypass_set_phase('STRAIGHT_RETURN', target)
                    return (0.0, 0.0, 'OBSTACLE RIGHT 90 DONE -> RETURN')

                # Final LEFT turn puts the robot back on the original heading.
                self.finish_obstacle_bypass()
                return (0.0, 0.0, 'OBSTACLE BYPASS DONE -> SEEK RIGHT')

            turn = clamp(
                self.obstacle_bypass_turn_kp * yaw_error,
                -self.obstacle_bypass_turn_limit,
                self.obstacle_bypass_turn_limit
            )

            return (
                0.0,
                turn,
                f'OBSTACLE {phase} err={yaw_error_deg:+.1f}deg'
            )

        # -------------------------------------------------------------
        # STRAIGHT phases: odometry gives travelled distance while yaw hold
        # keeps each side of the rectangle straight.
        # -------------------------------------------------------------
        if phase == 'STRAIGHT_OUT':
            target_yaw = wrap_angle(base + math.pi / 2.0)
            target_distance = self.obstacle_bypass_lateral_m
            next_phase = 'TURN_RIGHT_FORWARD'
            next_yaw = base
            state = 'OBSTACLE MOVE LEFT'

        elif phase == 'STRAIGHT_PASS':
            target_yaw = base
            target_distance = self.obstacle_bypass_forward_m
            next_phase = 'TURN_RIGHT_IN'
            next_yaw = wrap_angle(base - math.pi / 2.0)
            state = 'OBSTACLE PASS FORWARD'

        elif phase == 'STRAIGHT_RETURN':
            target_yaw = wrap_angle(base - math.pi / 2.0)
            target_distance = self.obstacle_bypass_lateral_m
            next_phase = 'TURN_LEFT_RESUME'
            next_yaw = base
            state = 'OBSTACLE RETURN RIGHT'

        else:
            self.finish_obstacle_bypass()
            return (0.0, 0.0, 'OBSTACLE BYPASS UNKNOWN -> SEEK RIGHT')

        travelled = self.obstacle_bypass_distance()

        if travelled >= target_distance:
            self.obstacle_bypass_set_phase(next_phase, next_yaw)
            return (0.0, 0.0, f'{state} DONE')

        yaw_error = wrap_angle(target_yaw - current)
        turn = clamp(
            0.85 * self.obstacle_bypass_turn_kp * yaw_error,
            -0.20,
            0.20
        )

        return (
            min(self.obstacle_bypass_speed, self.max_speed),
            turn,
            f'{state} {travelled:.2f}/{target_distance:.2f}m'
        )

    def publish_frozen_lane_debug(
        self,
        image,
        near_y,
        mid_y,
        look_y,
        far_y,
        status
    ):
        """Keep lane perception/debug alive without mutating controller state."""
        white = self.create_white_mask(image)
        h = image.shape[0]
        white[:int(h * self.roi_top_ratio), :] = 0

        (
            _all_segments,
            lane_segments,
            left_short_lane_segments,
            thickness_rejected,
        ) = self.extract_segments(white, near_y)

        visible_lane_segments = lane_segments + left_short_lane_segments
        front = self.range_sector_percentile(
            0.0, self.front_sector_deg, self.front_percentile)
        self.publish_debug(
            image,
            status,
            visible_lane_segments,
            thickness_rejected,
            [],
            [],
            None,
            near_y,
            mid_y,
            look_y,
            far_y,
            front,
            0.0,
        )

    def publish_stop_freeze_lane_debug(
        self,
        image,
        near_y,
        mid_y,
        look_y,
        far_y
    ):
        left = max(0.0, self.stop_freeze_until - time.monotonic())
        self.publish_frozen_lane_debug(
            image, near_y, mid_y, look_y, far_y,
            f'STOP FREEZE {left:.1f}s')

    def publish_traffic_light_hold_lane_debug(
        self,
        image,
        near_y,
        mid_y,
        look_y,
        far_y
    ):
        self.publish_frozen_lane_debug(
            image, near_y, mid_y, look_y, far_y,
            f'LIGHT {self.traffic_light_confirmed_state} HOLD')

    # =================================================================
    # MULTI-BAND STRIP CENTROID LANE TRACKER (Gold-Standard AV Perception)
    # =================================================================

    def track_lane_multiband(self, image):
        h, w = image.shape[:2]

        # 1. High-contrast white lane mask (Reject shiny mat reflections)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 195], dtype=np.uint8)
        upper = np.array([180, 80, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Clear upper horizon & apply dynamic Trapezoid Envelope (cuts off outer mat edges and straight branches)
        mask[:250, :] = 0
        for y_idx in range(250, h):
            ratio = (y_idx - 250.0) / (440.0 - 250.0)
            left_bound = int(80.0 - ratio * (80.0 - 20.0))
            right_bound = int(455.0 + ratio * (580.0 - 455.0))
            mask[y_idx, :left_bound] = 0
            mask[y_idx, right_bound:] = 0

        # 2. Multi-band horizontal slice inspection (bottom to top)
        sample_ys = [410, 380, 350, 320, 290]
        band_h = 20

        def get_peaks(slice_m):
            col_hist = np.sum(slice_m > 0, axis=0)
            peaks = []
            in_p = False
            run = []
            for x in range(len(col_hist)):
                if col_hist[x] >= 4:
                    run.append(x)
                    in_p = True
                else:
                    if in_p and len(run) >= 3:
                        weights = col_hist[run]
                        c = float(np.sum(np.array(run) * weights) / max(1, np.sum(weights)))
                        peaks.append(c)
                    run = []
                    in_p = False
            if in_p and len(run) >= 3:
                weights = col_hist[run]
                c = float(np.sum(np.array(run) * weights) / max(1, np.sum(weights)))
                peaks.append(c)
            return sorted(peaks)

        right_pts = []
        centers = []
        prev_x = getattr(self, 'last_right_anchor_x', None)

        for y in sample_ys:
            slice_m = mask[y - band_h // 2:y + band_h // 2, :]
            peaks = get_peaks(slice_m)
            hw = 88.0 + (y - 290.0) * (245.0 - 88.0) / (410.0 - 290.0)
            nominal_right_x = 320.0 + hw

            if not right_pts:
                # Bottom anchor layer: look for right lane candidates
                cands = [p for p in peaks if 360 <= p <= 575]
                if cands:
                    target_ref = prev_x if prev_x is not None else nominal_right_x
                    best_x = min(cands, key=lambda p: abs(p - target_ref))
                    right_pts.append((int(best_x), y))
                    centers.append((int(best_x - hw), y))
                    self.last_right_anchor_x = best_x
            else:
                last_x = right_pts[-1][0]
                # Curvature-adaptive window: in left turns dx can be -50 to -80px
                cands = [p for p in peaks if (last_x - 85.0) <= p <= (last_x + 15.0)]
                if cands:
                    # If multiple candidates (e.g. inner curve vs outer branch), prefer inner curve
                    if len(cands) > 1 and (max(cands) - min(cands)) > 45.0:
                        best_x = min(cands)
                    else:
                        best_x = min(cands, key=lambda p: abs(p - (last_x - 38.0)))
                    right_pts.append((int(best_x), y))
                    centers.append((int(best_x - hw), y))
                else:
                    best_x = last_x - 38.0
                    right_pts.append((int(best_x), y))
                    centers.append((int(best_x - hw), y))

        # 3. Compute Steering from Lookahead Center (Pure Pursuit)
        if centers:
            self.multiband_lost_count = 0
            near_c = centers[0][0]
            look_idx = min(2, len(centers) - 1)
            look_c = centers[look_idx][0]

            lat_err = near_c - 320.0
            look_err = look_c - 320.0

            raw_turn = -self.kp_lateral * (look_err / 160.0)
            turn = (1.0 - self.steer_alpha) * self.last_turn + self.steer_alpha * raw_turn
            turn = float(clamp(turn, -self.max_turn, self.max_turn))
            self.last_turn = turn

            speed = float(clamp(self.max_speed * (1.0 - 0.20 * abs(turn)), 0.070, 0.12))
            state_str = f'RIGHT-LINE ({len(right_pts)}/5)'
        else:
            self.multiband_lost_count = getattr(self, 'multiband_lost_count', 0) + 1
            if self.multiband_lost_count <= 5:
                turn = self.last_turn * 0.88
                speed = max(0.065, self.max_speed * 0.8)
                state_str = f'GLIDE ({self.multiband_lost_count})'
            else:
                turn = 0.0
                speed = 0.03
                state_str = 'SEARCHING'
            lat_err = 0.0
            look_err = 0.0

        # 4. Clean visualization
        debug = image.copy()
        if len(right_pts) >= 2:
            cv2.polylines(
                debug,
                [np.array(right_pts, dtype=np.int32)],
                False,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )
        for pt in right_pts:
            cv2.circle(debug, pt, 6, (0, 255, 0), -1, cv2.LINE_AA)

        if len(centers) >= 2:
            cv2.polylines(
                debug,
                [np.array(centers, dtype=np.int32)],
                False,
                (255, 255, 0),
                3,
                cv2.LINE_AA
            )
        for pt in centers:
            cv2.circle(debug, pt, 6, (255, 0, 255), -1, cv2.LINE_AA)

        cv2.line(debug, (320, 470), (320, 435), (100, 200, 255), 2, cv2.LINE_AA)

        # Semi-transparent HUD Card
        overlay = debug.copy()
        cv2.rectangle(overlay, (12, 12), (370, 82), (18, 22, 30), -1)
        cv2.addWeighted(overlay, 0.70, debug, 0.30, 0, debug)
        cv2.rectangle(debug, (12, 12), (370, 82), (70, 85, 105), 1, cv2.LINE_AA)

        steer_dir = 'LEFT' if turn > 0.04 else ('RIGHT' if turn < -0.04 else 'STRAIGHT')
        cv2.putText(
            debug,
            f'LANE: {state_str}',
            (22, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            debug,
            f'STEER: {turn:+.2f} rad/s ({steer_dir})',
            (22, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            debug,
            f'eLat: {lat_err:+.0f}px | eLook: {look_err:+.0f}px',
            (22, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 205, 230),
            1,
            cv2.LINE_AA
        )

        self.log_every(
            0.30,
            'multiband',
            f'MULTIBAND | {state_str} | eLat={lat_err:+.0f}px eLook={look_err:+.0f}px | v={speed:.3f} w={turn:+.2f}'
        )

        return speed, turn, debug

    # =================================================================
    # MAIN CONTROL
    # =================================================================

    def control(self):

        # =============================================================
        # CAMERA
        # =============================================================

        if self.image is None:

            self.stop()

            return

        image = (
            self.image.copy()
        )

        h, w = (
            image.shape[:2]
        )

        near_y = int(
            h
            *
            self.near_y_ratio
        )

        mid_y = int(
            h
            *
            self.mid_y_ratio
        )

        look_y = int(
            h
            *
            self.lookahead_y_ratio
        )

        far_y = int(
            h
            *
            self.far_y_ratio
        )

        # =============================================================
        # STOP SIGN FREEZE
        #
        # Continue camera lane extraction/debug, but do not advance or reset any
        # lane/lost/recovery/obstacle state while the STOP hold is active.
        # =============================================================
        if not self.lane_only_mode and self.stop_freeze_is_active():
            self.stop()
            self.publish_stop_freeze_lane_debug(
                image, near_y, mid_y, look_y, far_y)
            return

        # =============================================================
        # TRAFFIC LIGHT HOLD
        # RED/YELLOW confirmed for the strict 0.8 s window freeze motion and
        # controller state. GREEN confirmed with the same rule releases it.
        # =============================================================
        if not self.lane_only_mode and self.traffic_light_hold_is_active():
            self.stop()
            self.publish_traffic_light_hold_lane_debug(
                image, near_y, mid_y, look_y, far_y)
            self.log_every(
                0.5,
                'traffic_light_hold',
                f'LIGHT {self.traffic_light_confirmed_state} -> STOP')
            return

        # =============================================================
        # LIDAR SAFETY
        #
        # This runs BEFORE lane/lost behaviour.
        # Therefore long-lost straight mode still stops for obstacles.
        # =============================================================

        front = (
            self.range_sector_percentile(
                0.0,
                self.front_sector_deg,
                self.front_percentile
            )
        )

        pitch_deg = abs(
            math.degrees(
                self.imu_pitch
            )
        )

        on_ramp = (
            pitch_deg
            >=
            self.ramp_pitch_deg
        )

        active_stop = (
            self.ramp_stop_distance
            if on_ramp
            else
            self.stop_distance
        )

        if (
            front
            <
            active_stop
        ):

            self.obstacle_count += 1

        else:

            self.obstacle_count = 0

        # Once a bypass has started, it owns motion until the rectangular
        # manoeuvre is complete. Camera detection is paused independently by
        # detection_tick(), so no sign/light/person state can change mid-bypass.
        if not self.lane_only_mode and self.obstacle_bypass_active:

            command = self.obstacle_bypass_command()

            if command is not None:
                speed, turn, state = command
                self.drive(speed, turn)
                self.log_every(
                    0.25,
                    'obstacle_bypass',
                    f'{state} | front={front:.2f} v={speed:.3f} w={turn:+.2f}'
                )
                return

        # =============================================================
        # CROSSWALK PEDESTRIAN REACTION
        #
        # After the sign disappears the state machine spends 3 s in DELAY.
        # Then camera pink-person detection is enabled.  A stop requires BOTH
        # a person inside the forward image zone and a front LiDAR obstacle.
        # If clear, start a 1.5 s 0.20 m/s lane-follow boost.
        # =============================================================
        if not self.lane_only_mode:
            self.crosswalk_boost_is_active()  # expire the timer if needed

            if self.crosswalk_phase in ('WATCH', 'BOOST'):
                sample_fresh = self.crosswalk_person_sample_is_fresh()

                # At 5 Hz this normally lasts only a fraction of a second. If LiDAR
                # already sees something while waiting for the matching camera sample,
                # stop conservatively instead of deciding CLEAR from stale vision.
                if not sample_fresh and front < active_stop:
                    self.stop()
                    self.log_every(
                        0.25, 'crosswalk_wait_camera',
                        f'CROSSWALK waiting fresh person frame | front={front:.2f}')
                    return

                if sample_fresh and self.crosswalk_person_obstacle_match(
                        front, active_stop):
                    if self.crosswalk_phase == 'BOOST':
                        self.interrupt_crosswalk_boost_for_person()
                    self.stop()
                    self.log_every(
                        0.20, 'crosswalk_person_stop',
                        f'CROSSWALK PERSON + LIDAR -> STOP | front={front:.2f}')
                    return

                if (self.crosswalk_phase == 'WATCH' and sample_fresh and
                        front >= active_stop):
                    self.start_crosswalk_boost(time.monotonic())

        if (
            self.obstacle_count
            >=
            self.obstacle_confirm_frames
        ):

            if self.lane_only_mode:
                self.stop()
                self.log_every(
                    0.5,
                    'obstacle',
                    (
                        f'OBSTACLE STOP (lane_only) | '
                        f'front={front:.2f} pitch={pitch_deg:.1f}'
                    )
                )
                return

            # During the armed crosswalk sequence the front obstacle is handled
            # by the camera+LiDAR pedestrian gate above.  DELAY intentionally
            # keeps lane-following for the requested 3 seconds; WATCH/BOOST stop
            # only on the corroborated pink-person condition.
            if self.crosswalk_owns_front_obstacle():
                self.log_every(
                    0.5, 'crosswalk_obstacle_owned',
                    f'CROSSWALK owns front LiDAR | phase={self.crosswalk_phase} '
                    f'front={front:.2f}')
            else:
                # Only when genuinely tracking RIGHT inside the HW section do we
                # replace the default hard stop with the rectangular bypass.
                if (
                    self.highway_mode
                    and
                    self.lane_locked
                    and
                    self.tracked_lane_side == 'right'
                    and
                    not on_ramp
                ):

                    self.start_obstacle_bypass()
                    self.stop()
                    return

                # Before HW_ENTRY and after HW_EXIT: stop only while obstacle is
                # present. Once LiDAR clears, normal lane control resumes.
                self.stop()

                mode_text = (
                    'HIGHWAY ON but bypass unavailable'
                    if self.highway_mode
                    else 'STOP/WAIT mode'
                )
                self.log_every(
                    0.5,
                    'obstacle',
                    (
                        f'OBSTACLE STOP | {mode_text} | '
                        f'front={front:.2f} pitch={pitch_deg:.1f}'
                    )
                )

                return

        # =============================================================
        # PERCEPTION
        # =============================================================

        # Multi-Band Strip Centroid Lane Tracker (Gold-Standard AV Perception)
        if self.lane_only_mode:
            speed, turn, debug_img = self.track_lane_multiband(image)
            self.drive(speed, turn)
            self.publish_lane_debug_frame(debug_img)
            return

        white = (
            self.create_white_mask(
                image
            )
        )

        white[
            :int(
                h
                *
                self.roi_top_ratio
            ),
            :
        ] = 0

        (
            all_segments,
            lane_segments,
            left_short_lane_segments,
            thickness_rejected
        ) = self.extract_segments(
            white,
            near_y
        )

        left_lane_segments = (
            lane_segments
            +
            left_short_lane_segments
        )

        # =============================================================
        # INITIAL ACQUIRE
        # =============================================================

        if not self.lane_locked:

            seed = (
                self.choose_initial_seed(
                    lane_segments,
                    w,
                    near_y
                )
            )

            chain = (
                self.build_chain(
                    seed,
                    lane_segments
                )
            )

            measurement = (
                self.chain_measurement(
                    chain,
                    near_y,
                    mid_y,
                    look_y,
                    far_y
                )
            )

            chain_thickness = (
                self.chain_median_thickness(
                    chain
                )
            )

            if (
                measurement
                is not None
            ):

                if self.acquire_samples:

                    previous = (
                        self.acquire_samples[
                            -1
                        ]
                    )

                    consistent = (
                        abs(
                            measurement[
                                'x_near'
                            ]
                            -
                            previous[
                                'x_near'
                            ]
                        )
                        <
                        70.0
                        and
                        angle_diff_deg(
                            measurement[
                                'path_angle'
                            ],
                            previous[
                                'path_angle'
                            ]
                        )
                        <
                        55.0
                    )

                else:

                    consistent = True

                if consistent:

                    self.acquire_samples.append(
                        measurement
                    )

                    if (
                        chain_thickness
                        is not None
                    ):

                        self.acquire_thickness_samples.append(
                            chain_thickness
                        )

                else:

                    self.acquire_samples = [
                        measurement
                    ]

                    if (
                        chain_thickness
                        is None
                    ):

                        self.acquire_thickness_samples = []

                    else:

                        self.acquire_thickness_samples = [
                            chain_thickness
                        ]

            else:

                self.acquire_samples = []
                self.acquire_thickness_samples = []

            self.stop()

            if (
                len(
                    self.acquire_samples
                )
                >=
                self.acquire_frames
            ):

                recent = (
                    self.acquire_samples[
                        -self.acquire_frames:
                    ]
                )

                locked = {
                    'x_near':
                        float(
                            np.median(
                                [
                                    q['x_near']
                                    for q in recent
                                ]
                            )
                        ),

                    'x_mid':
                        float(
                            np.median(
                                [
                                    q['x_mid']
                                    for q in recent
                                ]
                            )
                        ),

                    'x_look':
                        float(
                            np.median(
                                [
                                    q['x_look']
                                    for q in recent
                                ]
                            )
                        ),

                    'x_far':
                        float(
                            np.median(
                                [
                                    q['x_far']
                                    for q in recent
                                ]
                            )
                        ),

                    'path_angle':
                        float(
                            np.median(
                                [
                                    q['path_angle']
                                    for q in recent
                                ]
                            )
                        )
                }

                self.set_track_from_measurement(
                    locked
                )

                self.reference_path_angle_deg = (
                    self.track_path_angle_deg
                )

                self.reference_preview_dx = (
                    self.track_x_near
                    -
                    self.track_x_look
                )

                self.right_reference_path_angle_deg = (
                    self.reference_path_angle_deg
                )

                self.left_reference_path_angle_deg = (
                    -self.right_reference_path_angle_deg
                )

                self.right_reference_preview_dx = (
                    self.reference_preview_dx
                )

                self.left_reference_preview_dx = (
                    -self.right_reference_preview_dx
                )

                self.right_target_x_near = (
                    self.track_x_near
                    +
                    self.target_shift_px
                )

                self.left_target_x_near = (
                    float(w)
                    -
                    self.right_target_x_near
                )

                self.target_x_near = (
                    self.right_target_x_near
                )

                self.tracked_lane_side = 'right'
                self.recovery_lane_side = None

                self.last_good_yaw = (
                    self.current_yaw()
                )

                self.last_good_turn = 0.0

                self.last_chain = (
                    chain
                )

                self.lane_locked = True
                self.get_logger().info(
                    f'LANE LOCKED: Target x_near={self.target_x_near:.1f}px | '
                    f'Heading={self.reference_path_angle_deg:.1f}deg | '
                    f'Preview_dx={self.reference_preview_dx:.1f}px (CENTERED)'
                )

                self.lost_count = 0
                self.lost_cardinal_target = None

                if (
                    self.acquire_thickness_samples
                ):

                    recent_thickness = (
                        self.acquire_thickness_samples[
                            -self.acquire_frames:
                        ]
                    )

                    self.reference_lane_thickness = float(
                        np.median(
                            np.asarray(
                                recent_thickness,
                                dtype=np.float32
                            )
                        )
                    )

                thickness_display = (
                    float('nan')
                    if
                    self.reference_lane_thickness
                    is None
                    else
                    self.reference_lane_thickness
                )

                self.get_logger().info(
                    (
                        'RIGHT POLYLINE LOCKED | '
                        f'N={self.track_x_near:.1f} '
                        f'LOOK={self.track_x_look:.1f} '
                        f'path={self.track_path_angle_deg:+.1f}deg | '
                        f'THICKNESS={thickness_display:.1f}px | '
                        f'yaw={math.degrees(self.last_good_yaw):+.1f}'
                    )
                )

            self.publish_debug(
                image,
                (
                    f'ACQUIRE '
                    f'{len(self.acquire_samples)}/'
                    f'{self.acquire_frames}'
                ),
                lane_segments,
                thickness_rejected,
                chain,
                [],
                measurement,
                near_y,
                mid_y,
                look_y,
                far_y,
                front,
                0.0
            )

            return

        # =============================================================
        # NORMAL TRACK
        # =============================================================

        tracking_lane_segments = (
            left_lane_segments
            if self.tracked_lane_side == 'left'
            else lane_segments
        )

        seed = (
            self.choose_tracking_seed(
                tracking_lane_segments
            )
        )

        chain = (
            self.build_chain(
                seed,
                tracking_lane_segments
            )
        )

        measurement = (
            self.chain_measurement(
                chain,
                near_y,
                mid_y,
                look_y,
                far_y
            )
        )

        # During the SEEK BOTH window use the coherent left vector, even if
        # the ordinary left chain happened to accept a crossing fragment.
        # The right boundary continues through the original pipeline.
        if (
            self.tracked_lane_side == 'left'
            and self.seek_both_window_active()
        ):
            measurement, chain = self.easy_left_seek_measurement(
                white,
                near_y,
                mid_y,
                look_y,
                far_y,
                {
                    'x_near': self.track_x_near,
                    'path_angle': self.track_path_angle_deg
                }
            )

        transverse = (
            self.detect_transverse(
                all_segments,
                (
                    chain
                    if chain
                    else
                    self.last_chain
                )
            )
        )

        track_ok = False

        if (
            measurement
            is not None
            and
            self.measurement_plausible(
                measurement
            )
        ):

            track_ok = (
                self.update_track(
                    measurement
                )
            )

        if track_ok:

            self.last_chain = (
                chain
            )

            self.lost_count = 0
            self.lost_cardinal_target = None

            # A stable lane lock means the 8 s SEEK BOTH failure condition
            # did not occur.  Reset its timer.  The normal SEEK BOTH persistence
            # window itself is still allowed to remain alive as before.
            self.reset_seek_both_fail_recovery(
                clear_turn_state=True
            )

            # Do not cancel an active SEEK BOTH window just because the lane
            # was reacquired.  Only re-arm a future window after the current
            # persistence window has actually expired.
            self.maybe_rearm_seek_both_after_valid_track()

            self.transverse_counter = 0

            self.recovery_samples = []
            self.recovery_lane_side = None

            self.last_good_yaw = (
                self.current_yaw()
            )

            self.maybe_update_thickness_reference(
                chain
            )

        # =============================================================
        # LEFT FALLBACK -> CONTINUOUSLY SEEK RIGHT BOUNDARY
        # =============================================================
        #
        # A left boundary recovered during LONG LOST is only temporary.
        # Keep controlling from it, but in parallel inspect the right half of
        # the image every frame.  Switch back only after the right boundary is
        # stable for reacquire_frames consecutive samples.
        #
        # This block runs only while the left track itself is valid.  If the
        # left track is lost, long_lost_dual_seek_active() above makes the LOST
        # recovery search both sides immediately, again with right priority.
        # =============================================================

        if (
            track_ok
            and
            self.tracked_lane_side == 'left'
        ):

            return_right_candidates = [
                seg
                for seg in lane_segments
                if (
                    seg['x_near']
                    >=
                    w * 0.50
                )
            ]

            return_right_seed = (
                self.choose_recovery_seed_for_side(
                    return_right_candidates,
                    w,
                    near_y,
                    'right'
                )
            )

            return_right_chain = (
                self.build_chain(
                    return_right_seed,
                    return_right_candidates
                )
            )

            return_right_measurement = (
                self.chain_measurement(
                    return_right_chain,
                    near_y,
                    mid_y,
                    look_y,
                    far_y
                )
            )

            if (
                return_right_measurement
                is not None
            ):

                if self.right_return_samples:

                    previous = (
                        self.right_return_samples[-1]
                    )

                    consistent = (
                        abs(
                            return_right_measurement['x_near']
                            -
                            previous['x_near']
                        )
                        <
                        75.0
                        and
                        angle_diff_deg(
                            return_right_measurement['path_angle'],
                            previous['path_angle']
                        )
                        <
                        70.0
                    )

                else:

                    consistent = True

                if consistent:

                    self.right_return_samples.append(
                        return_right_measurement
                    )

                else:

                    self.right_return_samples = [
                        return_right_measurement
                    ]

            else:

                self.right_return_samples = []

            if (
                len(self.right_return_samples)
                >=
                self.reacquire_frames
            ):

                recent_right = (
                    self.right_return_samples[
                        -self.reacquire_frames:
                    ]
                )

                recovered_right = {
                    'x_near': float(
                        np.median(
                            [q['x_near'] for q in recent_right]
                        )
                    ),
                    'x_mid': float(
                        np.median(
                            [q['x_mid'] for q in recent_right]
                        )
                    ),
                    'x_look': float(
                        np.median(
                            [q['x_look'] for q in recent_right]
                        )
                    ),
                    'x_far': float(
                        np.median(
                            [q['x_far'] for q in recent_right]
                        )
                    ),
                    'path_angle': float(
                        np.median(
                            [q['path_angle'] for q in recent_right]
                        )
                    )
                }

                self.set_track_from_measurement(
                    recovered_right
                )

                self.tracked_lane_side = 'right'

                if self.right_target_x_near is not None:
                    self.target_x_near = (
                        self.right_target_x_near
                    )

                if (
                    self.right_reference_path_angle_deg
                    is not None
                ):
                    self.reference_path_angle_deg = (
                        self.right_reference_path_angle_deg
                    )

                if (
                    self.right_reference_preview_dx
                    is not None
                ):
                    self.reference_preview_dx = (
                        self.right_reference_preview_dx
                    )

                self.last_chain = (
                    return_right_chain
                )

                self.recovery_samples = []
                self.recovery_lane_side = None
                self.right_return_samples = []

                self.maybe_update_thickness_reference(
                    return_right_chain
                )

                measurement = (
                    recovered_right
                )

                chain = (
                    return_right_chain
                )

                self.get_logger().info(
                    (
                        'LEFT FALLBACK -> RIGHT REACQUIRED | '
                        f'N={self.track_x_near:.1f} '
                        f'LOOK={self.track_x_look:.1f}'
                    )
                )

        elif self.tracked_lane_side != 'left':

            # Never carry stale right-return evidence into a later fallback.
            self.right_return_samples = []

        # =============================================================
        # LOST / REACQUIRE
        # =============================================================

        if not track_ok:

            if transverse:

                self.transverse_counter = (
                    self.transverse_hold_frames
                )

            elif (
                self.transverse_counter
                >
                0
            ):

                self.transverse_counter -= 1

            self.lost_count += 1

            if self.lost_count == 1:
                self.lost_cardinal_target = self.nearest_cardinal_yaw(
                    self.current_yaw()
                )

                self.recovery_lane_side = (
                    self.tracked_lane_side
                )

                self.get_logger().info(
                    'LOST CARDINAL TARGET | '
                    f'yaw={math.degrees(self.current_yaw()):+.1f}deg | '
                    f'target={math.degrees(self.lost_cardinal_target) % 360.0:.0f}deg'
                )

            # ---------------------------------------------------------
            # Recovery lane search.
            #
            # Before the first SEEK BOTH trigger, preserve the original
            # one-side recovery.  Once LONG LOST has exceeded the configured
            # delay, open a persistent SEEK BOTH window.
            #
            # That window survives reacquisition.  On a later fresh LOST, keep
            # the requested 3 s HOLD LAST LANE YAW first; after the hold, search
            # BOTH outer boundaries again while the same window is still alive.
            # When the window expires, recovery falls back to the cardinal IMU
            # anchors until a future recovery cycle is re-armed.
            #
            # If both are valid in the same frame, the right boundary wins.
            # lane_segments has already passed thickness filtering.
            # ---------------------------------------------------------

            dual_seek_active = (
                self.long_lost_dual_seek_active()
            )

            seek_right_after_timeout = (
                self.maybe_trigger_seek_right_after_failed_both(
                    dual_seek_active
                )
            )

            if seek_right_after_timeout:
                # The timeout helper closes the SEEK BOTH window.  From this
                # point onward this LOST episode is RIGHT-only recovery.
                dual_seek_active = False

            right_seed = None
            right_chain = []
            right_measurement = None

            left_seed = None
            left_chain = []
            left_measurement = None

            if dual_seek_active:

                right_candidates = [
                    seg
                    for seg in lane_segments
                    if (
                        seg['x_near']
                        >=
                        w * 0.50
                    )
                ]

                right_seed = (
                    self.choose_recovery_seed_for_side(
                        right_candidates,
                        w,
                        near_y,
                        'right'
                    )
                )

                right_chain = (
                    self.build_chain(
                        right_seed,
                        right_candidates
                    )
                )

                right_measurement = (
                    self.chain_measurement(
                        right_chain,
                        near_y,
                        mid_y,
                        look_y,
                        far_y
                    )
                )

                if (
                    self.recovery_lane_side == 'left'
                    and self.recovery_samples
                ):
                    left_reference = self.recovery_samples[-1]
                elif (
                    self.tracked_lane_side == 'left'
                    and self.track_x_near is not None
                ):
                    left_reference = {
                        'x_near': self.track_x_near,
                        'path_angle': self.track_path_angle_deg
                    }
                else:
                    left_reference = {
                        'path_angle': self.left_reference_path_angle_deg
                    }

                left_measurement, left_chain = (
                    self.easy_left_seek_measurement(
                        white,
                        near_y,
                        mid_y,
                        look_y,
                        far_y,
                        left_reference
                    )
                )
                left_seed = left_chain[0] if left_chain else None

                right_valid = (
                    right_measurement
                    is not None
                )

                left_valid = (
                    left_measurement
                    is not None
                )

                # Right priority when both sides are simultaneously valid.
                if (
                    right_valid
                    and
                    left_valid
                ):

                    selected_side = 'right'

                elif (
                    self.recovery_lane_side == 'right'
                    and
                    right_valid
                ):

                    selected_side = 'right'

                elif (
                    self.recovery_lane_side == 'left'
                    and
                    left_valid
                ):

                    selected_side = 'left'

                elif right_valid:

                    selected_side = 'right'

                elif left_valid:

                    selected_side = 'left'

                else:

                    selected_side = (
                        self.recovery_lane_side
                    )

                if (
                    selected_side
                    !=
                    self.recovery_lane_side
                ):

                    self.recovery_samples = []

                    self.get_logger().info(
                        (
                            'LONG LOST RECOVERY SIDE -> '
                            f'{selected_side.upper()}'
                        )
                    )

                self.recovery_lane_side = (
                    selected_side
                )

                if selected_side == 'right':

                    recovery_seed = (
                        right_seed
                    )

                    recovery_chain = (
                        right_chain
                    )

                    recovery_measurement = (
                        right_measurement
                    )

                elif selected_side == 'left':

                    recovery_seed = (
                        left_seed
                    )

                    recovery_chain = (
                        left_chain
                    )

                    recovery_measurement = (
                        left_measurement
                    )

                else:

                    recovery_seed = None
                    recovery_chain = []
                    recovery_measurement = None

            else:

                # After the 8 s SEEK BOTH timeout, finish the LEFT 90-degree
                # turn before accepting any lane.  This prevents a transient
                # line seen during the sweep from cancelling the manoeuvre.
                if (
                    self.seek_right_after_left_turn_active
                    and
                    not self.seek_right_turn_complete
                ):

                    selected_side = 'right'
                    self.recovery_lane_side = 'right'
                    recovery_seed = None
                    recovery_chain = []
                    recovery_measurement = None

                else:

                    selected_side = (
                        'right'
                        if self.seek_right_after_left_turn_active
                        else self.tracked_lane_side
                    )

                    self.recovery_lane_side = (
                        selected_side
                    )

                    if selected_side == 'left':

                        side_candidates = [
                            seg
                            for seg in left_lane_segments
                            if (
                                seg['x_near']
                                <
                                w * 0.50
                            )
                        ]

                    else:

                        side_candidates = [
                            seg
                            for seg in lane_segments
                            if (
                                seg['x_near']
                                >=
                                w * 0.50
                            )
                        ]

                    recovery_seed = (
                        self.choose_recovery_seed_for_side(
                            side_candidates,
                            w,
                            near_y,
                            selected_side
                        )
                    )

                    recovery_chain = (
                        self.build_chain(
                            recovery_seed,
                            side_candidates
                        )
                    )

                    recovery_measurement = (
                        self.chain_measurement(
                            recovery_chain,
                            near_y,
                            mid_y,
                            look_y,
                            far_y
                        )
                    )

            if (
                recovery_measurement
                is not None
            ):

                if self.recovery_samples:

                    previous = (
                        self.recovery_samples[
                            -1
                        ]
                    )

                    consistent = (
                        abs(
                            recovery_measurement[
                                'x_near'
                            ]
                            -
                            previous[
                                'x_near'
                            ]
                        )
                        <
                        75.0
                        and
                        angle_diff_deg(
                            recovery_measurement[
                                'path_angle'
                            ],
                            previous[
                                'path_angle'
                            ]
                        )
                        <
                        70.0
                    )

                else:

                    consistent = True

                if consistent:

                    self.recovery_samples.append(
                        recovery_measurement
                    )

                else:

                    self.recovery_samples = [
                        recovery_measurement
                    ]

            else:

                self.recovery_samples = []

            # ---------------------------------------------------------
            # REACQUIRE
            # ---------------------------------------------------------

            if (
                len(
                    self.recovery_samples
                )
                >=
                self.reacquire_frames
            ):

                recent = (
                    self.recovery_samples[
                        -self.reacquire_frames:
                    ]
                )

                recovered = {
                    'x_near':
                        float(
                            np.median(
                                [
                                    q['x_near']
                                    for q in recent
                                ]
                            )
                        ),

                    'x_mid':
                        float(
                            np.median(
                                [
                                    q['x_mid']
                                    for q in recent
                                ]
                            )
                        ),

                    'x_look':
                        float(
                            np.median(
                                [
                                    q['x_look']
                                    for q in recent
                                ]
                            )
                        ),

                    'x_far':
                        float(
                            np.median(
                                [
                                    q['x_far']
                                    for q in recent
                                ]
                            )
                        ),

                    'path_angle':
                        float(
                            np.median(
                                [
                                    q['path_angle']
                                    for q in recent
                                ]
                            )
                        )
                }

                self.set_track_from_measurement(
                    recovered
                )

                recovered_side = (
                    self.recovery_lane_side
                    if self.recovery_lane_side
                    in ('right', 'left')
                    else
                    self.tracked_lane_side
                )

                self.tracked_lane_side = (
                    recovered_side
                )

                if recovered_side == 'left':

                    # Follow the real left boundary, but control it as a
                    # temporary proxy for a right boundary 200 px to its right.
                    # Algebraically:
                    #   (left_x + offset) - right_target
                    # is the same lateral error as:
                    #   left_x - (right_target - offset)
                    if (
                        self.right_target_x_near
                        is not None
                    ):

                        self.target_x_near = clamp(
                            self.right_target_x_near
                            -
                            self.left_recovery_offset_px,
                            0.0,
                            float(w - 1)
                        )

                    elif (
                        self.left_target_x_near
                        is not None
                    ):

                        self.target_x_near = (
                            self.left_target_x_near
                        )

                    if (
                        self.left_reference_path_angle_deg
                        is not None
                    ):

                        self.reference_path_angle_deg = (
                            self.left_reference_path_angle_deg
                        )

                    if (
                        self.left_reference_preview_dx
                        is not None
                    ):

                        self.reference_preview_dx = (
                            self.left_reference_preview_dx
                        )

                    self.right_return_samples = []

                    self.get_logger().info(
                        (
                            'LEFT FALLBACK ACTIVE | '
                            f'offset={self.left_recovery_offset_px:.0f}px | '
                            'continuously seeking RIGHT'
                        )
                    )

                elif (
                    self.right_target_x_near is not None
                ):

                    self.target_x_near = (
                        self.right_target_x_near
                    )

                    if (
                        self.right_reference_path_angle_deg
                        is not None
                    ):

                        self.reference_path_angle_deg = (
                            self.right_reference_path_angle_deg
                        )

                    if (
                        self.right_reference_preview_dx
                        is not None
                    ):

                        self.reference_preview_dx = (
                            self.right_reference_preview_dx
                        )

                    self.right_return_samples = []

                self.last_chain = (
                    recovery_chain
                )

                self.last_good_yaw = (
                    self.current_yaw()
                )

                self.last_good_turn = 0.0
                self.last_turn = 0.0

                self.lost_count = 0
                self.lost_cardinal_target = None

                self.reset_seek_both_fail_recovery(
                    clear_turn_state=True
                )

                self.maybe_rearm_seek_both_after_valid_track()

                self.transverse_counter = 0

                self.recovery_samples = []
                self.recovery_lane_side = None

                self.maybe_update_thickness_reference(
                    recovery_chain
                )

                track_ok = True

                measurement = (
                    recovered
                )

                chain = (
                    recovery_chain
                )

                self.get_logger().info(
                    (
                        f'REACQUIRED {self.tracked_lane_side.upper()} POLYLINE | '
                        f'N={self.track_x_near:.1f} '
                        f'LOOK={self.track_x_look:.1f} '
                        f'yaw={math.degrees(self.current_yaw()):+.1f}'
                    )
                )

            # ---------------------------------------------------------
            # STILL LOST
            # ---------------------------------------------------------

            if not track_ok:

                # =====================================================
                # NEW V4.8:
                #
                # LONG LOST DOES NOT STOP.
                # =====================================================

                if (
                    self.lost_count
                    >
                    self.long_lost_frames
                ):

                    (
                        speed,
                        turn,
                        state,
                        yaw_error_deg
                    ) = self.long_lost_command()

                else:

                    (
                        speed,
                        turn,
                        state,
                        yaw_error_deg
                    ) = self.short_lost_command()

                self.drive(
                    speed,
                    turn
                )

                thickness_text = (
                    'None'
                    if
                    self.reference_lane_thickness
                    is None
                    else
                    f'{self.reference_lane_thickness:.1f}'
                )

                self.log_every(
                    0.25,
                    'lost',
                    (
                        f'{state} | '
                        f'lost={self.lost_count} '
                        f'yaw={math.degrees(self.current_yaw()):+.1f} '
                        f'target={math.degrees(self.lost_target_yaw()) % 360.0:.0f} '
                        f'err={yaw_error_deg:+.1f} | '
                        f'v={speed:.3f} '
                        f'w={turn:+.2f} | '
                        f'thRef={thickness_text}px '
                        f'accepted={len(lane_segments)} '
                        f'rejectTh={len(thickness_rejected)} '
                        f'side={self.recovery_lane_side or self.tracked_lane_side} '
                        f'longT={self.long_lost_elapsed_sec():.1f}s '
                        f'dual={int(self.long_lost_dual_seek_active())} '
                        f'sbRemain={self.seek_both_window_remaining_sec():.1f}s '
                        f'left90Right={int(self.seek_right_after_left_turn_active)} '
                        f'reacq='
                        f'{len(self.recovery_samples)}/'
                        f'{self.reacquire_frames}'
                    )
                )

                if transverse:

                    display_state = (
                        'TRANSVERSE -> '
                        +
                        state
                    )

                else:

                    display_state = (
                        state
                    )

                self.publish_debug(
                    image,
                    display_state,
                    lane_segments,
                    thickness_rejected,
                    recovery_chain,
                    transverse,
                    recovery_measurement,
                    near_y,
                    mid_y,
                    look_y,
                    far_y,
                    front,
                    turn
                )

                return

        # =============================================================
        # NORMAL CONTROL
        # =============================================================

        lateral_error_px = (
            self.track_x_near
            -
            self.target_x_near
        )

        lateral_turn = (
            -self.kp_lateral
            *
            (
                lateral_error_px
                /
                (
                    w / 2.0
                )
            )
        )

        heading_error_deg = (
            self.track_path_angle_deg
            -
            self.reference_path_angle_deg
        )

        heading_turn = (
            self.kp_heading
            *
            (
                heading_error_deg
                /
                45.0
            )
        )

        current_preview_dx = (
            self.track_x_near
            -
            self.track_x_look
        )

        preview_error_px = (
            current_preview_dx
            -
            self.reference_preview_dx
        )

        preview_effective_px = (
            self.preview_effective_error(
                preview_error_px
            )
        )

        preview_turn = (
            self.kp_preview
            *
            (
                preview_effective_px
                /
                (
                    w / 2.0
                )
            )
        )

        preview_turn = clamp(
            preview_turn,
            -self.preview_turn_limit,
            self.preview_turn_limit
        )

        raw_turn = (
            lateral_turn
            +
            heading_turn
            +
            preview_turn
        )

        # =============================================================
        # OPTIONAL RIGHT-WALL SAFETY
        # =============================================================

        right_distance = (
            self.range_sector_percentile(
                self.right_wall_angle_deg,
                self.right_wall_sector_deg,
                30.0
            )
        )

        wall_turn = 0.0

        if (
            self.right_wall_gain
            >
            0.0
            and
            math.isfinite(
                right_distance
            )
            and
            right_distance
            <
            self.right_wall_safe_distance
        ):

            wall_turn = (
                self.right_wall_gain
                *
                (
                    self.right_wall_safe_distance
                    -
                    right_distance
                )
            )

            raw_turn += (
                wall_turn
            )

        raw_turn = clamp(
            raw_turn,
            -self.max_turn,
            self.max_turn
        )

        # =============================================================
        # SMOOTH STEERING
        # =============================================================

        turn = (
            (
                1.0
                -
                self.steer_alpha
            )
            *
            self.last_turn
            +
            self.steer_alpha
            *
            raw_turn
        )

        turn = clamp(
            turn,
            -self.max_turn,
            self.max_turn
        )

        self.last_turn = turn
        self.last_good_turn = turn

        self.last_good_yaw = (
            self.current_yaw()
        )

        # =============================================================
        # SPEED
        # =============================================================

        turn_ratio = clamp(
            abs(
                turn
            )
            /
            max(
                self.max_turn,
                1e-6
            ),
            0.0,
            1.0
        )

        speed = (
            self.max_speed
            *
            (
                1.0
                -
                0.55
                *
                turn_ratio
            )
        )

        speed = max(
            0.042,
            speed
        )

        if on_ramp:

            speed = min(
                speed,
                0.060
            )

        # Only the accepted RAMP sign temporarily overrides the old
        # pitch-based ramp cap. Outside its 2 s window the 0.060 cap remains.
        ramp_boost_active = False if self.lane_only_mode else self.ramp_boost_is_active()
        green_light_fast_active = (
            False if self.lane_only_mode else (self.green_light_fast_is_active() and not on_ramp))
        crosswalk_boost_active = (
            False if self.lane_only_mode else (self.crosswalk_boost_is_active() and not on_ramp and
            front >= active_stop))

        linear_limit = None
        if ramp_boost_active:
            speed = self.ramp_boost_speed
            linear_limit = self.ramp_boost_speed
        elif green_light_fast_active:
            speed = self.green_light_speed
            linear_limit = self.green_light_speed
        elif crosswalk_boost_active:
            speed = self.crosswalk_boost_speed
            linear_limit = self.crosswalk_boost_speed

        self.drive(
            speed,
            turn,
            linear_limit=linear_limit
        )

        if self.tracked_lane_side == 'left':

            state = (
                'TRACK LEFT FALLBACK +200PX | SEEK RIGHT'
            )

            if on_ramp:
                state += ' RAMP'

        elif on_ramp:

            state = (
                'TRACK POLYLINE RAMP'
            )

        elif (
            abs(
                preview_effective_px
            )
            >
            0.0
        ):

            state = (
                'TRACK POLYLINE + PREVIEW'
            )

        else:

            state = (
                'TRACK POLYLINE'
            )

        if ramp_boost_active:
            state += (
                f' | RAMP BOOST {self.ramp_boost_speed:.2f}m/s '
                f'{max(0.0, self.ramp_boost_until - time.monotonic()):.1f}s')
        elif green_light_fast_active:
            state += (
                f' | GREEN FAST {self.green_light_speed:.2f}m/s')
        elif crosswalk_boost_active:
            state += (
                f' | CROSSWALK BOOST {self.crosswalk_boost_speed:.2f}m/s '
                f'{max(0.0, self.crosswalk_boost_until - time.monotonic()):.1f}s')

        if self.seek_both_window_active():
            state += (
                f' | SEEK BOTH MEM '
                f'{self.seek_both_window_remaining_sec():.1f}s'
            )

        self.publish_debug(
            image,
            state,
            lane_segments,
            thickness_rejected,
            chain,
            transverse,
            measurement,
            near_y,
            mid_y,
            look_y,
            far_y,
            front,
            turn,
            preview_error_px,
            preview_effective_px
        )

        thickness_text = (
            'None'
            if
            self.reference_lane_thickness
            is None
            else
            f'{self.reference_lane_thickness:.1f}'
        )

        self.log_every(
            0.30,
            'track',
            (
                f'{state} | '
                f'N={self.track_x_near:.0f}/'
                f'{self.target_x_near:.0f} '
                f'LOOK={self.track_x_look:.0f} '
                f'side={self.tracked_lane_side} '
                f'rightSeek={len(self.right_return_samples)}/'
                f'{self.reacquire_frames} '
                f'thRef={thickness_text}px | '
                f'eLat={lateral_error_px:+.0f}px '
                f'eHead={heading_error_deg:+.1f}deg '
                f'ePrevEff={preview_effective_px:+.0f}px | '
                f'LAT={lateral_turn:+.2f} '
                f'HEAD={heading_turn:+.2f} '
                f'PREV={preview_turn:+.2f} '
                f'WALL={wall_turn:+.2f} | '
                f'v={speed:.3f} '
                f'w={turn:+.2f}'
            )
        )


# =====================================================================
# Shutdown
# =====================================================================

def catch_sigterm():

    stopping = {
        'now': False
    }

    signal.signal(
        signal.SIGTERM,
        lambda *_:
        stopping.update(
            now=True
        )
    )

    return stopping


def spin(
    node,
    stopping
):

    while (
        rclpy.ok()
        and
        not stopping[
            'now'
        ]
    ):

        try:

            rclpy.spin_once(
                node,
                timeout_sec=0.1
            )

        except Exception:

            if (
                stopping[
                    'now'
                ]
                or
                not rclpy.ok()
            ):

                break

            raise


def main(args=None):

    rclpy.init(
        args=args
    )

    stopping = (
        catch_sigterm()
    )

    node = Starter()

    try:

        spin(
            node,
            stopping
        )

    except (
        KeyboardInterrupt,
        ExternalShutdownException
    ):

        pass

    finally:

        if rclpy.ok():

            node.stop()

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':

    main()
