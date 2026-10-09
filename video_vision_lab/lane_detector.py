#!/usr/bin/env python3
"""
Anti-Glare Lane Perception Engine (UEH Team 7 - CRC 2026).
Bộ lọc chống chói quang học và thuật toán bám vạch kẻ đường chính xác cao.
Được tinh chỉnh tối ưu cho sa bàn bạt đen phản chiếu ánh đèn trần LED.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from . import config


class AntiGlareLaneDetector:
    """Bộ xử lý nhận diện làn đường chống chói & bám vạch liền bên phải."""

    def __init__(
        self,
        target_right_x: float = config.TARGET_RIGHT_X,
        roi_top_ratio: float = config.ROI_TOP_RATIO,
        near_y_ratio: float = config.NEAR_Y_RATIO,
        lookahead_y_ratio: float = config.LOOKAHEAD_Y_RATIO,
    ):
        self.target_right_x = target_right_x
        self.roi_top_ratio = roi_top_ratio
        self.near_y_ratio = near_y_ratio
        self.lookahead_y_ratio = lookahead_y_ratio

        # Kernel Top-Hat hình chữ nhật ngang loại bỏ triệt để bóng đèn trần to loang rộng
        self.kernel_tophat = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.TOPHAT_KERNEL_W, config.TOPHAT_KERNEL_H)
        )

        # Bộ nhớ theo dõi liên tục vạch phải (Temporal Memory & Smooth Filter)
        self.last_poly: Optional[np.ndarray] = None
        self.last_curvature: float = 0.0
        self.last_error: float = 0.0
        self.consecutive_lost: int = 0
        self.jump_reject_count: int = 0

    def create_anti_glare_mask(self, image: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """Tạo mask nhị phân miễn nhiễm chói đèn trần, đồng thời phát hiện vạch dừng/người đi bộ.
        
        Returns:
            (clean_mask, stop_line_detected, crosswalk_detected)
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. Lọc Top-Hat: Tách cấu trúc vạch hẹp (15-30px), khử vệt chói loang rộng (>60px)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel_tophat)
        _, tophat_bin = cv2.threshold(tophat, config.TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)

        # 2. Điều kiện màu HSV: V >= WHITE_V_MIN, S <= WHITE_S_MAX (vạch trắng phi màu)
        val_mask = cv2.inRange(
            hsv,
            np.array([0, 0, config.WHITE_V_MIN], dtype=np.uint8),
            np.array([180, config.WHITE_S_MAX, 255], dtype=np.uint8),
        )

        # 3. Kết hợp logic AND
        combined = cv2.bitwise_and(tophat_bin, val_mask)

        # 4. Cắt bỏ chân trời ROI và mép ngoài
        y_cut = int(h * self.roi_top_ratio)
        combined[:y_cut, :] = 0
        combined[:, :15] = 0
        combined[:, w - 15:] = 0

        # 5. Phát hiện Vạch dừng (Stop line) & Vạch đi bộ (Zebra crossing)
        stop_line_detected = False
        crosswalk_detected = False
        
        # Kiểm tra vùng trước mũi xe [y_cut : h - 40, 100 : w - 80]
        check_roi = combined[y_cut:int(h * 0.90), 100:w - 60]
        # Tính hình chiếu ngang (horizontal projection) để tìm vạch ngang
        h_proj = np.sum(check_roi > 0, axis=1)
        if len(h_proj) > 0 and np.max(h_proj) > 90:
            stop_line_detected = True

        # Đếm các khối hình chữ nhật nằm ngang (zebra blocks)
        cnts_all, _ = cv2.findContours(check_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        horizontal_blocks = 0
        for c in cnts_all:
            area = cv2.contourArea(c)
            if 150 <= area <= 4000:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw > bh * 1.4 and bw >= 35:
                    horizontal_blocks += 1
        if horizontal_blocks >= 3:
            crosswalk_detected = True
            stop_line_detected = True

        # 6. Giới hạn hành lang làn phải (Right Corridor Polygon)
        corridor = np.zeros_like(combined)
        poly_corridor = np.array([
            [config.CORRIDOR_TOP_LEFT_X, y_cut],
            [w - config.CORRIDOR_MARGIN_RIGHT, y_cut],
            [w - config.CORRIDOR_MARGIN_RIGHT, h],
            [config.CORRIDOR_BOT_LEFT_X, h],
        ], dtype=np.int32)
        cv2.fillPoly(corridor, [poly_corridor], 255)
        clean_mask = cv2.bitwise_and(combined, corridor)

        # 7. Phân biệt vạch chính dày vs mép sàn ngoài bằng minAreaRect
        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours_info = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= config.MIN_LINE_AREA:
                    rect = cv2.minAreaRect(c)
                    length = max(rect[1])
                    thickness = min(rect[1])
                    if length >= config.MIN_LINE_LENGTH:
                        contours_info.append((c, area, thickness, length))

            if contours_info:
                max_thickness = max(info[2] for info in contours_info)
                if max_thickness >= config.MIN_THICKNESS_THRESH:
                    thick_mask = np.zeros_like(clean_mask)
                    min_allowed = max(8.0, 0.32 * max_thickness)
                    for c, area, thickness, length in contours_info:
                        if thickness >= min_allowed:
                            cv2.drawContours(thick_mask, [c], -1, 255, -1)
                    clean_mask = thick_mask

        return clean_mask, stop_line_detected, crosswalk_detected

    def track_lane(self, mask: np.ndarray) -> Dict[str, Any]:
        """Dò quét vạch phải bằng Sliding Windows và khớp đa thức bậc 2."""
        h, w = mask.shape[:2]
        pts: List[Tuple[float, float]] = []
        win_h = config.WINDOW_HEIGHT
        y_limit = int(h * self.roi_top_ratio)
        y_start = int(h * self.near_y_ratio)

        prev_x: Optional[float] = None
        dx = 0.0

        for y_bot in range(y_start, y_limit, -win_h):
            y_top = y_bot - win_h
            y_mid = (y_top + y_bot) / 2.0
            cand_x: Optional[float] = None

            # Cửa sổ quét quanh vạch dự đoán
            if prev_x is None:
                if self.last_poly is not None:
                    exp_base = self.eval_poly(self.last_poly, y_mid)
                    if exp_base is not None and 140 <= exp_base <= w - 15:
                        x_start = max(130, int(exp_base - config.WINDOW_HALF_W))
                        x_end = min(w - 15, int(exp_base + config.WINDOW_HALF_W))
                    else:
                        x_start = 170
                        x_end = w - 15
                else:
                    x_start = 170
                    x_end = w - 15
            else:
                x_start = max(110, int(prev_x - config.WINDOW_HALF_W))
                x_end = min(w - 15, int(prev_x + config.WINDOW_HALF_W))

            if x_end > x_start:
                slice_data = mask[y_top:y_bot, x_start:x_end]
                col_sum = np.sum(slice_data > 0, axis=0)

                if len(col_sum) > 0 and np.max(col_sum) >= 2:
                    smoothed = np.convolve(col_sum, np.ones(5) / 5.0, mode='same')
                    peaks = []
                    in_p = False
                    p_start = 0
                    for i in range(len(smoothed)):
                        if smoothed[i] >= 2.0:
                            if not in_p:
                                in_p = True
                                p_start = i
                        else:
                            if in_p:
                                in_p = False
                                peaks.append((p_start, i, i - p_start, float(np.max(smoothed[p_start:i]))))
                    if in_p:
                        peaks.append((p_start, len(smoothed), len(smoothed) - p_start, float(np.max(smoothed[p_start:]))))

                    if peaks:
                        # Ưu tiên vạch có bề ngang to hơn (peak width lớn nhất)
                        best_peak = max(peaks, key=lambda p: p[2])
                        if best_peak[2] >= 3:
                            cand_x = x_start + best_peak[0] + float(np.argmax(smoothed[best_peak[0]:best_peak[1]]))

            if cand_x is not None:
                pts.append((cand_x, y_mid))
                if prev_x is not None:
                    dx = cand_x - prev_x
                prev_x = cand_x
            elif prev_x is not None and abs(dx) > 0.0:
                prev_x = prev_x + dx * 0.65

        # Khớp đa thức
        poly: Optional[np.ndarray] = None
        status = "LOST"

        if len(pts) >= 3:
            py = [p[1] for p in pts]
            px = [p[0] for p in pts]
            try:
                poly = np.polyfit(py, px, 2)
                status = "TRACKING"
            except Exception:
                poly = None
        elif len(pts) == 2:
            py = [p[1] for p in pts]
            px = [p[0] for p in pts]
            try:
                m, c = np.polyfit(py, px, 1)
                poly = np.array([0.0, float(m), float(c)])
                status = "TRACKING_LINEAR"
            except Exception:
                poly = None

        near_y = float(h * self.near_y_ratio)
        look_y = float(h * self.lookahead_y_ratio)

        # Kiểm tra nhảy vọt tọa độ (Jump Rejection)
        if poly is not None:
            x_chk = self.eval_poly(poly, look_y)
            if self.last_poly is not None and x_chk is not None:
                x_old = self.eval_poly(self.last_poly, look_y)
                if x_old is not None:
                    jump = abs(x_chk - x_old)
                    if jump > config.OUTLIER_JUMP_THRESH:
                        self.jump_reject_count += 1
                        if self.jump_reject_count <= 6:
                            poly = self.last_poly
                            status = "HOLD_PREV"
                        else:
                            # Cua gắt thực sự sau 6 frame lệch liên tục
                            poly = 0.55 * poly + 0.45 * self.last_poly
                            self.jump_reject_count = 0
                    else:
                        self.jump_reject_count = 0
                        poly = 0.82 * poly + 0.18 * self.last_poly
            self.last_poly = poly
            self.consecutive_lost = 0
        else:
            # Mất vạch tạm thời: Giữ đa thức cũ tối đa 8 frames
            self.consecutive_lost += 1
            if self.last_poly is not None and self.consecutive_lost <= 8:
                poly = self.last_poly
                status = "HOLD_PREV"
            else:
                self.last_poly = None
                status = "LOST"

        # Tính toán các chỉ số dẫn hướng
        lookahead_x = self.eval_poly(poly, look_y) if poly is not None else None
        near_x = self.eval_poly(poly, near_y) if poly is not None else None

        curvature = self.calc_curvature(poly, look_y) if poly is not None else 0.0
        self.last_curvature = curvature

        # Độ lệch lái (Steering Error tính bằng pixel)
        if lookahead_x is not None:
            steering_error = float(lookahead_x - self.target_right_x)
            self.last_error = steering_error
        else:
            steering_error = self.last_error

        # Góc hướng (Heading tangent angle tính bằng độ)
        heading_angle = 0.0
        if poly is not None:
            # dx/dy = 2*a*y + b
            dx_dy = 2 * poly[0] * look_y + poly[1]
            # Góc so với trục thẳng đứng
            heading_angle = float(np.degrees(np.arctan(dx_dy)))

        return {
            "status": status,
            "poly": poly,
            "points": pts,
            "lookahead_pt": (lookahead_x, look_y) if lookahead_x is not None else None,
            "near_pt": (near_x, near_y) if near_x is not None else None,
            "steering_error": steering_error,
            "curvature": curvature,
            "heading_angle": heading_angle,
        }

    @staticmethod
    def eval_poly(poly: Optional[np.ndarray], y: float) -> Optional[float]:
        """Tính giá trị x từ y bằng đa thức bậc 2 hoặc bậc 1."""
        if poly is None:
            return None
        return float(poly[0] * (y**2) + poly[1] * y + poly[2])

    @staticmethod
    def calc_curvature(poly: Optional[np.ndarray], y: float) -> float:
        """Tính độ cong mặt đường signed curvature k = x'' / (1 + (x')^2)^(1.5)."""
        if poly is None:
            return 0.0
        dx_dy = 2.0 * poly[0] * y + poly[1]
        d2x_dy2 = 2.0 * poly[0]
        denom = (1.0 + dx_dy**2) ** 1.5
        if abs(denom) < 1e-6:
            return 0.0
        return float(d2x_dy2 / denom)

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Quy trình nhận diện vạch đường hoàn chỉnh."""
        clean_mask, stop_line, crosswalk = self.create_anti_glare_mask(image)
        lane_info = self.track_lane(clean_mask)
        lane_info["mask"] = clean_mask
        lane_info["stop_line_detected"] = stop_line
        lane_info["crosswalk_detected"] = crosswalk
        return lane_info
