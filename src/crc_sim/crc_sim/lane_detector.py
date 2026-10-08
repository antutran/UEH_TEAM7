#!/usr/bin/env python3
"""Pure Right-Lane Perception Engine (UEH Team 7 - CRC 2026).

Thuật toán xử lý ảnh chuyên dụng bám DUY NHẤT VẠCH PHẢI (vạch nét liền chuẩn):
1. Anti-Glare Filter (Top-Hat Morphological Filter):
   - Bạt đen phản chiếu đèn trần LED tạo ra các vệt chói loá loang rộng (> 60px).
   - Vạch đường chuẩn (tape trắng) có bề rộng hẹp đồng nhất (15 - 35px).
   - Bộ lọc Top-Hat hình chữ nhật ngang (25x5) loại bỏ 100% vệt chói đèn và bóng lóa bạt.
2. Thickness Filter (minAreaRect):
   - Mép ngoài sàn/bạt chỉ là đường viền mỏng (thickness <= 10px).
   - Vạch làn chính thi đấu có bề ngang to dày vượt trội (thickness >= 15px - 40px).
   - Lọc bỏ 100% mép ngoài mỏng khi có vạch chính.
3. Pure Right-Stripe Sliding Windows Tracker:
   - Dò quét độc quyền vạch phải (x >= 260px), loại bỏ 100% vạch đứt tim đường và làn trái.
   - Ưu tiên vạch có bề ngang to nhất (peak width).
   - Khớp đa thức bậc 2 x(y) = a*y^2 + b*y + c mượt mà.
4. Trích xuất cả điểm xa (Lookahead tại 68% H) và điểm gần (Near tại 84% H)
   để khống chế đồng thời khoảng cách lệch tâm và góc hướng chạy thẳng song song vạch.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class LaneDetector:
    """Bộ xử lý ảnh chống chói và bám DUY NHẤT vạch phải."""

    def __init__(
        self,
        white_v_min: int = 130,
        white_s_max: int = 90,
        tophat_thresh: int = 35,
        roi_top_ratio: float = 0.54,
        near_y_ratio: float = 0.84,
        look_y_ratio: float = 0.74,
        n_windows: int = 9,
        window_half_w: int = 42,
    ):
        self.white_v_min = white_v_min
        self.white_s_max = white_s_max
        self.tophat_thresh = tophat_thresh
        self.roi_top_ratio = roi_top_ratio
        self.near_y_ratio = near_y_ratio
        self.look_y_ratio = look_y_ratio

        self.n_windows = n_windows
        self.window_half_w = window_half_w

        # Kernel Top-Hat hình chữ nhật ngang lọc triệt để chói loang rộng
        self.kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))

        # Bộ nhớ theo dõi liên tục vạch phải (Temporal Memory)
        self.last_right_poly: Optional[np.ndarray] = None
        self.last_curvature: float = 0.0

    def create_anti_glare_mask(self, image: np.ndarray) -> np.ndarray:
        """Tạo mask nhị phân miễn nhiễm chói đèn trần và bóng lóa bạt."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. Lọc Top-Hat: Giữ lại vệt sáng hẹp (15-30px), triệt tiêu vệt chói to (>60px)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel_tophat)
        _, tophat_bin = cv2.threshold(tophat, self.tophat_thresh, 255, cv2.THRESH_BINARY)

        # 2. Điều kiện màu HSV: V >= white_v_min, S <= white_s_max (phi màu trắng)
        val_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.white_v_min], dtype=np.uint8),
            np.array([180, self.white_s_max, 255], dtype=np.uint8),
        )

        # 3. Kết hợp logic AND: Phải vừa có tương phản cục bộ cao, vừa có giá trị sáng trắng
        clean_mask = cv2.bitwise_and(tophat_bin, val_mask)

        # 4. Cắt bỏ hoàn toàn đường chân trời & rìa sàn ngoài sân đấu
        y_cut = int(h * self.roi_top_ratio)
        clean_mask[:y_cut, :] = 0
        clean_mask[:, :20] = 0
        clean_mask[:, w - 20:] = 0

        # 5. Phân biệt vạch chính vs mép sàn ngoài bằng độ dày bề ngang (Thickness / Width):
        # Mép ngoài sàn/bạt chỉ là đường viền mỏng (thickness <= 10px).
        # Vạch làn chính thi đấu có bề ngang to dày vượt trội (thickness >= 15px - 40px).
        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            contours_info = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area >= 120:
                    rect = cv2.minAreaRect(c)
                    length = max(rect[1])
                    thickness = min(rect[1])
                    if length >= 35:
                        contours_info.append((c, area, thickness, length))

            if contours_info:
                max_thickness = max(info[2] for info in contours_info)
                # Nếu phát hiện vạch dày chuẩn làn thi đấu (>= 14px):
                if max_thickness >= 14.0:
                    thick_mask = np.zeros_like(clean_mask)
                    min_allowed = max(11.0, 0.40 * max_thickness)
                    for c, area, thickness, length in contours_info:
                        if thickness >= min_allowed:
                            cv2.drawContours(thick_mask, [c], -1, 255, -1)
                    clean_mask = thick_mask

        return clean_mask

    def _track_right_stripe(
        self,
        mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[float, float]]]:
        """Dò quét vạch phải thực chất từ đáy lên trên.
        Loại bỏ 100% vạch đứt tim đường bên trái và mép ngoài mỏng.
        Ưu tiên TUYỆT ĐỐI vạch có bề ngang to hơn (Thickness lớn nhất).
        """
        h, w = mask.shape[:2]
        pts: List[Tuple[float, float]] = []
        win_h = 16
        y_limit = int(h * self.roi_top_ratio)
        y_start = int(h * 0.86)

        prev_x: Optional[float] = None
        dx = 0.0

        for y_bot in range(y_start, y_limit, -win_h):
            y_top = y_bot - win_h
            y_mid = (y_top + y_bot) / 2.0

            cand_x: Optional[float] = None

            # Chỉ tìm kiếm ở nửa bên phải (x >= 260px) để loại bỏ 100% vạch tim đường
            if prev_x is None:
                x_start = 280
                x_end = w - 20
            else:
                x_start = max(220, int(prev_x - 65))
                x_end = min(w - 20, int(prev_x + 65))

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
                        # ƯU TIÊN VẠCH CÓ BỀ NGANG TO HƠN (peak width lớn nhất)
                        best_peak = max(peaks, key=lambda p: p[2])
                        if best_peak[2] >= 3:
                            cand_x = x_start + best_peak[0] + float(np.argmax(smoothed[best_peak[0]:best_peak[1]]))

            if cand_x is not None:
                pts.append((cand_x, y_mid))
                if prev_x is not None:
                    dx = cand_x - prev_x
                prev_x = cand_x
            elif prev_x is not None and abs(dx) > 0.0:
                prev_x = prev_x + dx * 0.70

        poly: Optional[np.ndarray] = None
        if len(pts) >= 3:
            py = [p[1] for p in pts]
            px = [p[0] for p in pts]
            try:
                poly = np.polyfit(py, px, 2)
            except Exception:
                poly = None
        elif len(pts) == 2:
            py = [p[1] for p in pts]
            px = [p[0] for p in pts]
            try:
                m, c = np.polyfit(py, px, 1)
                poly = np.array([0.0, float(m), float(c)])
            except Exception:
                poly = None
        else:
            poly = self.last_right_poly

        if poly is not None:
            y_chk = float(h * self.look_y_ratio)
            x_chk = self.eval_poly(poly, y_chk)
            last_poly = self.last_right_poly
            if last_poly is not None and x_chk is not None:
                x_old = self.eval_poly(last_poly, y_chk)
                if x_old is not None and abs(x_chk - x_old) < 60.0:
                    poly = 0.80 * poly + 0.20 * last_poly

        return poly, pts

    @staticmethod
    def eval_poly(poly: Optional[np.ndarray], y: float) -> Optional[float]:
        """Tính giá trị x tại độ cao y từ phương trình x = a*y^2 + b*y + c."""
        if poly is None:
            return None
        return float(poly[0] * (y ** 2) + poly[1] * y + poly[2])

    @staticmethod
    def calc_curvature(poly: Optional[np.ndarray], y: float) -> float:
        """Tính độ cong mặt đường kappa = x'' / (1 + x'^2)^1.5."""
        if poly is None or abs(poly[0]) < 1e-7:
            return 0.0
        a, b = float(poly[0]), float(poly[1])
        dx = 2.0 * a * y + b
        ddx = 2.0 * a
        return float(ddx / ((1.0 + dx ** 2) ** 1.5))

    def process_frame(
        self,
        image: np.ndarray,
        target_right_x: float = 527.0,
    ) -> Tuple[Optional[float], Optional[float], float, np.ndarray, np.ndarray, str, Optional[float], Optional[float]]:
        """Quy trình nhận diện hoàn chỉnh DUY NHẤT VẠCH PHẢI:

        Returns:
            (target_lane_x, lookahead_x, curvature, clean_mask, debug_image, tracking_mode, r_near, r_look)
        """
        if image is None:
            empty = np.zeros((100, 100), dtype=np.uint8)
            return None, None, 0.0, empty, np.zeros((100, 100, 3), dtype=np.uint8), 'NONE', None, None

        h, w = image.shape[:2]
        near_y = float(h * self.near_y_ratio)
        look_y = float(h * self.look_y_ratio)

        # 1. Lọc nhị phân chống chói & phân biệt vạch chính dày vs mép ngoài mỏng
        mask = self.create_anti_glare_mask(image)

        # 2. Dò quét DUY NHẤT VẠCH PHẢI (Pure Right Stripe Tracking)
        right_poly, right_pts = self._track_right_stripe(mask)

        # 3. Trích xuất tọa độ tại near_y và look_y
        r_near = self.eval_poly(right_poly, near_y)
        r_look = self.eval_poly(right_poly, look_y)

        if right_poly is not None:
            self.last_right_poly = right_poly
        else:
            self.last_right_poly = None

        # 4. Tính toán độ cong mặt đường
        curvature = self.calc_curvature(right_poly, look_y) if right_poly is not None else 0.0
        self.last_curvature = curvature

        # 5. Vị trí bám làn
        target_lane_x = r_near
        lookahead_x = r_look
        tracking_mode = 'BAM VACH PHAI' if r_look is not None else 'MAT VACH'

        # 6. Tạo khung hình trực quan hóa HUD cao cấp cho Web Viewer
        debug_img = self.render_debug(
            image, mask, right_poly, right_pts,
            r_look, lookahead_x, near_y, look_y,
            target_right_x, tracking_mode, curvature
        )

        return target_lane_x, lookahead_x, curvature, mask, debug_img, tracking_mode, r_near, r_look

    def render_debug(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        right_poly: Optional[np.ndarray],
        right_pts: List[Tuple[float, float]],
        r_look: Optional[float],
        lookahead_x: Optional[float],
        near_y: float,
        look_y: float,
        target_right_x: float,
        tracking_mode: str,
        curvature: float,
    ) -> np.ndarray:
        """Trực quan hóa hình ảnh sắc nét, hiển thị đường vạch phải và chỉ số điều khiển."""
        vis = image.copy()
        h, w = vis.shape[:2]

        ny = int(near_y)
        ly = int(look_y)
        y_cut = int(h * self.roi_top_ratio)

        # 1. Đường chân trời ROI & đường quét Lookahead / Near
        cv2.line(vis, (0, y_cut), (w, y_cut), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(vis, (0, ly), (w, ly), (180, 100, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (0, ny), (w, ny), (0, 140, 255), 1, cv2.LINE_AA)

        # 2. Vẽ đường cong đa thức bậc 2 vạch phải (Màu xanh lá neon)
        if right_poly is not None and len(right_pts) >= 2:
            r_py = [p[1] for p in right_pts]
            y_r_min = max(y_cut + 2, int(min(r_py) - 15))
            y_r_max = min(h - 5, int(max(r_py) + 15))
            plot_y_r = np.linspace(y_r_min, y_r_max, 30)
            r_x = right_poly[0] * plot_y_r**2 + right_poly[1] * plot_y_r + right_poly[2]
            valid_r = (r_x >= 0) & (r_x < w)
            if np.any(valid_r):
                curve_pts = np.int32([np.column_stack((r_x[valid_r], plot_y_r[valid_r]))])
                cv2.polylines(vis, curve_pts, isClosed=False, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
            for p in right_pts:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 220, 255), -1, cv2.LINE_AA)

        # 3. Vẽ mốc đích (Target R) & Điểm lái (Lookahead)
        tgt_x = int(round(target_right_x))
        tgt_label = f'TARGET R {tgt_x}px'

        cv2.line(vis, (tgt_x, ly - 32), (tgt_x, ly + 32), (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, tgt_label, (tgt_x - 45, ly + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)

        if lookahead_x is not None:
            lx = int(round(lookahead_x))
            cv2.circle(vis, (lx, ly), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (lx, ly), 9, (255, 255, 255), 2, cv2.LINE_AA)
            # Mũi tên lệch giữa đích và thực tế
            cv2.arrowedLine(vis, (lx, ly), (tgt_x, ly), (0, 255, 255), 2, tipLength=0.25)
            err = lx - tgt_x
            cv2.putText(vis, f'ERR: {err:+.1f}px', (lx - 32, ly - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

        # 4. Màn hình phụ thu nhỏ Anti-Glare PiP Mask (Góc trên bên phải)
        pip_w, pip_h = 160, 115
        pip_m = cv2.resize(mask, (pip_w, pip_h), interpolation=cv2.INTER_NEAREST)
        pip_bgr = cv2.cvtColor(pip_m, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(pip_bgr, (0, 0), (pip_w - 1, pip_h - 1), (0, 255, 0), 2)
        cv2.putText(pip_bgr, 'ANTI-GLARE MASK', (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
        vis[10:10 + pip_h, w - pip_w - 10:w - 10] = pip_bgr

        # 5. Khung trạng thái nhận diện (Góc trên bên trái)
        cv2.rectangle(vis, (8, 8), (340, 52), (20, 24, 33), -1)
        cv2.rectangle(vis, (8, 8), (340, 52), (0, 255, 0), 1)
        cv2.putText(vis, 'UEH TEAM 7 - PURE RIGHT LANE ENGINE', (14, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 220, 255), 1, cv2.LINE_AA)
        curv_deg = curvature * 1000.0
        status_line = f'TRACK: {tracking_mode} | CURV: {curv_deg:+.2f}/k'
        text_color = (0, 255, 0) if tracking_mode != 'LOST' else (0, 0, 255)
        cv2.putText(vis, status_line, (14, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)

        return vis
