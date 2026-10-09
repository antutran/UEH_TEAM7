#!/usr/bin/env python3
"""
Traffic Sign Perception Engine (UEH Team 7 - CRC 2026).
Bộ nhận diện biển báo giao thông sa bàn thi đấu:
  - Khớp mẫu (Multi-Scale Template Matching) với bộ biển báo chuẩn CRC:
    'hw_entry', 'hw_exit', 'turn_right', 'turn_left', 'forward', 'stop', 'ramp', 'tunnel', 'bus'
  - Bộ lọc chống nhiễu áo người tham gia thi đấu ở hậu cảnh.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import glob
import cv2
import numpy as np

from . import config


class TrafficSignDetector:
    """Bộ nhận diện biển báo giao thông trên sa bàn."""

    def __init__(self, templates_dir: str = config.TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self.templates: Dict[str, np.ndarray] = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Tải các file mẫu biển báo PNG trong thư mục images/."""
        if not os.path.exists(self.templates_dir):
            return

        for path in glob.glob(os.path.join(self.templates_dir, "*.png")):
            name = os.path.splitext(os.path.basename(path))[0]
            tpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if tpl is None:
                continue

            if len(tpl.shape) == 3 and tpl.shape[2] == 4:
                # Xử lý kênh alpha suốt
                alpha = tpl[:, :, 3] / 255.0
                rgb = tpl[:, :, :3]
                tpl_bgr = (rgb * alpha[:, :, None] + (1 - alpha[:, :, None]) * 255).astype(np.uint8)
            else:
                tpl_bgr = tpl

            # Lưu ảnh xám chuẩn hoá kích thước 64x64
            gray = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
            self.templates[name] = cv2.resize(gray, (64, 64))

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Phát hiện và phân loại các biển báo giao thông xuất hiện trong ảnh."""
        if not self.templates:
            return []

        h, w = image.shape[:2]
        # Vùng tìm kiếm biển báo: hai bên mép đường và tầm cao vừa phải
        y_min = config.SIGN_ROI_Y_MIN
        y_max = min(h - 50, config.SIGN_ROI_Y_MAX)
        roi = image[y_min:y_max, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1. Mask màu biển báo xanh dương (Phần lớn biển chỉ dẫn: rẽ phải, đi thẳng, cao tốc, bus)
        blue_mask = cv2.inRange(
            hsv,
            np.array([config.SIGN_BLUE_H_MIN, config.SIGN_BLUE_S_MIN, config.SIGN_BLUE_V_MIN], dtype=np.uint8),
            np.array([config.SIGN_BLUE_H_MAX, 255, 255], dtype=np.uint8),
        )

        # 2. Mask màu đỏ (Biển STOP)
        red_m1 = cv2.inRange(hsv, np.array([0, 80, 70], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
        red_m2 = cv2.inRange(hsv, np.array([168, 80, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        red_mask = cv2.bitwise_or(red_m1, red_m2)

        # 3. Mask màu vàng (Biển cảnh báo dốc cầu ramp / tunnel)
        yellow_mask = cv2.inRange(
            hsv,
            np.array([18, 90, 90], dtype=np.uint8),
            np.array([36, 255, 255], dtype=np.uint8),
        )

        combined_masks = [
            ("blue", blue_mask),
            ("red", red_mask),
            ("yellow", yellow_mask),
        ]

        detected_signs = []

        for color_name, mask in combined_masks:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                # Biển báo trên sa bàn có diện tích vừa phải (80 - 2500 px)
                if 80 <= area <= 2500:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    aspect = bw / float(bh)
                    # Biển báo thường là hình vuông/tròn (tỉ lệ 0.6 - 1.6)
                    if 0.55 <= aspect <= 1.7 and bw <= 65 and bh <= 65:
                        abs_y = y_min + by
                        abs_x = bx

                        # Lấy vùng ảnh biển báo
                        crop = image[abs_y:abs_y + bh, abs_x:abs_x + bw]
                        if crop.shape[0] < 10 or crop.shape[1] < 10:
                            continue

                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        crop_resized = cv2.resize(crop_gray, (64, 64))

                        # Khớp mẫu với các templates tương ứng
                        best_match = None
                        best_score = -1.0

                        for t_name, t_gray in self.templates.items():
                            # Lọc màu tương ứng
                            if color_name == "red" and t_name != "stop":
                                continue
                            if color_name == "yellow" and t_name not in ["ramp", "tunnel"]:
                                continue
                            if color_name == "blue" and t_name in ["stop", "ramp", "tunnel"]:
                                continue

                            res = cv2.matchTemplate(crop_resized, t_gray, cv2.TM_CCOEFF_NORMED)
                            score = float(res[0, 0])
                            if score > best_score:
                                best_score = score
                                best_match = t_name

                        if best_match and best_score >= config.SIGN_MATCH_THRESHOLD:
                            detected_signs.append({
                                "name": best_match,
                                "score": best_score,
                                "color": color_name,
                                "bbox": (abs_x, abs_y, bw, bh),
                            })

        # Loại bỏ các box trùng lặp (Non-Maximum Suppression đơn giản)
        final_signs = []
        for s in sorted(detected_signs, key=lambda x: x["score"], reverse=True):
            overlap = False
            for kept in final_signs:
                kbx, kby, kbw, kbh = kept["bbox"]
                sbx, sby, sbw, sbh = s["bbox"]
                if abs(kbx - sbx) < 25 and abs(kby - sby) < 25:
                    overlap = True
                    break
            if not overlap:
                final_signs.append(s)

        return final_signs
