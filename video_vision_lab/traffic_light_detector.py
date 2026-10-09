#!/usr/bin/env python3
"""
Traffic Light Perception Engine (UEH Team 7 - CRC 2026).
Bộ nhận diện đèn tín hiệu giao thông sa bàn CRC 2026.
Cột đèn 3 bóng dạng hộp đứng:
  - XANH (Green)  : Bóng trên cùng
  - VÀNG (Yellow) : Bóng ở giữa
  - ĐỎ   (Red)    : Bóng dưới cùng
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import cv2
import numpy as np

from . import config


class TrafficLightDetector:
    """Bộ nhận diện đèn giao thông chính xác cao cho sa bàn CRC."""

    def __init__(self, history_len: int = 4):
        self.history_len = history_len
        self.state_history = deque(maxlen=history_len)
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Phát hiện đèn giao thông và xác định trạng thái XANH / ĐỎ / VÀNG / NONE."""
        h, w = image.shape[:2]
        
        # Cắt ROI tìm kiếm đèn (phía trên mặt đường)
        y1, y2 = config.TL_ROI_Y_MIN, min(h, config.TL_ROI_Y_MAX)
        x1, x2 = config.TL_ROI_X_MIN, min(w, config.TL_ROI_X_MAX)
        roi = image[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]

        b = roi[:, :, 0].astype(float)
        g = roi[:, :, 1].astype(float)
        r = roi[:, :, 2].astype(float)

        # 1. Mask màu đèn phát sáng (LED Prominence)
        # Đèn Xanh: G sáng vượt trội
        green_mask = (
            (g > config.GREEN_MIN_INTENSITY) &
            (g > (r + b) / 2.0 + config.GREEN_PROMINENCE)
        ).astype(np.uint8)

        # Đèn Đỏ: R sáng vượt trội
        red_mask = (
            (r > config.RED_MIN_INTENSITY) &
            (r > g + config.RED_PROMINENCE) &
            (r > b + config.RED_PROMINENCE)
        ).astype(np.uint8)

        # Đèn Vàng: Cả R và G đều cao, B thấp
        yellow_mask = (
            (r > config.YELLOW_MIN_INTENSITY) &
            (g > 120) &
            (b < 110) &
            (np.abs(r - g) < 55)
        ).astype(np.uint8)

        candidates = []

        # 2. Tìm ứng viên Đèn Xanh (Bóng trên cùng)
        cnts_g, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs_g = []
        for c in cnts_g:
            area = cv2.contourArea(c)
            if config.MIN_LAMP_AREA <= area <= config.MAX_LAMP_AREA:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh)
                if 0.45 <= aspect <= 2.2:
                    blobs_g.append((by, bx, bw, bh, area, c))
        # Sắp xếp theo y tăng dần (lấy bóng thực trên cao, loại bỏ bóng phản chiếu dưới sàn)
        blobs_g.sort(key=lambda item: item[0])
        for by, bx, bw, bh, area, c in blobs_g:
            # Kiểm tra thân hộp đèn màu đen xung quanh & bên dưới bóng xanh
            hx1 = max(0, bx - 12)
            hx2 = min(rw, bx + bw + 12)
            hy1 = max(0, by - 12)
            hy2 = min(rh, by + bh + int(bh * 2.8))  # Thân kéo dài xuống dưới
            housing = roi[hy1:hy2, hx1:hx2]
            dark_ratio = np.mean(cv2.cvtColor(housing, cv2.COLOR_BGR2GRAY) < 85)
            if dark_ratio >= 0.18:
                candidates.append({
                    "state": "GREEN",
                    "confidence": float(min(1.0, 0.5 + dark_ratio)),
                    "lamp_bbox": (x1 + bx, y1 + by, bw, bh),
                    "housing_bbox": (x1 + hx1, y1 + hy1, hx2 - hx1, hy2 - hy1),
                    "area": area,
                    "y_pos": by,
                })
                break

        # 3. Tìm ứng viên Đèn Đỏ (Bóng dưới cùng)
        cnts_r, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs_r = []
        for c in cnts_r:
            area = cv2.contourArea(c)
            if config.MIN_LAMP_AREA <= area <= config.MAX_LAMP_AREA:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh)
                if 0.45 <= aspect <= 2.2:
                    blobs_r.append((by, bx, bw, bh, area, c))
        blobs_r.sort(key=lambda item: item[0])
        for by, bx, bw, bh, area, c in blobs_r:
            # Kiểm tra thân hộp đèn màu đen phía trên bóng đỏ (vì bóng đỏ ở dưới cùng!)
            hx1 = max(0, bx - 14)
            hx2 = min(rw, bx + bw + 14)
            hy1 = max(0, by - int(bh * 2.8))  # Thân kéo dài lên trên
            hy2 = min(rh, by + bh + 15)
            housing = roi[hy1:hy2, hx1:hx2]
            dark_ratio = np.mean(cv2.cvtColor(housing, cv2.COLOR_BGR2GRAY) < 85)
            if dark_ratio >= 0.16:
                candidates.append({
                    "state": "RED",
                    "confidence": float(min(1.0, 0.5 + dark_ratio)),
                    "lamp_bbox": (x1 + bx, y1 + by, bw, bh),
                    "housing_bbox": (x1 + hx1, y1 + hy1, hx2 - hx1, hy2 - hy1),
                    "area": area,
                    "y_pos": by,
                })
                break

        # 4. Tìm ứng viên Đèn Vàng (Bóng ở giữa)
        cnts_y, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs_y = []
        for c in cnts_y:
            area = cv2.contourArea(c)
            if config.MIN_LAMP_AREA <= area <= config.MAX_LAMP_AREA:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh)
                if 0.45 <= aspect <= 2.2:
                    blobs_y.append((by, bx, bw, bh, area, c))
        blobs_y.sort(key=lambda item: item[0])
        for by, bx, bw, bh, area, c in blobs_y:
            hx1 = max(0, bx - 12)
            hx2 = min(rw, bx + bw + 12)
            hy1 = max(0, by - int(bh * 1.5))
            hy2 = min(rh, by + bh + int(bh * 1.5))
            housing = roi[hy1:hy2, hx1:hx2]
            dark_ratio = np.mean(cv2.cvtColor(housing, cv2.COLOR_BGR2GRAY) < 85)
            if dark_ratio >= 0.18:
                candidates.append({
                    "state": "YELLOW",
                    "confidence": float(min(1.0, 0.5 + dark_ratio)),
                    "lamp_bbox": (x1 + bx, y1 + by, bw, bh),
                    "housing_bbox": (x1 + hx1, y1 + hy1, hx2 - hx1, hy2 - hy1),
                    "area": area,
                    "y_pos": by,
                })
                break

        # Chọn ứng viên có độ tin cậy và kích thước tốt nhất
        current_state = "NONE"
        best_cand = None
        if candidates:
            best_cand = max(candidates, key=lambda c: c["confidence"] * np.sqrt(c["area"]))
            current_state = best_cand["state"]

        # Lọc mượt trạng thái qua lịch sử frames (Temporal Filter)
        self.state_history.append(current_state)
        # Bầu chọn đa số (Majority voting)
        states = list(self.state_history)
        final_state = max(set(states), key=states.count)

        # Ước lượng khoảng cách đến đèn (cm)
        dist_cm = None
        bbox = None
        lamp_bbox = None
        conf = 0.0

        if best_cand and final_state != "NONE":
            bbox = best_cand["housing_bbox"]
            lamp_bbox = best_cand["lamp_bbox"]
            conf = best_cand["confidence"]
            self.last_bbox = bbox
            # Thấu kính tiêu cự: chiều cao bóng bh tỉ lệ nghịch với khoảng cách
            lamp_h = lamp_bbox[3]
            if lamp_h > 0:
                dist_cm = float(np.clip(1600.0 / lamp_h, 30.0, 250.0))
        elif self.last_bbox is not None and final_state != "NONE":
            bbox = self.last_bbox
            conf = 0.5

        return {
            "state": final_state,
            "confidence": conf,
            "bbox": bbox,
            "lamp_bbox": lamp_bbox,
            "distance_cm": dist_cm,
        }
