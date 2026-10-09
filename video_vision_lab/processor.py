#!/usr/bin/env python3
"""
Unified Perception Pipeline (UEH Team 7 - CRC 2026).
Tích hợp toàn bộ các bộ nhận diện thành 1 đường ống xử lý (Pipeline) duy nhất:
  - Làn đường chống chói & bám vạch
  - Đèn tín hiệu giao thông (Xanh, Vàng, Đỏ)
  - Biển báo giao thông
  - Dựng khung hình Dashboard AR Telemetry
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time
import cv2
import numpy as np

from . import config
from .lane_detector import AntiGlareLaneDetector
from .traffic_light_detector import TrafficLightDetector
from .sign_detector import TrafficSignDetector
from .visualizer import VisionVisualizer


@dataclass
class VisionResult:
    """Kết quả đầu ra của 1 chu kỳ xử lý ảnh."""
    frame_idx: int
    time_sec: float
    fps: float

    # Lane perception
    steering_error: float
    heading_angle: float
    curvature: float
    lane_status: str
    stop_line_detected: bool
    crosswalk_detected: bool
    lane_poly: Optional[np.ndarray]

    # Traffic light
    traffic_light_state: str
    traffic_light_confidence: float
    traffic_light_dist_cm: Optional[float]

    # Traffic signs
    detected_signs: List[Dict[str, Any]]

    # Processed visuals
    dashboard_frame: np.ndarray
    anti_glare_mask: np.ndarray


class VisionPipeline:
    """Đường ống thị giác máy tính toàn diện cho video sa bàn."""

    def __init__(
        self,
        target_right_x: float = config.TARGET_RIGHT_X,
        enable_signs: bool = True,
        enable_traffic_light: bool = True,
    ):
        self.lane_detector = AntiGlareLaneDetector(target_right_x=target_right_x)
        self.tl_detector = TrafficLightDetector() if enable_traffic_light else None
        self.sign_detector = TrafficSignDetector() if enable_signs else None
        self.visualizer = VisionVisualizer()

        # Thống kê hiệu năng
        self._last_time = time.time()
        self._frame_count = 0
        self._fps = 0.0

    def process(
        self,
        raw_frame: np.ndarray,
        frame_idx: int = 0,
        total_frames: int = 0,
        time_sec: float = 0.0,
    ) -> VisionResult:
        """Xử lý một khung hình video và trả về đầy đủ kết quả nhận diện."""
        t0 = time.time()

        # 1. Chuẩn hóa kích thước khung hình về 640x480
        if raw_frame.shape[1] != config.FRAME_WIDTH or raw_frame.shape[0] != config.FRAME_HEIGHT:
            resized_frame = cv2.resize(raw_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        else:
            resized_frame = raw_frame

        # 2. Xử lý làn đường (Lane Tracking & Anti-Glare)
        lane_res = self.lane_detector.process(resized_frame)

        # 3. Nhận diện đèn giao thông
        tl_res = self.tl_detector.detect(resized_frame) if self.tl_detector else {"state": "NONE", "confidence": 0.0}

        # 4. Nhận diện biển báo giao thông
        signs_res = self.sign_detector.detect(resized_frame) if self.sign_detector else []

        # 5. Đo đạc FPS
        dt = t0 - self._last_time
        self._last_time = t0
        self._fps = 1.0 / dt if dt > 0.001 else self._fps

        # 6. Dựng Dashboard AR Telemetry
        dashboard = self.visualizer.render(
            raw_frame=resized_frame,
            lane_res=lane_res,
            tl_res=tl_res,
            sign_res=signs_res,
            fps=self._fps,
            frame_idx=frame_idx,
            total_frames=total_frames,
            time_sec=time_sec,
        )

        return VisionResult(
            frame_idx=frame_idx,
            time_sec=time_sec,
            fps=self._fps,
            steering_error=lane_res.get("steering_error", 0.0),
            heading_angle=lane_res.get("heading_angle", 0.0),
            curvature=lane_res.get("curvature", 0.0),
            lane_status=lane_res.get("status", "LOST"),
            stop_line_detected=lane_res.get("stop_line_detected", False),
            crosswalk_detected=lane_res.get("crosswalk_detected", False),
            lane_poly=lane_res.get("poly"),
            traffic_light_state=tl_res.get("state", "NONE"),
            traffic_light_confidence=tl_res.get("confidence", 0.0),
            traffic_light_dist_cm=tl_res.get("distance_cm"),
            detected_signs=signs_res,
            dashboard_frame=dashboard,
            anti_glare_mask=lane_res.get("mask"),
        )
