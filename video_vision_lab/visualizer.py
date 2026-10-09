#!/usr/bin/env python3
"""
HUD & Telemetry Visualization Engine (UEH Team 7 - CRC 2026).
Bộ giao diện trực quan hoá cao cấp đa khung hình (Multi-Panel Dashboard):
  - Khung chính: Camera AR View với đường cong đa thức vạch bám làn, hành lang di chuyển an toàn,
    hộp nhận diện đèn tín hiệu, biển báo giao thông và vector độ lệch lái.
  - Khung phụ: Mask nhị phân chống chói quang học (Anti-Glare Mask).
  - Khung chẩn đoán: Cận cảnh đèn tín hiệu, bóng đèn phát sáng & biển báo.
  - Khung Telemetry: Đồng hồ đo độ cong cua, góc lái, khoảng cách và lệnh điều khiển xe tự hành.
"""

from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

from . import config


class VisionVisualizer:
    """Bộ tạo giao diện trực quan hóa chuyên nghiệp cho phòng thí nghiệm xử lý ảnh."""

    def __init__(self, main_w: int = config.FRAME_WIDTH, main_h: int = config.FRAME_HEIGHT):
        self.main_w = main_w
        self.main_h = main_h

        # Kích thước khung tổng hợp Dashboard (1040 x 540)
        self.hud_w = 380
        self.total_w = self.main_w + self.hud_w
        self.total_h = self.main_h + 60  # Thêm thanh trạng thái đáy 60px

    def render(
        self,
        raw_frame: np.ndarray,
        lane_res: Dict[str, Any],
        tl_res: Dict[str, Any],
        sign_res: List[Dict[str, Any]],
        fps: float = 0.0,
        frame_idx: int = 0,
        total_frames: int = 0,
        time_sec: float = 0.0,
    ) -> np.ndarray:
        """Dựng khung hình Dashboard tổng hợp hoàn chỉnh."""
        # Đảm bảo raw_frame đúng kích thước 640x480
        if raw_frame.shape[1] != self.main_w or raw_frame.shape[0] != self.main_h:
            raw_frame = cv2.resize(raw_frame, (self.main_w, self.main_h))

        # 1. Vẽ các thành phần tăng cường AR lên ảnh Camera chính
        main_vis = raw_frame.copy()
        self._draw_lane_ar(main_vis, lane_res)
        self._draw_traffic_light_ar(main_vis, tl_res)
        self._draw_signs_ar(main_vis, sign_res)

        # 2. Tạo Canvas tổng Dashboard nền tối sang trọng
        dashboard = np.full((self.total_h, self.total_w, 3), 20, dtype=np.uint8)

        # Đặt khung Camera chính vào bên trái
        dashboard[0:self.main_h, 0:self.main_w] = main_vis

        # 3. Vẽ Bảng Điều Khiển Cạnh Phải (HUD Side Panel)
        self._draw_side_hud(
            dashboard,
            lane_res,
            tl_res,
            sign_res,
            raw_frame,
        )

        # 4. Vẽ Thanh Trạng Thái Đáy (Bottom Telemetry Bar)
        self._draw_bottom_bar(
            dashboard,
            lane_res,
            tl_res,
            fps,
            frame_idx,
            total_frames,
            time_sec,
        )

        # Đường viền phân cách hiện đại
        cv2.line(dashboard, (self.main_w, 0), (self.main_w, self.total_h), (50, 50, 50), 2)
        cv2.line(dashboard, (0, self.main_h), (self.total_w, self.main_h), (50, 50, 50), 2)

        return dashboard

    def _draw_lane_ar(self, vis: np.ndarray, lane: Dict[str, Any]) -> None:
        """Vẽ đường vạch bám làn và hành lang an toàn."""
        h, w = vis.shape[:2]
        y_cut = int(h * config.ROI_TOP_RATIO)
        poly = lane.get("poly")
        pts = lane.get("points", [])
        look_pt = lane.get("lookahead_pt")
        target_x = config.TARGET_RIGHT_X

        # Vạch chân trời ROI
        cv2.line(vis, (0, y_cut), (w, y_cut), (80, 80, 80), 1, cv2.LINE_AA)

        # Đường ngang quét Lookahead
        if look_pt:
            ly = int(look_pt[1])
            cv2.line(vis, (0, ly), (w, ly), (140, 70, 200), 1, cv2.LINE_AA)

        # Vẽ đường cong đa thức vạch phải (Neon Green)
        if poly is not None:
            plot_y = np.linspace(y_cut + 5, h - 5, 40)
            plot_x = poly[0] * plot_y**2 + poly[1] * plot_y + poly[2]
            valid = (plot_x >= 0) & (plot_x < w)
            if np.any(valid):
                curve_pts = np.int32([np.column_stack((plot_x[valid], plot_y[valid]))])
                # Phát quang Glow ngoài
                cv2.polylines(vis, curve_pts, isClosed=False, color=(0, 160, 0), thickness=6, lineType=cv2.LINE_AA)
                # Lõi xanh neon sáng
                cv2.polylines(vis, curve_pts, isClosed=False, color=(50, 255, 50), thickness=3, lineType=cv2.LINE_AA)

                # Vẽ hành lang di chuyển an toàn giữa tim đường và vạch phải (Cyan tint)
                overlay = vis.copy()
                left_x = plot_x - 170  # Ước lượng độ rộng làn thi đấu (~170px)
                left_valid = np.clip(left_x, 0, w - 1)
                corridor_pts = np.vstack([
                    np.column_stack((left_valid[valid], plot_y[valid])),
                    np.column_stack((plot_x[valid], plot_y[valid]))[::-1]
                ])
                cv2.fillPoly(overlay, [np.int32(corridor_pts)], (0, 180, 220))
                cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)

        # Vẽ các điểm dò quét Sliding Windows (Màu vàng rực rỡ)
        for p in pts:
            cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 220, 255), -1, cv2.LINE_AA)

        # Mốc đích Target R & Điểm nhìn xa Lookahead
        if look_pt:
            lx, ly = int(look_pt[0]), int(look_pt[1])
            tx = int(target_x)

            # Target mark (Màu cam)
            cv2.line(vis, (tx, ly - 22), (tx, ly + 22), (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"TGT {tx}px", (tx - 35, ly + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1, cv2.LINE_AA)

            # Lookahead bullseye
            cv2.circle(vis, (lx, ly), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (lx, ly), 10, (255, 255, 255), 2, cv2.LINE_AA)

            # Mũi tên sai lệch (Error vector)
            err = lane.get("steering_error", 0.0)
            arrow_color = (0, 255, 255) if abs(err) < 30 else (0, 69, 255)
            cv2.arrowedLine(vis, (lx, ly), (tx, ly), arrow_color, 2, tipLength=0.25)
            cv2.putText(vis, f"ERR: {err:+.1f}px", (lx - 35, ly - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, arrow_color, 1, cv2.LINE_AA)

        # Cảnh báo Vạch dừng xe (Stop Line) hoặc Người đi bộ (Crosswalk)
        if lane.get("crosswalk_detected"):
            cv2.rectangle(vis, (80, y_cut + 40), (w - 80, y_cut + 75), (0, 0, 200), -1)
            cv2.putText(vis, "PEDESTRIAN CROSSWALK DETECTED", (110, y_cut + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        elif lane.get("stop_line_detected"):
            cv2.rectangle(vis, (120, y_cut + 40), (w - 120, y_cut + 75), (0, 140, 255), -1)
            cv2.putText(vis, "STOP LINE DETECTED", (170, y_cut + 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_traffic_light_ar(self, vis: np.ndarray, tl: Dict[str, Any]) -> None:
        """Vẽ khung nhận diện đèn tín hiệu giao thông."""
        state = tl.get("state", "NONE")
        bbox = tl.get("bbox")
        lamp_bbox = tl.get("lamp_bbox")
        dist = tl.get("distance_cm")

        if state == "NONE" or bbox is None:
            return

        x, y, w, h = bbox
        color_map = {
            "GREEN": ((0, 255, 0), "[GREEN - GO]"),
            "RED": ((0, 0, 255), "[RED - STOP]"),
            "YELLOW": ((0, 230, 255), "[YELLOW - CAUTION]"),
        }
        b_color, label = color_map.get(state, ((200, 200, 200), state))

        # Khung viền hộp đèn phát sáng
        cv2.rectangle(vis, (x, y), (x + w, y + h), b_color, 2, cv2.LINE_AA)

        # Highlight bóng đèn đang sáng
        if lamp_bbox:
            lx, ly, lw, lh = lamp_bbox
            cv2.circle(vis, (lx + lw // 2, ly + lh // 2), max(lw, lh) // 2 + 3, b_color, 2, cv2.LINE_AA)

        # Nhãn trạng thái trên đỉnh hộp đèn
        dist_str = f" ({dist:.0f}cm)" if dist else ""
        text = f"{label}{dist_str}"
        cv2.rectangle(vis, (x - 20, y - 24), (x + w + 70, y), (10, 10, 10), -1)
        cv2.putText(vis, text, (x - 16, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_color, 1, cv2.LINE_AA)

    def _draw_signs_ar(self, vis: np.ndarray, signs: List[Dict[str, Any]]) -> None:
        """Vẽ khung nhận diện biển báo giao thông."""
        for s in signs:
            x, y, w, h = s["bbox"]
            name = s["name"]
            score = s["score"]

            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 180, 0), 2, cv2.LINE_AA)
            cv2.rectangle(vis, (x, y - 20), (x + max(70, w + 30), y), (15, 15, 15), -1)
            cv2.putText(
                vis,
                f"{name} ({score:.2f})",
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 220, 50),
                1,
                cv2.LINE_AA,
            )

    def _draw_side_hud(
        self,
        dashboard: np.ndarray,
        lane: Dict[str, Any],
        tl: Dict[str, Any],
        signs: List[Dict[str, Any]],
        raw_frame: np.ndarray,
    ) -> None:
        """Vẽ cột hiển thị công nghệ cao cạnh phải."""
        hx = self.main_w + 10

        # --- PHẦN 1: ANTI-GLARE MASK VIEW (Trên cùng) ---
        cv2.putText(dashboard, "OPTICAL ANTI-GLARE MASK", (hx + 5, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

        mask = lane.get("mask")
        if mask is not None:
            # Resize mask về 360 x 140
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # Thêm viền màu cho trực quan
            mask_resized = cv2.resize(mask_rgb, (self.hud_w - 20, 140))
            dashboard[35:175, hx:hx + self.hud_w - 20] = mask_resized

        # --- PHẦN 2: TRAFFIC LIGHT & SIGN INSPECTOR ---
        y_sec2 = 195
        cv2.putText(dashboard, "TRAFFIC PERCEPTION STATUS", (hx + 5, y_sec2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

        # Mini status circles cho 3 màu đèn
        tl_state = tl.get("state", "NONE")
        c_green = (0, 255, 0) if tl_state == "GREEN" else (30, 60, 30)
        c_yellow = (0, 230, 255) if tl_state == "YELLOW" else (40, 60, 60)
        c_red = (0, 0, 255) if tl_state == "RED" else (60, 30, 30)

        cy = y_sec2 + 30
        cv2.circle(dashboard, (hx + 30, cy), 14, c_green, -1, cv2.LINE_AA)
        cv2.circle(dashboard, (hx + 80, cy), 14, c_yellow, -1, cv2.LINE_AA)
        cv2.circle(dashboard, (hx + 130, cy), 14, c_red, -1, cv2.LINE_AA)

        status_txt = f"LIGHT: {tl_state}"
        txt_color = (0, 255, 0) if tl_state == "GREEN" else ((0, 0, 255) if tl_state == "RED" else (160, 160, 160))
        cv2.putText(dashboard, status_txt, (hx + 170, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 2, cv2.LINE_AA)

        # Biển báo nhận diện
        sign_y = cy + 38
        if signs:
            s_top = signs[0]
            cv2.putText(dashboard, f"SIGN: {s_top['name'].upper()} ({s_top['score']:.2f})",
                        (hx + 15, sign_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(dashboard, "SIGN: NONE DETECTED",
                        (hx + 15, sign_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1, cv2.LINE_AA)

        # --- PHẦN 3: AUTOPILOT TELEMETRY & DECISION ENGINE ---
        y_sec3 = 300
        cv2.putText(dashboard, "AUTOPILOT TELEMETRY", (hx + 5, y_sec3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

        # Vẽ bảng thông số
        err = lane.get("steering_error", 0.0)
        heading = lane.get("heading_angle", 0.0)
        curv = lane.get("curvature", 0.0)
        lane_status = lane.get("status", "LOST")

        curv_desc = "STRAIGHT"
        if curv > 0.002:
            curv_desc = "SHARP RIGHT" if curv > 0.005 else "GENTLE RIGHT"
        elif curv < -0.002:
            curv_desc = "SHARP LEFT" if curv < -0.005 else "GENTLE LEFT"

        rows = [
            ("LANE TRACKER", f"{lane_status}", (50, 255, 50) if "TRACK" in lane_status else (0, 165, 255)),
            ("STEER ERROR", f"{err:+.1f} px", (0, 255, 255) if abs(err) < 30 else (0, 69, 255)),
            ("HEADING ANGLE", f"{heading:+.1f} deg", (220, 220, 220)),
            ("ROAD CURVATURE", f"{curv_desc}", (220, 220, 220)),
        ]

        ty = y_sec3 + 26
        for label, val, val_col in rows:
            cv2.putText(dashboard, label, (hx + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(dashboard, val, (hx + 175, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, val_col, 1, cv2.LINE_AA)
            ty += 24

        # Quyết định điều khiển xe tự hành (Mission Action)
        y_act = ty + 12
        action_text = "CRUISE - FOLLOW LANE"
        action_bg = (0, 130, 0)

        if tl_state == "RED":
            action_text = "STOP - RED LIGHT"
            action_bg = (0, 0, 180)
        elif lane.get("crosswalk_detected"):
            action_text = "SLOW - CROSSWALK"
            action_bg = (0, 120, 220)
        elif lane.get("stop_line_detected"):
            action_text = "BRAKE - STOP LINE"
            action_bg = (0, 100, 200)
        elif "SHARP" in curv_desc:
            action_text = f"CORNER - {curv_desc}"
            action_bg = (180, 100, 0)
        elif lane_status == "LOST":
            action_text = "HOLD - SEARCHING LANE"
            action_bg = (140, 140, 0)

        cv2.rectangle(dashboard, (hx + 10, y_act), (hx + self.hud_w - 20, y_act + 40), action_bg, -1)
        cv2.rectangle(dashboard, (hx + 10, y_act), (hx + self.hud_w - 20, y_act + 40), (255, 255, 255), 1)
        cv2.putText(dashboard, action_text, (hx + 22, y_act + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_bottom_bar(
        self,
        dashboard: np.ndarray,
        lane: Dict[str, Any],
        tl: Dict[str, Any],
        fps: float,
        frame_idx: int,
        total_frames: int,
        time_sec: float,
    ) -> None:
        """Vẽ thanh trạng thái đáy (Telemetry Bar)."""
        by = self.main_h + 38
        mins = int(time_sec // 60)
        secs = time_sec % 60

        time_str = f"TIME: {mins:02d}:{secs:04.1f}s"
        frame_str = f"FRAME: {frame_idx}/{total_frames}" if total_frames > 0 else f"FRAME: {frame_idx}"
        fps_str = f"FPS: {fps:.1f}"

        cv2.putText(dashboard, "UEH TEAM 7 | CRC 2026 VISION LAB", (20, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 230, 255), 2, cv2.LINE_AA)

        cv2.putText(dashboard, time_str, (330, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(dashboard, frame_str, (470, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(dashboard, fps_str, (620, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 255, 50), 2, cv2.LINE_AA)
