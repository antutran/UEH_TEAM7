#!/usr/bin/env python3
"""
Automated Test & Verification Suite (UEH Team 7 - CRC 2026).
Tự động trích xuất và kiểm thử 6 kịch bản trọng yếu trong video:
  1. Vạch xuất phát (Start Line) & Đèn Xanh (Green Light)
  2. Đoạn thẳng chói bóng đèn trần LED (Anti-Glare Stress Test)
  3. Khúc cua gắt sang phải (Sharp Right Curve)
  4. Khu vực dốc cầu & chướng ngại vật (Bridge Ramp & Obstacles)
  5. Ngã tư có Đèn Đỏ & Vạch dừng / Người đi bộ (Red Light & Crosswalk)
  6. Cận cảnh Đèn Đỏ (Red Light Close Range)

Tất cả các ảnh kiểm thử được lưu vào: video_vision_lab/snapshots/
"""

import os
import sys
import time
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from video_vision_lab import config
from video_vision_lab.processor import VisionPipeline


def run_tests():
    print("=" * 70)
    print("🧪 KIỂM THỬ THỊ GIÁC MÁY TÍNH TRÊN 6 KỊCH BẢN THỰC TẾ (VIDEO MOV)")
    print("=" * 70)

    if not os.path.exists(config.DEFAULT_VIDEO_PATH):
        print(f"[LỖI] Không tìm thấy video: {config.DEFAULT_VIDEO_PATH}")
        sys.exit(1)

    os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(config.DEFAULT_VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0

    pipeline = VisionPipeline()
    results_summary = []

    for scene_name, frame_idx in config.KEY_SCENES.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[CẢNH BÁO] Không đọc được frame {frame_idx}")
            continue

        time_sec = frame_idx / fps
        t0 = time.time()
        res = pipeline.process(
            raw_frame=frame,
            frame_idx=frame_idx,
            total_frames=total_frames,
            time_sec=time_sec,
        )
        dt = (time.time() - t0) * 1000.0  # ms

        # Lưu ảnh snapshot kết quả
        out_name = f"test_{scene_name}_frame{frame_idx:04d}.jpg"
        out_path = os.path.join(config.SNAPSHOTS_DIR, out_name)
        cv2.imwrite(out_path, res.dashboard_frame)

        results_summary.append({
            "scene": scene_name,
            "frame": frame_idx,
            "time": f"{time_sec:.1f}s",
            "lane_status": res.lane_status,
            "steering_err": f"{res.steering_error:+.1f}px",
            "heading": f"{res.heading_angle:+.1f}°",
            "tl_state": res.traffic_light_state,
            "stop_line": res.stop_line_detected,
            "crosswalk": res.crosswalk_detected,
            "latency_ms": f"{dt:.1f}ms",
            "snapshot": out_name,
        })

        print(f"\n✅ Đã kiểm thử [{scene_name}] @ Frame {frame_idx} ({time_sec:.1f}s):")
        print(f"   - Bám làn: {res.lane_status} | Sai số lái: {res.steering_error:+.1f}px | Góc: {res.heading_angle:+.1f}°")
        print(f"   - Đèn giao thông: {res.traffic_light_state} (Độ tin cậy: {res.traffic_light_confidence:.2f})")
        print(f"   - Vạch dừng / Đi bộ: Stop={res.stop_line_detected}, Crosswalk={res.crosswalk_detected}")
        print(f"   - Thời gian xử lý: {dt:.1f} ms | Đã lưu: snapshots/{out_name}")

    cap.release()

    print("\n" + "=" * 70)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ KIỂM THỬ:")
    print(f"{'Kịch bản':<24} | {'Frame':<6} | {'Bám làn':<12} | {'Đèn':<8} | {'Vạch dừng':<10} | {'Độ trễ':<8}")
    print("-" * 70)
    for r in results_summary:
        stop_str = "CÓ" if (r["stop_line"] or r["crosswalk"]) else "KHÔNG"
        print(f"{r['scene']:<24} | {r['frame']:<6} | {r['lane_status']:<12} | {r['tl_state']:<8} | {stop_str:<10} | {r['latency_ms']:<8}")
    print("=" * 70)
    print(f"📁 Toàn bộ ảnh trực quan kết quả đã được lưu tại:")
    print(f"   {config.SNAPSHOTS_DIR}")


if __name__ == "__main__":
    run_tests()
