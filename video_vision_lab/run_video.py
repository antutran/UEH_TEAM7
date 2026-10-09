#!/usr/bin/env python3
"""
Video Vision Runner (UEH Team 7 - CRC 2026).
Script thực thi chính để xem, kiểm thử và phân tích video sa bàn.

Cách dùng:
  # 1. Chạy tương tác với giao diện GUI:
  python3 run_video.py

  # 2. Bắt đầu từ cảnh đèn đỏ (frame 1440):
  python3 run_video.py --start-frame 1440

  # 3. Nhảy tới cảnh cua gắt:
  python3 run_video.py --scene 3

  # 4. Xuất video kết quả ra file MP4 (không cần mở cửa sổ):
  python3 run_video.py --save-video output_processed.mp4 --headless --max-frames 500

Phím điều khiển khi đang mở cửa sổ GUI:
  - [SPACE]       : Tạm dừng / Tiếp tục chạy (Pause / Play)
  - [D] / [Right] : Tiến 1 frame (khi tạm dừng) hoặc nhảy +40 frames
  - [A] / [Left]  : Lùi 1 frame (khi tạm dừng) hoặc nhảy -40 frames
  - [S]           : Chụp và lưu ảnh Dashboard hiện tại vào snapshots/
  - [R]           : Quay về đầu video (frame 0)
  - [1] -> [6]    : Nhảy nhanh tới 6 kịch bản trọng điểm:
      [1]: Vạch xuất phát & Đèn xanh
      [2]: Đoạn thẳng chói đèn trần
      [3]: Khúc cua gắt sang phải
      [4]: Chân dốc cầu & vòng xuyến
      [5]: Đèn đỏ ngã tư & vạch dừng
      [6]: Cận cảnh đèn đỏ
  - [Q] / [ESC]   : Thoát chương trình
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

# Cho phép chạy trực tiếp từ thư mục video_vision_lab hoặc từ project root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from video_vision_lab import config
from video_vision_lab.processor import VisionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Video Vision Runner for CRC 2026")
    parser.add_argument(
        "--video",
        type=str,
        default=config.DEFAULT_VIDEO_PATH,
        help="Đường dẫn file video MOV/MP4",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Đường dẫn file MP4 xuất kết quả (nếu muốn lưu)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Khung hình bắt đầu (0-based)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Số khung hình tối đa cần xử lý",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Nhảy trực tiếp đến cảnh (1..6 hoặc tên cảnh)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Tốc độ khung hình khi xem GUI (FPS)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Chạy ngầm không mở cửa sổ cv2.imshow (dành cho xuất video)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"[LỖI] Không tìm thấy file video tại: {video_path}")
        sys.exit(1)

    os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 16.0

    print("=" * 60)
    print("🚗 PHÒNG THÍ NGHIỆM XỬ LÝ ẢNH VIDEO - UEH TEAM 7 (CRC 2026)")
    print(f"🎬 Video: {os.path.basename(video_path)}")
    print(f"📊 Tổng frames: {total_frames} | FPS gốc: {video_fps:.2f} | Thời lượng: {total_frames / video_fps:.1f}s")
    print("=" * 60)

    # Xử lý chọn cảnh bắt đầu
    start_frame = args.start_frame
    if args.scene:
        scene_key = args.scene.strip()
        matched_frame = None
        for k, v in config.KEY_SCENES.items():
            if scene_key == k or scene_key == k.split("_")[0] or scene_key in k:
                matched_frame = v
                print(f"[SCENE] Đã chọn cảnh {k} -> Nhảy tới frame {matched_frame}")
                break
        if matched_frame is not None:
            start_frame = matched_frame

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pipeline = VisionPipeline()

    # Chuẩn bị VideoWriter nếu có cờ --save-video
    writer = None
    if args.save_video:
        out_path = args.save_video
        if not os.path.isabs(out_path):
            out_path = os.path.join(CURRENT_DIR, out_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Dashboard kích thước: total_w=1020, total_h=540
        dashboard_w = pipeline.visualizer.total_w
        dashboard_h = pipeline.visualizer.total_h
        writer = cv2.VideoWriter(out_path, fourcc, video_fps, (dashboard_w, dashboard_h))
        print(f"[XUẤT VIDEO] Đang ghi hình vào: {out_path} ({dashboard_w}x{dashboard_h} @ {video_fps:.1f} FPS)")

    paused = False
    curr_frame_idx = start_frame
    frames_processed = 0
    tracked_count = 0
    tl_counts = {"GREEN": 0, "RED": 0, "YELLOW": 0, "NONE": 0}

    window_name = "UEH Team 7 - Video Vision Lab (CRC 2026)"
    delay_ms = max(1, int(1000.0 / args.fps))

    t_start = time.time()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[HẾT VIDEO] Đã đọc hết tất cả các khung hình!")
                    break

                curr_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                time_sec = curr_frame_idx / video_fps

                result = pipeline.process(
                    raw_frame=frame,
                    frame_idx=curr_frame_idx,
                    total_frames=total_frames,
                    time_sec=time_sec,
                )

                frames_processed += 1
                if "TRACK" in result.lane_status:
                    tracked_count += 1
                tl_counts[result.traffic_light_state] = tl_counts.get(result.traffic_light_state, 0) + 1

                if writer:
                    writer.write(result.dashboard_frame)

                dashboard_display = result.dashboard_frame
            else:
                # Đang Pause: giữ nguyên frame cũ
                pass

            if not args.headless:
                cv2.imshow(window_name, dashboard_display)
                key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

                if key in [ord("q"), 27]:  # Q hoặc ESC
                    print("\n[THOÁT] Người dùng đã dừng chương trình.")
                    break
                elif key == ord(" "):  # Phím SPACE
                    paused = not paused
                    print(f"\n[{'TẠM DỪNG' if paused else 'TIẾP TỤC'}] Frame {curr_frame_idx}")
                elif key in [ord("d"), 83]:  # Phím D hoặc mũi tên phải
                    if paused:
                        # Tiến 1 frame
                        ret, frame = cap.read()
                        if ret:
                            curr_frame_idx += 1
                            result = pipeline.process(frame, curr_frame_idx, total_frames, curr_frame_idx / video_fps)
                            dashboard_display = result.dashboard_frame
                    else:
                        # Nhảy tới +40 frame
                        new_f = min(total_frames - 1, curr_frame_idx + 40)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)
                elif key in [ord("a"), 81]:  # Phím A hoặc mũi tên trái
                    # Lùi frame
                    new_f = max(0, curr_frame_idx - (1 if paused else 40))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_f)
                    ret, frame = cap.read()
                    if ret:
                        curr_frame_idx = new_f
                        result = pipeline.process(frame, curr_frame_idx, total_frames, curr_frame_idx / video_fps)
                        dashboard_display = result.dashboard_frame
                elif key == ord("s"):  # Phím S: Lưu Snapshot
                    snap_name = f"snapshot_frame_{curr_frame_idx:04d}.jpg"
                    snap_path = os.path.join(config.SNAPSHOTS_DIR, snap_name)
                    cv2.imwrite(snap_path, dashboard_display)
                    print(f"\n📸 [LƯU ẢNH] Đã lưu snapshot tại: {snap_path}")
                elif key == ord("r"):  # Phím R: Về đầu
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    curr_frame_idx = 0
                elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")]:
                    idx_scene = int(chr(key))
                    scene_keys = list(config.KEY_SCENES.keys())
                    target_scene = scene_keys[idx_scene - 1]
                    target_frame = config.KEY_SCENES[target_scene]
                    print(f"\n[NHẢY CẢNH] Cảnh {idx_scene}: {target_scene} -> Frame {target_frame}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            # Kiểm tra số frame tối đa
            if args.max_frames and frames_processed >= args.max_frames:
                print(f"\n[HOÀN THÀNH] Đã xử lý đủ {args.max_frames} frames theo yêu cầu.")
                break

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"✅ [HOÀN TẤT] File video đã lưu thành công: {args.save_video}")
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("📈 TỔNG KẾT HIỆU NĂNG XỬ LÝ ẢNH:")
    print(f"⏱️ Tổng thời gian: {elapsed:.2f}s | Khung hình xử lý: {frames_processed}")
    if frames_processed > 0:
        print(f"⚡ Tốc độ xử lý trung bình: {frames_processed / elapsed:.1f} FPS")
        print(f"🛣️ Tỉ lệ bám làn thành công: {tracked_count / frames_processed * 100:.1f}% ({tracked_count}/{frames_processed})")
        print(f"🚦 Thống kê đèn giao thông: Xanh={tl_counts['GREEN']}, Đỏ={tl_counts['RED']}, Vàng={tl_counts['YELLOW']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
