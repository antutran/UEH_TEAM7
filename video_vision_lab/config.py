#!/usr/bin/env python3
"""
Configuration parameters for Video Vision Lab (UEH Team 7 - CRC 2026).
Tất cả các tham số điều khiển thuật toán xử lý ảnh cho video sa bàn được tập trung tại đây.
"""

import os

# --- ĐƯỜNG DẪN DỮ LIỆU ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DEFAULT_VIDEO_PATH = os.path.join(
    PROJECT_DIR, "images", "Screen Recording 2026-10-09 at 20.20.59.mov"
)
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "images")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")

# --- KÍCH THƯỚC KHUNG HÌNH CHUẨN HOÁ ---
# Video gốc 2268x1702 được scale về 640x480 (chuẩn camera trên xe Jetson Nano)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- BỘ LỌC CHỐNG CHÓI VÀ TẠO MASK VẠCH ĐƯỜNG (LANE MASK) ---
TOPHAT_KERNEL_W = 25
TOPHAT_KERNEL_H = 5
TOPHAT_THRESHOLD = 32

# Điều kiện màu trắng vạch kẻ đường (HSV)
WHITE_V_MIN = 125          # Giá trị sáng tối thiểu
WHITE_S_MAX = 90           # Độ bão hòa tối đa (trắng là phi bão hòa)

# Vùng quan tâm (ROI)
ROI_TOP_RATIO = 0.52       # Cắt bỏ phần trên (đường chân trời, người, tường)
NEAR_Y_RATIO = 0.84        # Điểm kiểm soát gần (đáy xe)
LOOKAHEAD_Y_RATIO = 0.72   # Điểm nhìn xa (dẫn hướng tay lái)
TARGET_RIGHT_X = 510.0     # Tọa độ X vạch phải lý tưởng khi xe chạy thẳng giữa làn

# Hành lang an toàn vạch phải (Corridor)
CORRIDOR_TOP_LEFT_X = 110
CORRIDOR_BOT_LEFT_X = 140
CORRIDOR_MARGIN_RIGHT = 10

# Phân biệt vạch chính dày vs mép bạt mỏng
MIN_LINE_AREA = 80
MIN_LINE_LENGTH = 30
MIN_THICKNESS_THRESH = 10.0

# --- THUẬT TOÁN BÁM LÀN SLIDING WINDOWS ---
N_WINDOWS = 10
WINDOW_HEIGHT = 16
WINDOW_HALF_W = 48
MIN_PIXELS_IN_WINDOW = 3
OUTLIER_JUMP_THRESH = 75.0  # Chặn nhảy vọt tọa độ do vệt chói bất ngờ

# --- THUẬT TOÁN NHẬN DIỆN ĐÈN GIAO THÔNG (TRAFFIC LIGHT) ---
# Đèn thi đấu CRC: Hộp đen dọc, 3 bóng: XANH (trên), VÀNG (giữa), ĐỎ (dưới)
TL_ROI_Y_MIN = 80
TL_ROI_Y_MAX = 380
TL_ROI_X_MIN = 180
TL_ROI_X_MAX = 580

# Ngưỡng màu đèn sáng (LED Blooming)
GREEN_MIN_INTENSITY = 135
GREEN_PROMINENCE = 14      # G > (R+B)/2 + 14

RED_MIN_INTENSITY = 135
RED_PROMINENCE = 24        # R > max(G,B) + 24

YELLOW_MIN_INTENSITY = 145

MIN_LAMP_AREA = 35
MAX_LAMP_AREA = 1800

# --- THUẬT TOÁN NHẬN DIỆN BIỂN BÁO (TRAFFIC SIGNS) ---
SIGN_ROI_Y_MIN = 185       # Biển báo cắm trên cột ven đường (loại bỏ người đứng xa)
SIGN_ROI_Y_MAX = 320
SIGN_BLUE_H_MIN = 100
SIGN_BLUE_H_MAX = 135
SIGN_BLUE_S_MIN = 38
SIGN_BLUE_V_MIN = 45

SIGN_TEMPLATE_SIZE = (48, 48)
SIGN_MATCH_THRESHOLD = 0.45  # Ngưỡng khớp mẫu chính xác cao

# --- CÁC MỐC KHUNG HÌNH ĐẶC TRƯNG TRONG VIDEO ---
KEY_SCENES = {
    "1_START_GREEN_LIGHT": 0,       # Vạch xuất phát, đèn xanh bật
    "2_GLARE_STRAIGHT": 360,        # Đoạn thẳng chói đèn trần
    "3_SHARP_RIGHT_CURVE": 840,     # Cua gắt sang phải
    "4_BRIDGE_RAMP": 1080,          # Chân dốc cầu & vòng xuyến
    "5_RED_LIGHT_STOP": 1540,       # Đèn đỏ ngã tư & vạch dừng
    "6_RED_LIGHT_CLOSE": 1600,      # Đèn đỏ cự ly gần
}
