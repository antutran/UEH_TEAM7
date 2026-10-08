#!/usr/bin/env python3
"""Lane following and autonomous driving node for UEH CRC 2026.

Thuật toán bám DUY NHẤT VẠCH PHẢI (Pure Right-Line Follower):
1. CHỈ QUAN TÂM DUY NHẤT VẠCH PHẢI (Right Line Only):
   - Hoàn toàn bỏ qua mọi vạch bên trái (vạch tim đường, vạch đứt, làn ngược chiều, ngã rẽ trái).
   - Vùng quét chỉ lấy nửa bên phải khung hình (x từ 280px -> 635px).
   - Giữ vạch mép phải luôn nằm ở vị trí chuẩn (target_right_x = 495px trên khung hình 640px).
     + Nếu vạch phải lệch > 495px: Xe đang lệch sang trái -> Bẻ lái sang phải để ôm vạch.
     + Nếu vạch phải lệch < 495px: Xe đang quá sát mép phải -> Bẻ lái sang trái để giữ khoảng cách an toàn.
2. NẾU MẤT VẠCH PHẢI -> TIẾP TỤC ĐI THẲNG (Straight Cruise):
   - Khi đi qua các khoảng trống/mất vạch: Giữ thẳng tuyệt đối bánh lái (steer = 0.0) với tốc độ ổn định
     cho đến khi camera nhận lại được vạch phải.
3. BỘ ĐIỀU KHIỂN CHỐNG LẮC LƯ (Anti-Wobble):
   - Vùng chết Deadzone (|error| <= 7px): Xe đi thẳng mượt mà, triệt tiêu hiện tượng "rắn bò".
   - Lọc thông thấp góc lái (Low-pass filter) và giới hạn gia tốc quay (Slew rate limiter),
     giúp 2 bánh xe chuyển hướng cực kỳ êm ái.
4. TÍCH HỢP AN TOÀN LIDAR & TRỰC QUAN HÓA CAMERA THEO THỜI GIAN THỰC.
"""

import math
import os
import signal
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Float32

try:
    from cv_bridge import CvBridge
    HAVE_CV = True
except ImportError:
    HAVE_CV = False

try:
    from gazebo_msgs.msg import ModelStates
    HAVE_GAZEBO_MSGS = True
except ImportError:
    HAVE_GAZEBO_MSGS = False

try:
    from .lane_detector import LaneDetector
except ImportError:
    from lane_detector import LaneDetector


class Starter(Node):

    def __init__(self):
        super().__init__('crc_starter')

        # Thông số vận tốc và an toàn
        self.declare_parameter('max_speed', 0.14)        # m/s (tốc độ tối đa khi đi thẳng)
        self.declare_parameter('min_speed', 0.06)        # m/s (tốc độ tối thiểu khi cua gắt)
        self.declare_parameter('straight_speed', 0.12)   # m/s (tốc độ khi mất vạch đi thẳng)
        self.declare_parameter('max_turn', 0.9)          # rad/s (giới hạn tốc độ quay)
        self.declare_parameter('stop_distance', 0.28)    # m (khoảng cách phanh an toàn cách cản trước ~8cm)
        self.declare_parameter('rate', 20.0)             # Hz (tần số điều khiển)

        # Thông số bám DUY NHẤT VẠCH PHẢI (Căn xe chạy chuẩn giữa làn)
        self.declare_parameter('target_right_x', 527.0)  # pixel (vị trí vạch phải tại look_y=355px khi xe ở giữa làn)
        self.declare_parameter('target_near_x', 580.0)   # pixel (mốc gần tại near_y=403px khống chế xe song song vạch)
        self.declare_parameter('k_heading', 0.0)         # Triệt tiêu sai số góc ảo, tập trung đi thẳng tuyệt đối
        self.declare_parameter('look_y_ratio', 0.74)     # Hạ thấp điểm ERR xuống 74% chiều cao ảnh (gần xe hơn)
        self.declare_parameter('right_search_min', 300)  # Chỉ quét x >= 300px (loại bỏ 100% vạch tim đường và làn ngược chiều)
        self.declare_parameter('right_search_max', 638)  # Đến sát mép phải ảnh
        self.declare_parameter('kp', 0.0030)             # Hệ số tỉ lệ P (bám thẳng giữa làn)
        self.declare_parameter('kd', 0.0008)             # Hệ số vi phân D
        self.declare_parameter('deadzone', 12.0)         # pixel (vùng chết 12px: xe đi thẳng tuyệt đối, 2 bánh đồng tốc)
        self.declare_parameter('max_steer_step', 0.07)   # rad/s mỗi chu kỳ (giới hạn gia tốc bẻ lái)
        self.declare_parameter('roi_top', 0.54)          # Quét từ 54% chiều cao ảnh
        self.declare_parameter('roi_bottom', 0.88)       # Quét đến 88% chiều cao ảnh
        # Đảo chiều lệnh bẻ lái (Mặc định False: chuẩn theo driver xe)
        self.declare_parameter('invert_steering', False)

        # Cấu hình hiển thị cửa sổ Camera
        has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        self.declare_parameter('show_view', has_display)

        self.max_speed = float(self.get_parameter('max_speed').value)
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.straight_speed = float(self.get_parameter('straight_speed').value)
        self.max_turn = float(self.get_parameter('max_turn').value)
        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.rate = float(self.get_parameter('rate').value)
        self.invert_steering = bool(self.get_parameter('invert_steering').value)

        self.target_right_x = float(self.get_parameter('target_right_x').value)
        self.target_near_x = float(self.get_parameter('target_near_x').value)
        self.k_heading = float(self.get_parameter('k_heading').value)
        self.look_y_ratio = float(self.get_parameter('look_y_ratio').value)
        self.right_search_min = int(self.get_parameter('right_search_min').value)
        self.right_search_max = int(self.get_parameter('right_search_max').value)

        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.deadzone = float(self.get_parameter('deadzone').value)
        self.max_steer_step = float(self.get_parameter('max_steer_step').value)
        self.roi_top = float(self.get_parameter('roi_top').value)
        self.roi_bottom = float(self.get_parameter('roi_bottom').value)
        self.show_view = bool(self.get_parameter('show_view').value)

        # Biến trạng thái điều khiển & bộ lọc
        self.prev_error = 0.0
        self.d_error_filtered = 0.0
        self.last_valid_steer = 0.0
        self.smooth_right_x = None

        # IMU Gyroscope heading-hold stabilizer
        self.gyro_z = 0.0              # Tốc độ quay Z hiện tại (rad/s) từ IMU
        self.imu_heading_offset = 0.0  # Tích phân offset để bù lái khi xe đang đi thẳng
        self.last_imu_time = None      # Thời điểm IMU message cuối
        # Spike filter: loại bỏ nhiễu camera nhảy vọt đột ngột khi IMU nói xe đang thẳng
        self.prev_lookahead_x = None   # Giá trị lookahead_x frame trước
        self.spike_hold_counter = 0    # Số frame liên tiếp đang reject spike

        # Cảm biến pin (voltage topic)
        self.battery_voltage = None    # V (None = chưa nhận)

        # Logic rẽ sau hầm (khi đường line cam bị kéo lệch hết sang phải, thực hiện 1 lần duy nhất)
        self.post_tunnel_state = 'IDLE'       # 'IDLE' -> 'STRAIGHT' -> 'TURN_RIGHT' -> 'DONE'
        self.post_tunnel_maneuver_done = False
        self.post_tunnel_trigger_count = 0
        self.post_tunnel_timer_start = 0.0
        self.post_tunnel_turn_start_yaw = 0.0
        self.post_tunnel_straight_time = 3.0     # Thời gian đi thẳng (giây)
        self.post_tunnel_turn_target_deg = 58.0  # Góc cua phải (độ)
        self.post_tunnel_trigger_x = 590.0       # Ngưỡng chạm x bên phải (px) - càng nhỏ càng kích hoạt sớm/sát hơn

        # Logic vượt xe né xe dừng trên cao tốc (parked_robot màu xanh, thực hiện 1 lần duy nhất)
        self.overtake_state = 'IDLE'             # 'IDLE' -> 'OVERTAKE_STEER_LEFT' -> 'OVERTAKE_DIAG_LEFT' -> 'OVERTAKE_STRAIGHTEN_LEFT' -> 'OVERTAKE_FOLLOW_LEFT' -> 'OVERTAKE_STEER_RIGHT' -> 'OVERTAKE_DIAG_RIGHT' -> 'OVERTAKE_STRAIGHTEN_RIGHT' -> 'DONE'
        self.overtake_done = False
        self.overtake_timer_start = 0.0
        self.overtake_trigger_dist = 0.70        # Khoảng cách phát hiện xe dừng phía trước (m)
        self.overtake_steer_time = 0.90          # Thời gian bẻ lái sang trái để chuyển làn (giây)
        self.overtake_steer_right_time = 1.30    # Thời gian rẽ phải sau khi bám lane trái để về lại làn phải (giây, tăng thêm theo yêu cầu)
        self.overtake_diag_time = 2.50           # Thời gian chạy chéo sang làn (giây)
        self.overtake_left_follow_time = 3.0    # Thời gian bám vạch trái một lát trước khi về lại làn phải (giây)
        self.overtake_speed_turn = 0.12          # Vận tốc khi bẻ lái (m/s)
        self.overtake_speed_straight = 0.14      # Vận tốc khi chạy thẳng/chéo (m/s)
        self.overtake_turn_rate = 0.85           # Tốc độ quay bẻ lái (rad/s)

        # Trạng thái theo dõi vị trí hầm: TRƯỚC HẦM CHỈ DÒ LANE, SAU HẦM MỚI ÁP DỤNG CÁC LOGIC THỦ CÔNG
        self.entered_tunnel = False
        self.has_passed_tunnel = False

        # Cảm biến
        self.image = None       # BGR image (480x640x3)
        self.scan = None        # sensor_msgs/LaserScan
        self.x = self.y = self.yaw = 0.0
        self._last_log = {}

        self.bridge = CvBridge() if HAVE_CV else None
        if not HAVE_CV:
            self.get_logger().warn(
                'cv_bridge not found! Cần cài đặt ros-humble-cv-bridge python3-opencv')

        self.declare_parameter('lane_only_mode', True)
        self.lane_only_mode = bool(self.get_parameter('lane_only_mode').value)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_lane_debug = self.create_publisher(Image, '/camera/lane_debug', 10)
        self.pub_debug_legacy = self.create_publisher(Image, '/lane_debug/image', 10)
        self.pub_debug_cam = self.create_publisher(Image, '/camera/debug_image', 10)

        self.create_subscription(Image, '/camera/image_raw',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan',
                                 self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(Imu, '/imu', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Float32, '/voltage', self.on_voltage, 10)
        if HAVE_GAZEBO_MSGS:
            self.create_subscription(ModelStates, '/model_states', self.on_model_states, 10)

        self.declare_parameter('kp_curvature', 0.0)    # Tắt bù cong khi đi thẳng
        self.kp_curvature = float(self.get_parameter('kp_curvature').value)

        self.lane_detector = LaneDetector(
            white_v_min=130,
            white_s_max=90,
            tophat_thresh=35,
            roi_top_ratio=0.54,
            near_y_ratio=0.84,
            look_y_ratio=self.look_y_ratio,
        )
        self.last_detector_debug = None
        self.last_curvature = 0.0

        self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(
            f'Anti-Glare Sliding Windows Follower ready | Target X={self.target_right_x}px | Speed={self.max_speed}m/s')

    # --- Sensor Callbacks ---

    def on_image(self, msg):
        if self.bridge is None:
            return
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.log_every(3.0, f'[CAMERA OK] Nhan frame {msg.width}x{msg.height}')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.y >= 1.20:
            self.has_passed_tunnel = True

    def on_imu(self, msg):
        """Nhận gyro Z từ IMU (dùng angular_velocity.z vì orientation_covariance[0]=-1 = invalid)."""
        now = time.time()
        gz = msg.angular_velocity.z
        # Lọc nhiễu nhỏ gyro (deadband 0.01 rad/s)
        if abs(gz) < 0.01:
            gz = 0.0
        # Lọc thông thấp gyro_z để tránh spike cảm biến
        self.gyro_z = 0.7 * gz + 0.3 * self.gyro_z
        self.last_imu_time = now

    def on_voltage(self, msg):
        """Nhận điện áp pin (V) từ cảm biến dòng trên xe."""
        self.battery_voltage = float(msg.data)

    def on_model_states(self, msg):
        if 'waffle' in msg.name:
            idx = msg.name.index('waffle')
            p = msg.pose[idx].position
            q = msg.pose[idx].orientation
            self.x, self.y = p.x, p.y
            self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            if self.y >= 1.20:
                self.has_passed_tunnel = True

    # --- Helper Functions ---

    def range_at(self, angle_deg, width_deg=18.0, min_dist=0.20):
        if self.scan is None or not self.scan.ranges:
            return float('inf')

        n = len(self.scan.ranges)
        half = int(round(width_deg / 2.0))
        centre = int(round(angle_deg)) % 360
        valid_ranges = []
        for d in range(-half, half + 1):
            idx = int((centre + d) % 360 * n / 360)
            r = self.scan.ranges[idx]
            # Bỏ qua các điểm < 0.20m (nhiễu phản xạ khung xe, cọc đỡ Waffle hoặc mặt dốc sát cản trước khi dốc chúi mũi)
            if math.isfinite(r) and r >= min_dist:
                valid_ranges.append(r)

        if len(valid_ranges) < 3:
            return float('inf')

        # Xác nhận vật cản bằng tối thiểu 3 tia quét để triệt tiêu hoàn toàn tia nhiễu đơn lẻ
        valid_ranges.sort()
        return valid_ranges[2]

    def drive(self, v, w):
        # Đảo chiều góc lái nếu phần cứng Yahboom bị đấu ngược kênh motor trái/phải
        w_cmd = -w if self.invert_steering else w

        # Forward-Only Differential Drive Policy:
        # Half-track = 0.1475m. Elevate forward velocity so inner wheel never rotates backward.
        wheel_half_track = 0.1475
        min_inner_forward = 0.025
        if v > 0.0:
            min_v_needed = min_inner_forward + abs(w_cmd) * wheel_half_track
            v = max(v, min_v_needed)

        msg = Twist()
        msg.linear.x = float(max(-self.max_speed, min(self.max_speed, v)))
        msg.angular.z = float(max(-self.max_turn, min(self.max_turn, w_cmd)))
        self.pub_cmd.publish(msg)

    def stop(self):
        try:
            self.pub_cmd.publish(Twist())
        except Exception:
            pass

    def log_every(self, seconds, text):
        now = time.time()
        if now - self._last_log.get(text[:20], 0.0) >= seconds:
            self._last_log[text[:20]] = now
            self.get_logger().info(text)

    def tick(self):
        try:
            self.control()
        except Exception as e:
            self.get_logger().error(f'control() raised: {e}')
            self.stop()

    # --- Thuật toán Xử lý ảnh: BÁM VẠCH CHỐNG CHÓI & CỬA SỔ TRƯỢT (LaneDetector) ---

    def detect_lines(self, img):
        if img is None:
            return None, None, 0.0, None, (0, 0), 'LOST', None, None

        target_lane_x, lookahead_x, curvature, mask, debug_img, tracking_mode, r_near, r_look = (
            self.lane_detector.process_frame(
                img,
                target_right_x=self.target_right_x,
            )
        )
        self.last_detector_debug = debug_img
        self.last_curvature = curvature

        if lookahead_x is not None:
            if self.smooth_right_x is None or abs(lookahead_x - self.smooth_right_x) > 80.0:
                self.smooth_right_x = lookahead_x
            else:
                self.smooth_right_x = 0.70 * lookahead_x + 0.30 * self.smooth_right_x
            filtered_lookahead = self.smooth_right_x
        else:
            self.smooth_right_x = None
            filtered_lookahead = None

        h = img.shape[0]
        roi_y = (int(h * self.lane_detector.roi_top_ratio), int(h * self.lane_detector.near_y_ratio))
        return target_lane_x, filtered_lookahead, curvature, mask, roi_y, tracking_mode, r_near, r_look

    def detect_right_line(self, img):
        target_lane_x, lookahead_x, curvature, mask, roi_y, tracking_mode, r_near, r_look = self.detect_lines(img)
        return lookahead_x, mask, roi_y, tracking_mode

    def render_debug_frame(self, img, right_x, mask, roi_y, status_text, speed, steer, error, feature_type='LINE'):
        if self.last_detector_debug is not None:
            debug = self.last_detector_debug.copy()
        else:
            debug = img.copy() if img is not None else np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = debug.shape[:2]
        # Thanh thông số HUD điều khiển phía dưới
        cv2.rectangle(debug, (8, h - 46), (w - 8, h - 8), (20, 24, 33), -1)
        cv2.rectangle(debug, (8, h - 46), (w - 8, h - 8), (0, 255, 0), 1)
        cv2.putText(debug, f'TRANG THAI: {status_text}', (16, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(debug, f'TOC DO: {speed:.2f} m/s | LAI: {steer:+.2f} rad/s | LECH: {error:+.1f}px | CHE DO: PHAI',
                    (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1, cv2.LINE_AA)

        # IMU gyro indicator (góc phải dưới HUD)
        gyro_text = f'GYRO: {self.gyro_z:+.3f} r/s'
        cv2.putText(debug, gyro_text, (w - 185, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1, cv2.LINE_AA)

        # Battery display (góc trên phải)
        if self.battery_voltage is not None:
            v = self.battery_voltage
            # 3S LiPo: full=12.6V, empty=9.6V
            batt_pct = max(0.0, min(100.0, (v - 9.6) / (12.6 - 9.6) * 100.0))
            # Màu sắc: xanh lá >50%, vàng >20%, đỏ <=20%
            if batt_pct > 50:
                batt_color = (0, 220, 50)
            elif batt_pct > 20:
                batt_color = (0, 200, 220)
            else:
                batt_color = (0, 60, 255)
            batt_text = f'PIN: {batt_pct:.0f}% ({v:.1f}V)'
            # Vẽ nền nhỏ
            (tw, th), _ = cv2.getTextSize(batt_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(debug, (w - tw - 14, 6), (w - 6, 6 + th + 8), (20, 24, 33), -1)
            cv2.rectangle(debug, (w - tw - 14, 6), (w - 6, 6 + th + 8), batt_color, 1)
            cv2.putText(debug, batt_text, (w - tw - 10, 6 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, batt_color, 1, cv2.LINE_AA)

        return debug

    # ------------------------------------------------------------------------
    # VÒNG LẶP ĐIỀU KHIỂN CHÍNH
    # ------------------------------------------------------------------------

    def control(self):
        # 1. Kiểm tra an toàn bằng LiDAR phía trước (góc 18 độ tránh chạm thành hầm)
        front_obstacle = self.range_at(0, width_deg=18.0)
        is_blocked = front_obstacle < self.stop_distance

        # Xác nhận đã ra khỏi hầm: Xe bắt buộc phải lên nửa trên sa bàn (y >= 1.20m).
        # Toàn bộ khu vực trước hầm và trong hầm luôn có y <= 0.0m (cửa ra hầm tại y = 0.0m).
        if self.y >= 1.20:
            self.has_passed_tunnel = True

        # 2. Xử lý ảnh: BÁM VẠCH PHẢI / BÁM RÌA DỐC CẦU / BÁM VẠCH TRÁI
        if self.image is None:
            self.stop()
            self.log_every(2.0, '[CAMERA WAIT] Dang cho /camera/image_raw (self.image is None)...')
            return

        target_lane_x, lookahead_x, curvature, mask, roi_y, tracking_mode, r_near, r_look = self.detect_lines(self.image)
        right_x = r_look

        speed = 0.0
        steer = 0.0
        error_display = 0.0
        feature_type = tracking_mode
        if not self.has_passed_tunnel:
            status_text = 'DO LANE TRUOC & TRONG HAM'
        else:
            status_text = f'BAM LAN ({tracking_mode})'

        # TOÀN BỘ CÁC LOGIC THỦ CÔNG CHỈ ĐƯỢC PHÉP KÍCH HOẠT SAU KHI ĐÃ ĐI QUA HẦM:
        if self.has_passed_tunnel:
            # Logic 1: Ngã ba sau hầm (khi vạch cam bị kéo lệch sang phải qua ngưỡng chạm 590px -> đi thẳng 3s -> ôm cua 60 độ)
            if not self.post_tunnel_maneuver_done and self.post_tunnel_state == 'IDLE':
                raw_err = (right_x - self.target_right_x) if right_x is not None else 0.0
                trigger_x = self.post_tunnel_trigger_x
                line_pulled_extreme = (right_x is not None) and (right_x >= trigger_x or raw_err >= (trigger_x - self.target_right_x))

                if line_pulled_extreme:
                    self.post_tunnel_trigger_count += 1
                    if self.post_tunnel_trigger_count >= 2:  # Đạt 2 frame (~0.1s) là kích hoạt ngay
                        self.post_tunnel_state = 'STRAIGHT'
                        self.post_tunnel_timer_start = time.time()
                        self.get_logger().info(
                            f'[KICH HOAT] Duong line cam cham nguong (rx={right_x:.1f}px >= {trigger_x:.0f}px, err={raw_err:+.1f}px) '
                            f'-> DI THANG {self.post_tunnel_straight_time:.1f}s -> CUA {self.post_tunnel_turn_target_deg:.0f} DO')
                else:
                    self.post_tunnel_trigger_count = 0

            # Logic 2: Vượt xe né xe dừng trên cao tốc
            # Chỉ kích hoạt sau khi đã ra khỏi hầm VÀ (đã xong ngã ba HOẶC đã ở trên đường cao tốc x < 0.5)
            is_highway = self.has_passed_tunnel and (self.post_tunnel_maneuver_done or self.x < 0.5)
            if not self.overtake_done and self.overtake_state == 'IDLE' and is_highway:
                if 0.20 <= front_obstacle <= self.overtake_trigger_dist:
                    self.overtake_state = 'OVERTAKE_STEER_LEFT'
                    self.overtake_timer_start = time.time()
                    self.get_logger().info(
                        f'[VUOT XE] Phat hien xe dung phia truoc o {front_obstacle:.2f}m <= {self.overtake_trigger_dist:.2f}m '
                        f'-> KICH HOAT NE XE QUA LAN TRAI!')

        # 0. Ưu tiên cao nhất: Chuỗi động tác VƯỢT XE NÉ XE DỪNG TRÊN CAO TỐC
        if self.overtake_state not in ('IDLE', 'DONE'):
            elapsed = time.time() - self.overtake_timer_start
            v_t = self.overtake_speed_turn
            v_s = self.overtake_speed_straight
            w_r = self.overtake_turn_rate

            if self.overtake_state == 'OVERTAKE_STEER_LEFT':
                if elapsed < self.overtake_steer_time:
                    speed = v_t
                    steer = w_r   # Bẻ lái sang trái (+w)
                    status_text = f'VUOT XE: BE LAI TRAI ({self.overtake_steer_time - elapsed:.1f}s)'
                    self.drive(speed, steer)
                else:
                    self.overtake_state = 'OVERTAKE_DIAG_LEFT'
                    self.overtake_timer_start = time.time()
                    self.drive(v_s, 0.0)
                    self.get_logger().info('[VUOT XE] Chuyen sang chay cheo sang lan trai...')

            elif self.overtake_state == 'OVERTAKE_DIAG_LEFT':
                if elapsed < self.overtake_diag_time:
                    speed = v_s
                    steer = 0.0   # Chạy chéo thẳng sang làn trái
                    status_text = f'VUOT XE: CHAY CHEO SANG TRAI ({self.overtake_diag_time - elapsed:.1f}s)'
                    self.drive(speed, steer)
                else:
                    self.overtake_state = 'OVERTAKE_STRAIGHTEN_LEFT'
                    self.overtake_timer_start = time.time()
                    self.drive(v_t, -w_r)
                    self.get_logger().info('[VUOT XE] Tra thang lai song song lan trai...')

            elif self.overtake_state == 'OVERTAKE_STRAIGHTEN_LEFT':
                if elapsed < self.overtake_steer_time:
                    speed = v_t
                    steer = -w_r  # Bẻ lái sang phải để trả thẳng song song trục đường (-w)
                    status_text = f'VUOT XE: TRA LAI SONG SONG ({self.overtake_steer_time - elapsed:.1f}s)'
                    self.drive(speed, steer)
                else:
                    self.overtake_state = 'OVERTAKE_FOLLOW_LEFT'
                    self.overtake_timer_start = time.time()
                    self.smooth_right_x = None
                    self.last_valid_steer = 0.0
                    self.get_logger().info(
                        f'[VUOT XE] Da vao lan trai -> KHOA DO LANE, DI THANG trong {self.overtake_left_follow_time:.1f}s qua mat xe dung...')

            elif self.overtake_state == 'OVERTAKE_FOLLOW_LEFT':
                if elapsed < self.overtake_left_follow_time:
                    # Đi thẳng trên làn trái, KHÔNG DÒ LANE theo yêu cầu
                    speed = v_s
                    steer = 0.0
                    self.last_valid_steer = 0.0
                    status_text = f'VUOT XE: DI THANG LAN TRAI ({self.overtake_left_follow_time - elapsed:.1f}s)'
                    self.drive(speed, 0.0)
                else:
                    # ĐÃ ĐI THẲNG QUA MẶT XE DỪNG -> ĐÁNH LÁI VỀ PHÍA VẠCH PHẢI!
                    self.overtake_state = 'OVERTAKE_STEER_RIGHT'
                    self.overtake_timer_start = time.time()
                    self.smooth_right_x = None
                    self.last_valid_steer = 0.0
                    self.drive(v_t, -w_r)
                    self.get_logger().info(
                        f'[VUOT XE] Da di thang lan trai {self.overtake_left_follow_time:.1f}s qua mat xe dung '
                        f'-> BAT DAU DANH LAI VE PHIA VACH PHAI ({self.overtake_steer_right_time:.1f}s)!')

            elif self.overtake_state == 'OVERTAKE_STEER_RIGHT':
                if elapsed < self.overtake_steer_right_time:
                    speed = v_t
                    steer = -w_r  # Bẻ lái sang phải để chuyển về làn ban đầu (-w)
                    status_text = f'VUOT XE: BE LAI PHAI VE LAN ({self.overtake_steer_right_time - elapsed:.1f}s)'
                    self.drive(speed, steer)
                else:
                    self.overtake_state = 'OVERTAKE_DIAG_RIGHT'
                    self.overtake_timer_start = time.time()
                    self.drive(v_s, 0.0)
                    self.get_logger().info('[VUOT XE] Chay cheo tro ve lan phai...')

            elif self.overtake_state == 'OVERTAKE_DIAG_RIGHT':
                if elapsed < self.overtake_diag_time:
                    speed = v_s
                    steer = 0.0   # Chạy chéo thẳng về làn phải
                    status_text = f'VUOT XE: CHAY CHEO VE PHAI ({self.overtake_diag_time - elapsed:.1f}s)'
                    self.drive(speed, steer)
                else:
                    self.overtake_state = 'OVERTAKE_STRAIGHTEN_RIGHT'
                    self.overtake_timer_start = time.time()
                    self.drive(v_t, w_r)
                    self.get_logger().info('[VUOT XE] Tra thang lai song song lan phai...')

            elif self.overtake_state == 'OVERTAKE_STRAIGHTEN_RIGHT':
                if elapsed < self.overtake_steer_right_time:
                    speed = v_t
                    steer = w_r   # Bẻ lái sang trái để trả thẳng song song vạch (+w)
                    status_text = f'VUOT XE: TRA LAI VE LAN PHAI ({self.overtake_steer_right_time - elapsed:.1f}s)'
                    self.drive(speed, steer)
                else:
                    self.overtake_state = 'DONE'
                    self.overtake_done = True
                    self.smooth_right_x = None
                    self.last_valid_steer = 0.0
                    self.prev_error = 0.0
                    self.d_error_filtered = 0.0
                    self.drive(self.straight_speed, 0.0)
                    self.get_logger().info(
                        '[VUOT XE] HOAN TAT VUOT XE! Da tro ve lan phai an toan -> TIEP TUC BAM LANE PHAI.')

        # 1. Ưu tiên phanh dừng nếu có vật cản ngoài chuỗi vượt xe
        elif is_blocked:
            self.stop()
            self.last_valid_steer = 0.0
            status_text = f'DUNG XE (Vat can {front_obstacle:.2f}m)'
            self.log_every(1.5, f'[CANH BAO] Vat can o {front_obstacle:.2f}m -> PHANH DUNG')

        # 2. Logic sau hầm - Giai đoạn 1: Tiếp tục đi thẳng trong 2 giây (tăng thêm 1s)
        elif self.post_tunnel_state == 'STRAIGHT':
            elapsed = time.time() - self.post_tunnel_timer_start
            straight_duration = self.post_tunnel_straight_time
            if elapsed < straight_duration:
                speed = self.straight_speed
                steer = 0.0
                self.last_valid_steer = 0.0
                status_text = f'QUA HAM: DI THANG ({straight_duration - elapsed:.1f}s)'
                self.drive(speed, steer)
                self.log_every(0.5, f'[QUA HAM] Tiep tuc di thang... con {straight_duration - elapsed:.1f}s')
            else:
                self.post_tunnel_state = 'TURN_RIGHT'
                self.post_tunnel_timer_start = time.time()
                self.post_tunnel_turn_start_yaw = self.yaw
                target_deg = self.post_tunnel_turn_target_deg
                self.get_logger().info(f'[QUA HAM] Het {straight_duration:.1f}s di thang -> Bat dau CUA VONG CUNG {target_deg:.0f} DO SANG PHAI (KHOA DO LINE)')
                speed = 0.08
                steer = -0.70
                status_text = f'QUA HAM: CUA PHAI {target_deg:.0f} DO (0/{target_deg:.0f} do)'
                self.drive(speed, steer)

        # 3. Logic sau hầm - Giai đoạn 2: Cua vòng cung sang phải 60 độ (KHÔNG nhận tín hiệu dò line)
        elif self.post_tunnel_state == 'TURN_RIGHT':
            elapsed = time.time() - self.post_tunnel_timer_start
            speed = 0.08
            steer = -0.70
            target_deg = self.post_tunnel_turn_target_deg

            # Tính góc đã quay sang phải (chuẩn hóa độ lệch yaw)
            delta_yaw = (self.yaw - self.post_tunnel_turn_start_yaw + math.pi) % (2.0 * math.pi) - math.pi
            turn_rad = -delta_yaw   # Quay phải: yaw giảm -> turn_rad dương
            turn_deg = math.degrees(turn_rad)

            status_text = f'QUA HAM: CUA PHAI {target_deg:.0f} DO ({max(0.0, turn_deg):.0f}/{target_deg:.0f} do)'
            self.drive(speed, steer)
            self.log_every(0.5, f'[QUA HAM] Dang om cua {target_deg:.0f} do (khoa do line)... goc quay: {turn_deg:.1f}/{target_deg:.0f} do ({elapsed:.1f}s)')

            max_turn_time = math.radians(target_deg) / 0.70 + 0.15
            turn_finished = (turn_deg >= (target_deg - 2.0)) or (elapsed >= max_turn_time)

            if turn_finished:
                self.post_tunnel_state = 'DONE'
                self.post_tunnel_maneuver_done = True
                self.smooth_right_x = None
                self.last_valid_steer = 0.0
                self.prev_error = 0.0
                self.d_error_filtered = 0.0
                self.get_logger().info(
                    f'[QUA HAM] Da om cua xong goc {target_deg:.0f} do ({turn_deg:.1f} do, {elapsed:.1f}s) '
                    '-> BAT DAU NHAN LAI TIN HIEU DO LANE!')

        # 4. Chế độ bám lane bình thường (Pure Right-Lane Follower + IMU Gyro Stabilizer)
        else:
            if lookahead_x is not None:
                # --- IMU Camera Spike Rejection ---
                # Nếu camera nhảy vọt đột ngột (>40px so với frame trước) nhưng gyro nói xe đang thẳng
                # (|gyro_z| < 0.08 rad/s) thì đây là spike ảo → giữ nguyên giá trị frame trước.
                is_camera_spike = False
                if self.prev_lookahead_x is not None:
                    cam_jump = abs(lookahead_x - self.prev_lookahead_x)
                    imu_is_straight = abs(self.gyro_z) < 0.08
                    if cam_jump > 40.0 and imu_is_straight:
                        is_camera_spike = True
                        self.spike_hold_counter = min(self.spike_hold_counter + 1, 5)

                if is_camera_spike and self.spike_hold_counter <= 4:
                    # Giữ nguyên lookahead của frame trước để loại nhiễu camera
                    lookahead_x = self.prev_lookahead_x
                    status_text += ' [SPIKE!]'
                else:
                    self.spike_hold_counter = 0

                self.prev_lookahead_x = lookahead_x

                # 1. Sai số khoảng cách ngang tại điểm nhìn xa lookahead
                e_look = lookahead_x - self.target_right_x
                raw_error = e_look
                error_display = raw_error

                # --- IMU Gyro Heading-Hold bù vào sai số ---
                # Khi xe đi thẳng (|raw_error| <= deadzone * 2): dùng gyro_z để bù giữ thẳng.
                # gyro_z > 0 = xe đang quay trái → cần bù phải → steer_imu âm.
                # gyro_z < 0 = xe đang quay phải → cần bù trái → steer_imu dương.
                # Hệ số k_gyro: chuyển đổi rad/s → rad/s lái. Mạnh vừa để ổn định trên thẳng.
                k_gyro = 0.25  # Điều chỉnh nếu cần: tăng lên nếu còn lắc, giảm nếu quá giật
                steer_imu = -self.gyro_z * k_gyro

                # Vùng chết Deadzone: khi xe ở sát điểm giữa (|raw_error| <= deadzone),
                # thẳng lái 100% để 2 bánh quay hoàn toàn đồng tốc, không bị lệch bánh!
                if abs(raw_error) <= self.deadzone:
                    error = 0.0
                    target_steer = 0.0
                    self.prev_error = 0.0
                    self.d_error_filtered = 0.0
                    # Chỉ dùng IMU bù thẳng trong deadzone (camera quá ổn rồi không cần PD)
                    steer = steer_imu
                    steer = max(-self.max_steer_step * 3, min(self.max_steer_step * 3, steer))
                    self.last_valid_steer = steer
                else:
                    error = raw_error - math.copysign(self.deadzone, raw_error)

                    # Lọc vi phân D (Low-pass filtered derivative)
                    dt = 1.0 / self.rate
                    raw_de = (error - self.prev_error) / dt if dt > 0 else 0.0
                    self.d_error_filtered = 0.60 * raw_de + 0.40 * self.d_error_filtered
                    self.prev_error = error

                    # 1. Bù lái phản hồi PD (Feedback bám thẳng giữa làn) + IMU gyro
                    steer_pd = - (self.kp * error + self.kd * self.d_error_filtered)
                    target_steer = steer_pd + steer_imu

                    # 2. Bộ lọc mượt tay lái (Slew rate & Exponential Smoothing)
                    filtered_steer = 0.40 * target_steer + 0.60 * self.last_valid_steer
                    steer_diff = filtered_steer - self.last_valid_steer
                    if abs(steer_diff) > self.max_steer_step:
                        filtered_steer = self.last_valid_steer + math.copysign(self.max_steer_step, steer_diff)

                    steer = filtered_steer
                    self.last_valid_steer = steer

                # 4. Điều chỉnh vận tốc thích nghi (Adaptive Speed)
                turn_factor = min(1.0, max(abs(raw_error) / 80.0, abs(curvature) * 120.0))
                speed = self.max_speed - (self.max_speed - self.min_speed) * turn_factor

                # 5. Truyền lệnh qua drive() với chính sách Forward-Only Differential Drive
                self.drive(speed, steer)
                self.log_every(2.0, f'{status_text}: err={raw_error:+.1f}px | gyro={self.gyro_z:+.3f}r/s | steer={steer:+.2f} rad/s | v={speed:.2f} m/s')

            else:
                # MẤT MỤC TIÊU -> Duy trì góc lái giảm dần để vào cua mượt mà, tránh mất lái đột ngột
                status_text = 'MAT VACH -> GIU LAI & DI THANG'
                speed = self.straight_speed
                steer = self.last_valid_steer * 0.70
                self.last_valid_steer = steer
                self.prev_error = 0.0
                self.d_error_filtered = 0.0
                self.smooth_right_x = None
                self.drive(speed, steer)
                self.log_every(2.0, f'Mat muc tieu -> Giu lai steer={steer:+.2f}, v={speed:.2f} m/s')

        # 3. Tạo khung hình trực quan & Hiển thị
        if self.image is not None and roi_y is not None:
            debug_frame = self.render_debug_frame(
                self.image, right_x, mask, roi_y, status_text, speed, steer, error_display, feature_type)

            # Phát ra ROS topic /camera/lane_debug và /lane_debug/image (cho Web Viewer)
            if self.bridge is not None and debug_frame is not None:
                try:
                    debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
                    self.pub_lane_debug.publish(debug_msg)
                    self.pub_debug_legacy.publish(debug_msg)
                    self.pub_debug_cam.publish(debug_msg)
                except Exception:
                    pass

            # Hiển thị trực tiếp nếu có giao diện
            if self.show_view and debug_frame is not None:
                try:
                    cv2.imshow('UEH CRC 2026 - Camera Do Lan (Lane Tracking)', debug_frame)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().warn(f'Khong the hien thi cua so OpenCV: {e}')
                    self.show_view = False


def catch_sigterm():
    stopping = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))
    return stopping


def spin(node, stopping):
    while rclpy.ok() and not stopping['now']:
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception:
            if stopping['now'] or not rclpy.ok():
                break
            raise


def main(args=None):
    rclpy.init(args=args)
    stopping = catch_sigterm()
    node = Starter()
    try:
        spin(node, stopping)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
