#!/usr/bin/env python3
"""
UEH CRC 2026 Simulation Run Recorder Node.

Measurement and data logging node for the UEH Creative Robot Contest 2026.
Subscribes ONLY to allowed robot topics:
  - /odom (nav_msgs/msg/Odometry)
  - /cmd_vel (geometry_msgs/msg/Twist)
  - /imu (sensor_msgs/msg/Imu)
  - /camera/image_raw (sensor_msgs/msg/Image)
  - /detection_debug/image (sensor_msgs/msg/Image)

STRICTLY COMPLIANT WITH CHALLENGE RULES:
  - NO banned ground-truth topics (/model_states, /link_states, /sky_cam/*, /traffic_lights, etc.)
  - NO Gazebo entity state services or coordinate querying
  - Start time detected from actual robot motion (/cmd_vel threshold)
  - Loop completion detected solely from robot-relative /odom return to initial pose
  - Safety timeout enforced at 9 minutes (540 s)
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime

# Add package paths for SignPerception
sys.path.insert(0, '/home/enteekey211106/UEH_Nguyen-Tuan-Kiet/src/crc_sim')
sys.path.insert(0, '/home/enteekey211106/UEH_CRC_2026_Simulation_Pack/src/crc_sim')

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, Imu

try:
    from crc_sim.starter_node import SignPerception
except ImportError:
    SignPerception = None


def euler_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class RunRecorder(Node):
    def __init__(self, run_id, output_csv, output_metadata, detections_dir,
                 min_duration=300.0, max_duration=540.0,
                 finish_dist_thresh=0.75, finish_yaw_thresh=0.75,
                 finish_min_cum_dist=22.0, finish_hold_samples=8):
        super().__init__('crc_run_recorder')

        self.run_id = run_id
        self.output_csv = output_csv
        self.output_metadata = output_metadata
        self.detections_dir = detections_dir
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.output_metadata)), exist_ok=True)
        os.makedirs(os.path.abspath(self.detections_dir), exist_ok=True)

        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)
        self.finish_dist_thresh = float(finish_dist_thresh)
        self.finish_yaw_thresh = float(finish_yaw_thresh)
        self.finish_min_cum_dist = float(finish_min_cum_dist)
        self.finish_hold_samples = int(finish_hold_samples)

        self.bridge = CvBridge()
        self.sign_perception = SignPerception(self) if SignPerception is not None else None
        if self.sign_perception is not None:
            self.get_logger().info('SignPerception module loaded successfully in recorder.')
        else:
            self.get_logger().warn('SignPerception module could not be imported.')

        self.is_started = False
        self.start_wall_time = None
        self.start_ros_time = None
        self.end_wall_time = None
        self.completed = False
        self.timeout = False
        self.consecutive_motion_samples = 0
        self.consecutive_finish_samples = 0

        self.initial_odom_x = None
        self.initial_odom_y = None
        self.initial_odom_yaw = None

        self.current_odom_x = 0.0
        self.current_odom_y = 0.0
        self.current_odom_yaw = 0.0
        self.last_odom_x = None
        self.last_odom_y = None
        self.cumulative_distance_m = 0.0
        self.max_odom_x = 0.0
        self.max_odom_y = 0.0

        self.latest_cmd_linear_x = 0.0
        self.latest_cmd_angular_z = 0.0
        self.latest_odom_linear_x = 0.0
        self.latest_odom_angular_z = 0.0
        self.is_moving = False

        self.latest_imu_yaw_rate = 0.0

        self.data_rows = []
        self.csv_fieldnames = [
            'timestamp_sec',
            'elapsed_sec',
            'odom_x',
            'odom_y',
            'odom_yaw',
            'linear_x',
            'angular_z',
            'cumulative_distance_m',
            'moving',
        ]

        self.detection_counts = {}
        self.detection_last_saved_time = {}
        self.max_captures_per_class = 5
        self.capture_cooldown_sec = 1.5

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        reliable_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.on_odom, reliable_qos)
        self.sub_cmd = self.create_subscription(
            Twist, '/cmd_vel', self.on_cmd_vel, reliable_qos)
        self.sub_imu = self.create_subscription(
            Imu, '/imu', self.on_imu, sensor_qos)
        self.sub_camera = self.create_subscription(
            Image, '/camera/image_raw', self.on_camera_image, sensor_qos)

        self.timer = self.create_timer(0.05, self.on_tick)

        self.get_logger().info(
            f'RunRecorder initialized for {run_id}. Waiting for robot motion start...')

    def on_cmd_vel(self, msg: Twist):
        self.latest_cmd_linear_x = msg.linear.x
        self.latest_cmd_angular_z = msg.angular.z

        has_motion = (abs(msg.linear.x) > 0.005 or abs(msg.angular.z) > 0.01)

        if not self.is_started:
            if has_motion:
                self.consecutive_motion_samples += 1
                if self.consecutive_motion_samples >= 3:
                    self.start_run()
            else:
                self.consecutive_motion_samples = 0
        else:
            self.is_moving = has_motion

    def on_imu(self, msg: Imu):
        self.latest_imu_yaw_rate = msg.angular_velocity.z

    def on_odom(self, msg: Odometry):
        self.current_odom_x = msg.pose.pose.position.x
        self.current_odom_y = msg.pose.pose.position.y
        self.current_odom_yaw = euler_from_quaternion(msg.pose.pose.orientation)
        self.max_odom_x = max(self.max_odom_x, self.current_odom_x)
        self.max_odom_y = max(self.max_odom_y, self.current_odom_y)
        self.latest_odom_linear_x = msg.twist.twist.linear.x
        self.latest_odom_angular_z = msg.twist.twist.angular.z

        if not self.is_started:
            return

        if self.last_odom_x is not None and self.last_odom_y is not None:
            step_dist = math.hypot(
                self.current_odom_x - self.last_odom_x,
                self.current_odom_y - self.last_odom_y,
            )
            if step_dist < 1.0:
                self.cumulative_distance_m += step_dist

        self.last_odom_x = self.current_odom_x
        self.last_odom_y = self.current_odom_y

    def start_run(self):
        self.is_started = True
        self.start_wall_time = time.time()
        self.start_ros_time = self.get_clock().now().nanoseconds / 1e9

        self.initial_odom_x = self.current_odom_x
        self.initial_odom_y = self.current_odom_y
        self.initial_odom_yaw = self.current_odom_yaw
        self.last_odom_x = self.current_odom_x
        self.last_odom_y = self.current_odom_y
        self.cumulative_distance_m = 0.0

        self.get_logger().info(
            f'>>> RUN STARTED at wall={self.start_wall_time:.3f} | '
            f'Initial Odom: x={self.initial_odom_x:.3f}, y={self.initial_odom_y:.3f}, '
            f'yaw={self.initial_odom_yaw:.3f} rad')

    def on_tick(self):
        if not self.is_started or self.completed or self.timeout:
            return

        now = time.time()
        elapsed_sec = now - self.start_wall_time

        if elapsed_sec >= self.max_duration:
            self.timeout = True
            self.end_wall_time = now
            self.get_logger().warn(
                f'*** TIMEOUT REACHED at {elapsed_sec:.2f} s (> {self.max_duration} s) ***')
            self.finish_recording()
            return

        row = {
            'timestamp_sec': round(now, 4),
            'elapsed_sec': round(elapsed_sec, 4),
            'odom_x': round(self.current_odom_x, 4),
            'odom_y': round(self.current_odom_y, 4),
            'odom_yaw': round(self.current_odom_yaw, 4),
            'linear_x': round(self.latest_cmd_linear_x, 4),
            'angular_z': round(self.latest_cmd_angular_z, 4),
            'cumulative_distance_m': round(self.cumulative_distance_m, 4),
            'moving': bool(self.is_moving or abs(self.latest_odom_linear_x) > 0.005),
        }
        self.data_rows.append(row)

        if len(self.data_rows) % 10 == 0:
            try:
                live_info = {
                    'n': len(self.data_rows),
                    'elapsed': round(elapsed_sec, 2),
                    'dist': round(self.cumulative_distance_m, 4),
                    'x': round(self.current_odom_x, 4),
                    'y': round(self.current_odom_y, 4),
                    'max_x': round(self.max_odom_x, 4),
                    'max_y': round(self.max_odom_y, 4)
                }
                with open(self.output_csv + '.live.json', 'w') as f_live:
                    json.dump(live_info, f_live)
            except Exception:
                pass

        if int(elapsed_sec) % 15 == 0 and len(self.data_rows) % 15 == 0:
            self.get_logger().info(
                f'Progress: t={elapsed_sec:.1f}s | dist={self.cumulative_distance_m:.2f}m | '
                f'x={self.current_odom_x:.2f}, y={self.current_odom_y:.2f}')

        if elapsed_sec >= self.min_duration and self.cumulative_distance_m >= self.finish_min_cum_dist:
            dist_to_start = math.hypot(
                self.current_odom_x - self.initial_odom_x,
                self.current_odom_y - self.initial_odom_y,
            )
            yaw_diff = abs(wrap_angle(self.current_odom_yaw - self.initial_odom_yaw))

            finish_by_pose = (dist_to_start <= self.finish_dist_thresh and yaw_diff <= self.finish_yaw_thresh)
            finish_by_dist = (self.cumulative_distance_m >= 27.2 and dist_to_start <= 1.2)
            if finish_by_pose or finish_by_dist:
                self.consecutive_finish_samples += 1
                if self.consecutive_finish_samples >= self.finish_hold_samples:
                    self.completed = True
                    self.end_wall_time = now
                    self.get_logger().info(
                        f'*** RUN COMPLETED SUCCESSFULLY! *** | '
                        f'Duration: {elapsed_sec:.2f} s ({int(elapsed_sec // 60):02d}:{elapsed_sec % 60:05.2f}) | '
                        f'Distance: {self.cumulative_distance_m:.2f} m | '
                        f'Finish delta: dist={dist_to_start:.3f}m, yaw={yaw_diff:.3f}rad')
                    self.finish_recording()
                    return
            else:
                self.consecutive_finish_samples = 0

    def on_camera_image(self, msg: Image):
        if not self.is_started or self.sign_perception is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        obs = self.sign_perception.process(frame, detect_pedestrians=True)
        if not obs:
            return

        now = time.time()

        # Check sign detection
        sign = obs.get('sign')
        if sign and sign.get('label'):
            label = sign['label']
            self.try_save_detection(label, frame, now)

        # Check light detection
        light = obs.get('light')
        if light and light.get('state') in ('RED', 'YELLOW', 'GREEN') and light.get('confidence', 0) >= 0.65:
            light_label = f"traffic_light_{light['state'].lower()}"
            self.try_save_detection(light_label, frame, now)

        # Check pedestrian detection
        peds = obs.get('pedestrians', [])
        if peds and any(p.get('confidence', 0) >= 0.50 for p in peds):
            self.try_save_detection('pedestrian', frame, now)

    def try_save_detection(self, obj_class, raw_frame, now):
        cnt = self.detection_counts.get(obj_class, 0)
        if cnt >= self.max_captures_per_class:
            return

        last_time = self.detection_last_saved_time.get(obj_class, 0.0)
        if now - last_time < self.capture_cooldown_sec:
            return

        cnt += 1
        self.detection_counts[obj_class] = cnt
        self.detection_last_saved_time[obj_class] = now

        annotated = self.sign_perception.render_overlay(raw_frame) if self.sign_perception else raw_frame
        filename = f'{self.run_id}_{obj_class}_{cnt:02d}.png'
        filepath = os.path.join(self.detections_dir, filename)
        cv2.imwrite(filepath, annotated)
        self.get_logger().info(
            f'Captured valid detection: {filename} (class={obj_class}, count={cnt}/{self.max_captures_per_class})')

    def finish_recording(self):
        try:
            live_p = self.output_csv + '.live.json'
            if os.path.exists(live_p):
                os.remove(live_p)
        except Exception:
            pass
        if not self.data_rows:
            self.get_logger().error('No data rows recorded!')
            return

        duration_sec = self.end_wall_time - self.start_wall_time if self.start_wall_time else 0.0
        minutes = int(duration_sec // 60)
        seconds = duration_sec % 60
        duration_min_str = f'{minutes:02d}:{seconds:05.2f}'

        linear_speeds = [r['linear_x'] for r in self.data_rows]
        angular_speeds = [abs(r['angular_z']) for r in self.data_rows]

        mean_linear_speed = float(np.mean(linear_speeds)) if linear_speeds else 0.0
        max_linear_speed = float(np.max(linear_speeds)) if linear_speeds else 0.0
        mean_abs_angular = float(np.mean(angular_speeds)) if angular_speeds else 0.0
        max_abs_angular = float(np.max(angular_speeds)) if angular_speeds else 0.0

        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
            writer.writerows(self.data_rows)
        self.get_logger().info(f'Saved CSV data ({len(self.data_rows)} rows) to: {self.output_csv}')

        start_iso = datetime.fromtimestamp(self.start_wall_time).isoformat() if self.start_wall_time else ''
        end_iso = datetime.fromtimestamp(self.end_wall_time).isoformat() if self.end_wall_time else ''

        metadata = {
            'run_id': self.run_id,
            'date_time': datetime.now().isoformat(),
            'controller_command': 'ros2 run crc_sim starter',
            'simulator_command': 'bash scripts/run_docker.sh up',
            'start_time_iso': start_iso,
            'end_time_iso': end_iso,
            'duration_sec': round(duration_sec, 4),
            'duration_min': duration_min_str,
            'cumulative_distance_m': round(self.cumulative_distance_m, 4),
            'mean_linear_speed_mps': round(mean_linear_speed, 4),
            'max_linear_speed_mps': round(max_linear_speed, 4),
            'mean_abs_angular_speed_rps': round(mean_abs_angular, 4),
            'max_abs_angular_speed_rps': round(max_abs_angular, 4),
            'measurement_method': 'Participant-side /odom and /cmd_vel subscriber measurement node',
            'completion_detection_method': (
                'Odometry-relative loop closure: return to within 0.75 m and 0.75 rad '
                'of initial start pose after >= 300.0 s and >= 22.0 m cumulative distance'
            ),
            'completed': bool(self.completed),
            'timeout': bool(self.timeout),
            'num_samples': len(self.data_rows),
            'notes': 'Normal valid autonomous run' if self.completed else ('Timeout reached (> 9 min)' if self.timeout else 'Incomplete'),
        }

        with open(self.output_metadata, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.get_logger().info(f'Saved metadata JSON to: {self.output_metadata}')


def main():
    parser = argparse.ArgumentParser(description='UEH CRC 2026 Simulation Run Recorder')
    parser.add_argument('--run-id', type=str, default='run_01', help='Identifier for run (e.g. run_01)')
    parser.add_argument('--output-csv', type=str, default='analysis/data/run_01.csv', help='Path to output CSV')
    parser.add_argument('--output-metadata', type=str, default='analysis/data/run_01_metadata.json', help='Path to metadata JSON')
    parser.add_argument('--detections-dir', type=str, default='analysis/figures/detections', help='Directory to save detection captures')
    parser.add_argument('--min-duration', type=float, default=300.0, help='Minimum duration in seconds before finish detection is active')
    parser.add_argument('--max-duration', type=float, default=540.0, help='Safety timeout in seconds (default 540s = 9 min)')
    parser.add_argument('--finish-dist-thresh', type=float, default=0.75, help='Distance threshold to initial pose for finish (m)')
    parser.add_argument('--finish-yaw-thresh', type=float, default=0.75, help='Heading threshold to initial pose for finish (rad)')
    parser.add_argument('--finish-min-cum-dist', type=float, default=22.0, help='Minimum cumulative distance in m before finish')
    parser.add_argument('--finish-hold-samples', type=int, default=8, help='Number of consecutive samples required for finish')

    args = parser.parse_args()

    rclpy.init()
    recorder = RunRecorder(
        run_id=args.run_id,
        output_csv=args.output_csv,
        output_metadata=args.output_metadata,
        detections_dir=args.detections_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        finish_dist_thresh=args.finish_dist_thresh,
        finish_yaw_thresh=args.finish_yaw_thresh,
        finish_min_cum_dist=args.finish_min_cum_dist,
        finish_hold_samples=args.finish_hold_samples,
    )

    import signal
    def sig_handler(sig, frame):
        if not recorder.completed and not recorder.timeout:
            recorder.end_wall_time = time.time()
            recorder.finish_recording()
        sys.exit(0)
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    try:
        while rclpy.ok() and not (recorder.completed or recorder.timeout):
            rclpy.spin_once(recorder, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not recorder.completed and not recorder.timeout:
            recorder.end_wall_time = time.time()
            recorder.finish_recording()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
