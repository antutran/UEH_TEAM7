#!/usr/bin/env python3
"""UEH CRC 2026 race template: a complete, simple driver to build on.

    ros2 launch crc_team race.launch.py              # car settings  (config/race_car.yaml)
    ros2 launch crc_team race.launch.py config:=sim  # simulator     (config/race_sim.yaml)

What it does, 20 times a second:
  * lane following: camera -> bird's-eye view -> lane centre -> pure-pursuit steering
  * START: waits for the green light (HSV colour detector, see light.py)
  * signs: reacts to /signs from the on-board YOLO detector (stop, slow down, junction
    direction, highway, red/green light), using the class names in the config file
  * obstacles: stops for anything in its lane closer than stop_distance (LiDAR); on the
    highway it overtakes a car that stays in front of it
Every number is in the YAML config file, so tuning needs no code change.
This is a starting point, not a winning solution: measure, tune and extend it.
"""
import math
import time

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan

from crc_team.birdseye import BirdsEye
from crc_team.config import DEFAULTS
from crc_team.lane import LaneDetector
from crc_team.light import TrafficLight



def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Race(Node):
    def __init__(self):
        super().__init__('race')
        path = self.declare_parameter('config', '').value
        self.cfg = dict(DEFAULTS)
        if path:
            with open(path) as f:
                self.cfg.update(yaml.safe_load(f) or {})
            self.get_logger().info(f'config: {path}')
        unknown = set(self.cfg) - set(DEFAULTS)
        if unknown:
            self.get_logger().warn(f'unknown config keys (typo?): {sorted(unknown)}')
        c = self.cfg

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_debug = self.create_publisher(CompressedImage, '/lane/debug/compressed', 2)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        try:                            # /signs exists on the car, not in the simulator
            from vision_msgs.msg import Detection2DArray
            self.create_subscription(Detection2DArray, '/signs', self.on_signs, 10)
        except ImportError:
            self.get_logger().warn('vision_msgs not installed: /signs ignored')

        self.light = TrafficLight(roi=c['light_roi'], min_area=c['light_min_area'],
                                  max_area=c['light_max_area'])
        self.lane = None                # built once the camera intrinsics are known
        self.frame = None               # latest RGB frame and its sequence number
        self.frame_seq = 0
        self.used_seq = 0
        self.result = None
        self.scan = None
        self.odom = None                # (x, y, yaw)
        self.dist = 0.0                 # distance driven (m), from /odom
        self.last_odom_xy = None

        self.state = 'WAIT_START' if c['wait_for_green'] else 'DRIVE'
        self.green_count = 0
        self.stop_until = 0.0
        self.slow_until = 0.0
        self.branch_until = 0.0
        self.highway = False
        self.red_seen = 0.0             # time of the last red-light detection
        self.cooldown = {}              # sign class -> time it may trigger again
        self.obstacle_since = None
        self.overtake_start = None
        self.offset = 0.0               # lateral target offset (m), + = left
        self.last_ok = 0.0
        self.last_w = 0.0
        self.last_ok_dist = 0.0         # odometry distance when the lane was last seen
        self.kappa = 0.0                # curvature the car is steering on (1/m, + = left)
        self.stats = {'frames': 0, 'lane_ms': 0.0}
        self.create_timer(0.05, self.step)
        self.create_timer(1.0, self.report)

    # --- callbacks: only store data, the work happens in step() ------------------------
    def on_image(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = arr[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        self.frame = img
        self.frame_seq += 1

    def on_info(self, msg):
        if self.lane is None:
            self.build_lane(np.array(msg.k).reshape(3, 3), (msg.width, msg.height))

    def on_scan(self, msg):
        self.scan = msg

    def on_odom(self, msg):
        p = msg.pose.pose
        xy = (p.position.x, p.position.y)
        if self.last_odom_xy is not None:
            self.dist += math.hypot(xy[0] - self.last_odom_xy[0], xy[1] - self.last_odom_xy[1])
        self.last_odom_xy = xy
        self.odom = (xy[0], xy[1], yaw_of(p.orientation))

    def on_signs(self, msg):
        c = self.cfg
        now = time.monotonic()
        for det in msg.detections:
            if not det.results:
                continue
            name = det.results[0].hypothesis.class_id
            if det.bbox.size_y < c['sign_min_height']:
                continue                # too far away to act on yet
            if name in c['red_classes']:
                self.red_seen = now
                continue
            if name in c['green_classes']:
                self.red_seen = 0.0
                continue
            if now < self.cooldown.get(name, 0.0):
                continue                # this sign was handled a moment ago
            self.cooldown[name] = now + c['sign_cooldown']
            self.get_logger().info(f'sign: {name} ({det.bbox.size_y:.0f} px tall)')
            if name in c['stop_classes']:
                self.stop_until = now + c['stop_time']
            elif name in c['slow_classes']:
                self.slow_until = now + c['slow_time']
            elif name in c['left_classes']:
                self.set_branch('left', now)
            elif name in c['right_classes']:
                self.set_branch('right', now)
            elif name in c['straight_classes']:
                self.set_branch('both', now)
            elif name in c['hw_enter_classes']:
                self.highway = True
            elif name in c['hw_exit_classes']:
                self.highway = False

    def set_branch(self, side, now):
        if self.lane is not None:
            self.lane.prefer = side
        self.branch_until = now + self.cfg['branch_time']

    # --- setup ------------------------------------------------------------------------------
    def build_lane(self, K, size):
        c = self.cfg
        if len(c['camera_matrix']) == 9:
            K = np.array(c['camera_matrix'], np.float64).reshape(3, 3)
        be = BirdsEye(K, c['dist_coeffs'], c['fisheye'], c['be_x_min'], c['be_x_max'],
                      c['be_half_width'], c['be_ppm'], size)
        if len(c['birdseye_points']) == 8 and len(c['birdseye_ground']) == 8:
            be.from_points(np.reshape(c['birdseye_points'], (4, 2)),
                           np.reshape(c['birdseye_ground'], (4, 2)))
            how = 'calibrated points'
        else:
            be.from_geometry(c['cam_height'], c['cam_x'], c['cam_y'], math.radians(c['cam_pitch_deg']))
            how = (f'geometry h={c["cam_height"]} m, x={c["cam_x"]} m, '
                   f'pitch={c["cam_pitch_deg"]} deg')
        self.lane = LaneDetector(be, c['lane_width'], c['line_width'], c['min_contrast'])
        seen = 100.0 * be.valid.mean()
        self.get_logger().info(f"bird's-eye {be.width}x{be.height} px from {how}; "
                               f'{seen:.0f}% of it is in the camera view')
        if seen < 30:
            self.get_logger().warn("most of the bird's-eye area is outside the camera view: "
                                   'check cam_height/cam_pitch_deg or be_x_min/be_x_max')

    # --- perception helpers --------------------------------------------------------------
    def obstacle_ahead(self, centre_y=0.0):
        """True when a LiDAR point lies in our corridor closer than stop_distance. The
        corridor bends with the arc the car is steering on, so walls in bends (tunnel) and
        cars in the other lane are not mistaken for obstacles."""
        s = self.scan
        if s is None:
            return False
        r = np.asarray(s.ranges, np.float32)
        a = s.angle_min + np.arange(r.size) * s.angle_increment
        ok = np.isfinite(r) & (r > s.range_min)
        x = r[ok] * np.cos(a[ok]) + self.cfg['lidar_x']
        y = r[ok] * np.sin(a[ok])
        path_y = centre_y + 0.5 * self.kappa * x * x       # arc of curvature kappa
        hit = (x > self.cfg['body_front']) & (x < self.cfg['stop_distance']) & \
              (np.abs(y - path_y) < self.cfg['corridor_half'])
        return bool(hit.sum() >= 2)     # 2 rays: ignore single noisy points

    # --- main loop ------------------------------------------------------------------------
    def step(self):
        c = self.cfg
        now = time.monotonic()
        if self.lane is None or self.frame is None:
            return
        if self.frame_seq != self.used_seq:      # process each camera frame once
            self.used_seq = self.frame_seq
            frame = self.frame
            t0 = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            self.result = self.lane.detect(gray)
            self.stats['lane_ms'] += (time.perf_counter() - t0) * 1000
            self.stats['frames'] += 1
            if self.state == 'WAIT_START':
                colour, _, _ = self.light.detect(frame)
                self.green_count = self.green_count + 1 if colour == 'green' else 0
            if self.pub_debug.get_subscription_count() > 0:
                self.publish_debug()
        if now > self.branch_until and self.lane.prefer != 'both':
            self.lane.prefer = 'both'

        v, w = 0.0, 0.0
        if self.state == 'WAIT_START':
            if self.green_count >= c['green_frames']:
                self.get_logger().info('green light: GO')
                self.state = 'DRIVE'
        elif now < self.stop_until:
            pass                                     # stop sign: wait
        elif now - self.red_seen < c['red_clear_time']:
            pass                                     # red light: wait
        elif self.state == 'OVERTAKE':
            v, w = self.follow_lane(now, target_offset=c['lane_width'])
            if self.dist - self.overtake_start > c['overtake_length'] and \
                    not self.obstacle_ahead(centre_y=-c['lane_width']):
                self.get_logger().info('overtake done, back to the right lane')
                self.state = 'DRIVE'
        else:                                        # DRIVE
            # Steer first, so the obstacle corridor follows the current lane even while
            # the car waits; then decide whether the corridor is free
            v, w = self.follow_lane(now, target_offset=0.0)
            if self.obstacle_ahead():
                v, w = 0.0, 0.0
                self.obstacle_since = self.obstacle_since or now
                waited = now - self.obstacle_since
                if self.highway and c['overtake'] and waited > c['overtake_wait']:
                    self.get_logger().info('obstacle on the highway: overtaking')
                    self.state = 'OVERTAKE'
                    self.overtake_start = self.dist
                    self.obstacle_since = None
                # else: stay stopped (pedestrian, car in front, ...)
            else:
                self.obstacle_since = None
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.pub.publish(cmd)

    def follow_lane(self, now, target_offset):
        """Pure pursuit on the lane centre; returns (v, w)."""
        c = self.cfg
        # Move the target offset smoothly (lane change over ~0.5 s)
        self.offset += float(np.clip(target_offset - self.offset, -0.05, 0.05))
        res = self.result
        vmax = c['highway_speed'] if self.highway else c['max_speed']
        if now < self.slow_until:
            vmax *= c['slow_factor']
        if res is not None and res.ok:
            L = c['lookahead']
            tx, ty = res.target(L)
            ty += self.offset
            kappa = 2.0 * ty / (tx * tx + ty * ty)  # circle through the car and the target
            self.kappa = kappa
            v = max(c['min_speed'], vmax / (1.0 + c['curve_slowdown'] * abs(kappa)))
            w = float(np.clip(v * kappa, -c['max_turn'], c['max_turn']))
            self.last_ok, self.last_w, self.last_ok_dist = now, w, self.dist
            return v, w
        if now - self.last_ok < c['lost_timeout']:
            return c['min_speed'], self.last_w       # lines lost briefly: keep turning
        if self.dist - self.last_ok_dist < c['blind_distance']:
            return c['min_speed'], 0.0               # zebra, junction...: go straight, slowly
        return 0.0, 0.0                              # really lost: stop (a referee repositions)

    def publish_debug(self):
        img = self.lane.draw(self.result)
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = 'jpeg'
            msg.data = jpg.tobytes()
            self.pub_debug.publish(msg)

    def report(self):
        n = self.stats['frames']
        res = self.result
        lines = '-' if res is None else \
            ('L' if res.left is not None else '.') + ('R' if res.right is not None else '.')
        ms = self.stats['lane_ms'] / n if n else 0.0
        self.get_logger().info(
            f'{self.state:<10} lines={lines} offset={res.offset(self.cfg["lookahead"]) if res and res.ok else 0:+.3f} m '
            f'frames={n}/s lane={ms:.1f} ms highway={self.highway} dist={self.dist:.1f} m')
        self.stats = {'frames': 0, 'lane_ms': 0.0}


def main():
    rclpy.init()
    node = Race()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())   # stop the car before exiting
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
