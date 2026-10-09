#!/usr/bin/env python3
"""Red circle follower: an example of a team node for the UEH CRC 2026 car.

Uses only the topics the simulation also provides, so it runs unchanged in Gazebo and on
the car:
  subscribes /camera/image_raw  sensor_msgs/Image (rgb8)
  publishes  /cmd_vel           geometry_msgs/Twist
  publishes  /red_follower/debug/compressed  JPEG with the detection drawn, only while
             someone subscribes (e.g. rqt_image_view on a laptop)

Control: turn to keep the circle centred, drive to keep its apparent size at target_size
(diameter / image height). Stops when the circle is lost or the camera goes quiet.

Kept light for the Jetson Nano: no cv_bridge (the image is wrapped with numpy), processing
on a 2x subsampled frame, one contour pass per frame.
"""
import math
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class RedFollower(Node):
    def __init__(self):
        super().__init__('red_follower')
        p = self.declare_parameter
        self.enabled = p('enabled', True).value
        self.target_size = p('target_size', 0.30).value   # wanted diameter / image height
        self.v_max = p('v_max', 0.15).value               # m/s forward
        self.v_back = p('v_back', 0.08).value             # m/s backward
        self.w_max = p('w_max', 1.0).value                # rad/s
        self.k_w = p('k_w', 1.5).value                    # rad/s per unit of horizontal error (-1..1)
        self.k_v = p('k_v', 0.8).value                    # m/s per unit of size error
        self.min_area = p('min_area', 80).value           # px on the subsampled frame
        self.min_circularity = p('min_circularity', 0.6).value
        self.lost_timeout = p('lost_timeout', 0.5).value  # s without a detection -> stop
        self.step = p('subsample', 2).value               # process every n-th pixel per axis

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_debug = self.create_publisher(CompressedImage, '/red_follower/debug/compressed', 2)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 2)
        self.kernel = np.ones((3, 3), np.uint8)
        self.last_seen = 0.0
        self.last_image = time.monotonic()
        self.frames = 0
        self.proc_time = 0.0
        self.create_timer(0.1, self.watchdog)
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f'Following a red circle: target size {self.target_size}, v_max {self.v_max} m/s, '
            f'w_max {self.w_max} rad/s, enabled={self.enabled}')

    def detect(self, rgb):
        """-> (cx, cy, radius) in subsampled pixels, or None."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Red wraps around hue 0: combine both ends of the hue range
        mask = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            perimeter = cv2.arcLength(c, True)
            circularity = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            if circularity < self.min_circularity:
                continue
            if best is None or area > best[0]:
                best = (area, c)
        if best is None:
            return None
        (cx, cy), radius = cv2.minEnclosingCircle(best[1])
        return cx, cy, radius

    def on_image(self, msg):
        t0 = time.perf_counter()
        self.last_image = time.monotonic()
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().warn(f'unsupported encoding {msg.encoding}', throttle_duration_sec=5.0)
            return
        frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        small = np.ascontiguousarray(frame[::self.step, ::self.step])
        if msg.encoding == 'bgr8':
            small = small[:, :, ::-1]
        h, w = small.shape[:2]

        det = self.detect(small)
        cmd = Twist()
        if det is not None:
            cx, cy, r = det
            self.last_seen = time.monotonic()
            x_err = (cx - w / 2.0) / (w / 2.0)            # -1 (left) .. +1 (right)
            size = 2.0 * r / h
            cmd.angular.z = float(np.clip(-self.k_w * x_err, -self.w_max, self.w_max))
            v = self.k_v * (self.target_size - size)
            # Drive less while badly off-centre, so the car turns first
            v *= max(0.0, 1.0 - abs(x_err))
            cmd.linear.x = float(np.clip(v, -self.v_back, self.v_max))
        if self.enabled and det is not None:
            self.pub_cmd.publish(cmd)

        if self.pub_debug.get_subscription_count() > 0:
            dbg = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            if det is not None:
                cv2.circle(dbg, (int(det[0]), int(det[1])), int(det[2]), (0, 255, 0), 2)
                cv2.putText(dbg, f'v={cmd.linear.x:+.2f} w={cmd.angular.z:+.2f}', (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            ok, jpg = cv2.imencode('.jpg', dbg, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                out = CompressedImage(format='jpeg', data=jpg.tobytes())
                out.header = msg.header
                self.pub_debug.publish(out)

        self.frames += 1
        self.proc_time += time.perf_counter() - t0

    def watchdog(self):
        # Circle lost or camera silent: send an explicit stop (the base also stops on its own
        # 0.5 s /cmd_vel timeout, this just makes it immediate)
        now = time.monotonic()
        if self.enabled and (now - self.last_seen > self.lost_timeout or now - self.last_image > 0.5):
            if now - self.last_seen < self.lost_timeout + 0.3 or now - self.last_image < 0.8:
                self.pub_cmd.publish(Twist())

    def report(self):
        if self.frames:
            self.get_logger().info(
                f'{self.frames / 5.0:.1f} fps, {1000.0 * self.proc_time / self.frames:.1f} ms/frame, '
                f'circle {"seen" if time.monotonic() - self.last_seen < 1.0 else "not seen"}')
        else:
            self.get_logger().warn('no images on /camera/image_raw')
        self.frames, self.proc_time = 0, 0.0


def main():
    rclpy.init()
    node = RedFollower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.pub_cmd.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
