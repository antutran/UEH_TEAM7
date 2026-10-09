#!/usr/bin/env python3
"""Camera bridge for the UEH CRC 2026 car.

The CSI camera (IMX219) can only be opened on the Jetson host through nvarguscamerasrc, so
the host service crc-camera (scripts/camera_stream.sh) scales and converts the frames on the
VIC hardware and streams raw RGBA frames over TCP to 127.0.0.1:5600. No JPEG decode is needed
here, which keeps the CPU load low on the Nano. This node publishes, like the simulation:

  /camera/image_raw             sensor_msgs/Image           rgb8, 640x480 by default
  /camera/camera_info           sensor_msgs/CameraInfo      pinhole model from the field of view
  /camera/image_raw/compressed  sensor_msgs/CompressedImage JPEG, encoded only while someone
                                                            subscribes (for viewing over Wi-Fi)

The node reconnects if the stream stops. camera_width/camera_height must match camera_stream.sh.
"""
import array
import math
import socket
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        p = self.declare_parameter
        # Numeric parameters accept ints too (e.g. counts_per_rev:=1320 from an env file)
        fp = lambda name, default: float(  # noqa: E731
            p(name, default, ParameterDescriptor(dynamic_typing=True)).value)
        self.addr = (p('camera_host', '127.0.0.1').value, p('camera_port', 5600).value)
        self.width = p('camera_width', 640).value
        self.height = p('camera_height', 480).value
        self.frame_id = p('camera_frame', 'camera_rgb_optical_frame').value
        # Horizontal field of view used for the pinhole intrinsics (IMX219 at full sensor width: ~62 deg)
        hfov_deg = fp('camera_hfov_deg', 62.2)
        self.jpeg_quality = p('camera_jpeg_quality', 70).value

        # Default (reliable) QoS like the simulation: a reliable publisher also serves best-effort
        # subscribers, while a best-effort one would not reach the students' depth-10 subscribers.
        self.pub_raw = self.create_publisher(Image, 'camera/image_raw', 5)
        self.pub_info = self.create_publisher(CameraInfo, 'camera/camera_info', 5)
        self.pub_jpeg = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 5)

        self.info = self.make_info(hfov_deg)
        # Reused message and pixel buffer: only the stamp and the pixels change per frame.
        # rclpy stores an array('B') as is, while assigning bytes costs ~100 ms per 640x480
        # frame (element-wise conversion), so the RGB pixels are copied straight into it.
        self.img = Image(height=self.height, width=self.width, encoding='rgb8',
                         is_bigendian=0, step=self.width * 3)
        self.img.header.frame_id = self.frame_id
        self.pixels = array.array('B', bytes(self.width * self.height * 3))
        self.pixels_np = np.frombuffer(self.pixels, np.uint8).reshape(self.height, self.width, 3)

        self.frames = 0
        self.connected = False
        self.create_timer(5.0, self.report)
        threading.Thread(target=self.reader, daemon=True).start()
        self.get_logger().info(
            f'Reading {self.width}x{self.height} RGBA frames from {self.addr[0]}:{self.addr[1]}')

    def make_info(self, hfov_deg):
        w, h = self.width, self.height
        fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        cx, cy = w / 2.0, h / 2.0
        info = CameraInfo(width=w, height=h, distortion_model='plumb_bob')
        info.header.frame_id = self.frame_id
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def reader(self):
        frame_bytes = self.width * self.height * 4
        buf = bytearray(frame_bytes)
        view = memoryview(buf)
        rgba = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 4)
        while rclpy.ok():
            try:
                with socket.create_connection(self.addr, timeout=3) as s:
                    s.settimeout(3)
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * frame_bytes)
                    self.connected = True
                    while rclpy.ok():
                        got = 0
                        while got < frame_bytes:  # one whole frame, received in place
                            n = s.recv_into(view[got:], frame_bytes - got)
                            if n == 0:
                                raise OSError('stream closed')
                            got += n
                        self.publish(rgba)
            except OSError:
                pass
            if self.connected:
                self.get_logger().warn('Camera stream lost, reconnecting (is crc-camera running?)')
            self.connected = False
            time.sleep(1.0)

    def publish(self, rgba):
        stamp = self.get_clock().now().to_msg()
        rgb = rgba[:, :, :3]  # view; the only copy is into the message buffer below
        if self.pub_raw.get_subscription_count() > 0:
            np.copyto(self.pixels_np, rgb)
            self.img.header.stamp = stamp
            self.img.data = self.pixels
            self.pub_raw.publish(self.img)
        self.info.header.stamp = stamp
        self.pub_info.publish(self.info)
        if self.pub_jpeg.get_subscription_count() > 0:
            ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                msg = CompressedImage(format='jpeg', data=jpeg.tobytes())
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                self.pub_jpeg.publish(msg)
        self.frames += 1

    def report(self):
        fps, self.frames = self.frames / 5.0, 0
        if self.connected:
            self.get_logger().info(f'camera: {fps:.1f} fps', throttle_duration_sec=60.0)


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
