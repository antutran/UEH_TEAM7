#!/usr/bin/env python3
"""LiDAR scan adapter for the UEH CRC 2026 car.

The simulated TurtleBot3 LiDAR publishes 360 rays at 1 degree steps from 0 to 359 degrees,
counter-clockwise, ranges[0] pointing straight ahead, valid from 0.12 to 3.5 m. The RPLIDAR C1
driver (sllidar_ros2) publishes a different layout (about 720 rays from -180 to +180 degrees).
This node resamples /scan_raw into the simulation layout on /scan, so code written in the
simulator (e.g. ranges[0] = front, ranges[90] = left) works unchanged on the car.

Each output ray takes the closest valid return that falls inside its 1-degree bin; bins with
no valid return are +inf, as in the simulation. Fully vectorised with numpy.
"""
import array
import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanAdapter(Node):
    def __init__(self):
        super().__init__('scan_adapter')
        p = self.declare_parameter
        # Numeric parameters accept ints too (e.g. counts_per_rev:=1320 from an env file)
        fp = lambda name, default: float(  # noqa: E731
            p(name, default, ParameterDescriptor(dynamic_typing=True)).value)
        self.bins = p('scan_rays', 360).value
        self.range_min = fp('scan_range_min', 0.12)
        self.range_max = fp('scan_range_max', 3.5)
        # Angle of the car's forward direction as seen by the LiDAR (mounting rotation), degrees CCW
        self.yaw_offset = math.radians(fp('scan_yaw_offset_deg', 0.0))
        self.frame_id = p('scan_frame', 'base_scan').value

        self.out = LaserScan(angle_min=0.0,
                             angle_max=2.0 * math.pi * (self.bins - 1) / self.bins,
                             angle_increment=2.0 * math.pi / self.bins,
                             range_min=self.range_min, range_max=self.range_max)
        self.out.header.frame_id = self.frame_id
        self.angles_cache = (None, None)  # (key, angles) for the input layout

        self.pub = self.create_publisher(LaserScan, 'scan', 10)
        self.create_subscription(LaserScan, 'scan_raw', self.on_scan, qos_profile_sensor_data)
        self.get_logger().info(
            f'/scan_raw -> /scan: {self.bins} rays, {self.range_min}-{self.range_max} m, '
            f'yaw offset {math.degrees(self.yaw_offset):.1f} deg')

    def input_angles(self, msg, n):
        key = (msg.angle_min, msg.angle_increment, n)
        if self.angles_cache[0] != key:
            angles = msg.angle_min + msg.angle_increment * np.arange(n, dtype=np.float64)
            self.angles_cache = (key, angles)
        return self.angles_cache[1]

    def on_scan(self, msg):
        r = np.frombuffer(msg.ranges, dtype=np.float32) if isinstance(msg.ranges, array.array) \
            else np.asarray(msg.ranges, dtype=np.float32)
        a = self.input_angles(msg, r.size) - self.yaw_offset
        valid = np.isfinite(r) & (r >= self.range_min) & (r <= self.range_max)
        idx = np.rint(np.mod(a[valid], 2.0 * math.pi) / self.out.angle_increment).astype(np.int64) % self.bins
        out = np.full(self.bins, np.inf, dtype=np.float32)
        np.minimum.at(out, idx, r[valid])

        self.out.header.stamp = msg.header.stamp
        self.out.scan_time = msg.scan_time
        self.out.time_increment = msg.scan_time / self.bins if msg.scan_time > 0 else 0.0
        # array('f') is stored as is; a Python list would be converted element by element
        self.out.ranges = array.array('f', out.tobytes())
        self.pub.publish(self.out)


def main():
    rclpy.init()
    node = ScanAdapter()
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
