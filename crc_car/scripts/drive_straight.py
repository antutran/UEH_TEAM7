#!/usr/bin/env python3
"""Drive straight at a constant speed until stopped, holding the heading with the IMU gyro.
Stops (and resumes) on an obstacle in front according to /scan. Test tool for the UEH CRC car."""
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32

SPEED = 0.15          # m/s
K_HEADING = 1.5       # rad/s per rad of heading error
W_MAX = 0.5           # rad/s
STOP_DIST = 0.40      # m, obstacle in the front sector -> stop
RESUME_DIST = 0.60    # m, resume when clear again
SECTOR_DEG = 20       # front sector half-width
# OBSTACLE_STOP=0 disables the obstacle stop (the car then only stops when this node is stopped)
OBSTACLE_STOP = os.environ.get("OBSTACLE_STOP", "1") != "0"


class DriveStraight(Node):
    def __init__(self):
        super().__init__("drive_straight")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Imu, "/imu", self.on_imu, 50)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Float32, "/voltage", lambda m: setattr(self, "volt", m.data), 10)
        self.yaw = 0.0
        self.t_imu = None
        self.front = float("inf")
        self.blocked = False
        self.odom0 = None
        self.dist = 0.0
        self.volt = 0.0
        self.t_start = time.monotonic()
        self.create_timer(0.05, self.control)
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            f"Driving straight at {SPEED} m/s, obstacle stop "
            f"{f'< {STOP_DIST} m' if OBSTACLE_STOP else 'DISABLED'}")

    def on_imu(self, msg):
        t = time.monotonic()
        if self.t_imu is not None:
            self.yaw += msg.angular_velocity.z * (t - self.t_imu)
        self.t_imu = t

    def on_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float32)
        n = r.size
        k = int(round(math.radians(SECTOR_DEG) / msg.angle_increment))
        sector = np.concatenate([r[:k + 1], r[n - k:]])
        valid = sector[np.isfinite(sector)]
        self.front = float(valid.min()) if valid.size else float("inf")

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.odom0 is None:
            self.odom0 = (p.x, p.y)
        self.dist = math.hypot(p.x - self.odom0[0], p.y - self.odom0[1])

    def control(self):
        if not OBSTACLE_STOP:
            self.blocked = False
        elif self.blocked and self.front > RESUME_DIST:
            self.blocked = False
            self.get_logger().info(f"path clear ({self.front:.2f} m), resuming")
        elif not self.blocked and self.front < STOP_DIST:
            self.blocked = True
            self.get_logger().warn(f"obstacle {self.front:.2f} m ahead, stopping")
        cmd = Twist()
        if not self.blocked and self.t_imu is not None:
            cmd.linear.x = SPEED
            cmd.angular.z = float(max(-W_MAX, min(W_MAX, -K_HEADING * self.yaw)))
        self.pub.publish(cmd)

    def report(self):
        self.get_logger().info(
            f"t={time.monotonic() - self.t_start:5.0f} s  odom distance {self.dist:5.2f} m  "
            f"heading error {math.degrees(self.yaw):+5.1f} deg  front {self.front:4.2f} m  "
            f"{'BLOCKED' if self.blocked else 'driving'}  battery {self.volt:.1f} V")


def main():
    rclpy.init()
    node = DriveStraight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
