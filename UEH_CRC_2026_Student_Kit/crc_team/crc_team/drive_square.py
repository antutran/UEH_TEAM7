#!/usr/bin/env python3
"""First program: drive a square using /odom, then stop.

    ros2 run crc_team drive_square
    ros2 run crc_team drive_square --ros-args -p side:=0.5 -p speed:=0.15 -p laps:=1

Shows the basics every team needs: publish /cmd_vel at a steady rate, read /odom, and
stop cleanly (the car also stops by itself 0.5 s after /cmd_vel goes quiet).
Put the car on the floor with about 1 x 1 m of free space to its front-left.
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def angle_diff(a, b):
    """a - b wrapped to [-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


class DriveSquare(Node):
    def __init__(self):
        super().__init__('drive_square')
        self.side = self.declare_parameter('side', 0.5).value          # m
        self.speed = self.declare_parameter('speed', 0.15).value       # m/s
        self.turn_speed = self.declare_parameter('turn_speed', 1.0).value  # rad/s
        self.laps = self.declare_parameter('laps', 1).value
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.pose = None                # (x, y, yaw) from /odom
        self.start = None               # pose at the start of the current leg
        self.leg = 0                    # even = straight, odd = turn
        self.create_timer(0.05, self.step)   # 20 Hz control loop

    def on_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def step(self):
        if self.pose is None:
            return                      # wait for the first odometry message
        if self.leg >= 8 * self.laps:
            self.pub.publish(Twist())   # done: keep sending zero
            return
        if self.start is None:
            self.start = self.pose
        x0, y0, yaw0 = self.start
        x, y, yaw = self.pose
        cmd = Twist()
        if self.leg % 2 == 0:           # straight
            done = math.hypot(x - x0, y - y0)
            remaining = self.side - done
            # Slow down over the last 10 cm so the car does not overshoot
            cmd.linear.x = self.speed * min(1.0, max(0.3, remaining / 0.10))
            # Hold the heading of the start of the leg
            cmd.angular.z = 2.0 * angle_diff(yaw0, yaw)
            finished = remaining <= 0.0
        else:                           # turn 90 degrees left
            turned = angle_diff(yaw, yaw0)
            remaining = math.pi / 2 - turned
            cmd.angular.z = self.turn_speed * min(1.0, max(0.3, remaining / 0.3))
            finished = remaining <= 0.02
        if finished:
            self.get_logger().info(f'leg {self.leg + 1}/{8 * self.laps} done at '
                                   f'x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f} deg')
            self.leg += 1
            self.start = None
            cmd = Twist()
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = DriveSquare()
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
