#!/usr/bin/env python3
"""Static TF tree of the UEH CRC 2026 car, with the same frame names as the simulation:

  base_footprint -> base_link -> imu_link
                              -> base_scan
                              -> camera_link -> camera_rgb_frame -> camera_rgb_optical_frame

Positions (metres, relative to base_link) are parameters, so each car can be measured once.
Published once on /tf_static (latched), so it costs nothing at run time.
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class StaticFrames(Node):
    def __init__(self):
        super().__init__('static_frames')
        p = self.declare_parameter
        # Numeric parameters accept ints too (e.g. counts_per_rev:=1320 from an env file)
        fp = lambda name, default: float(  # noqa: E731
            p(name, default, ParameterDescriptor(dynamic_typing=True)).value)
        base_z = fp('tf_base_z', 0.010)
        imu = p('tf_imu_xyz', [0.0, 0.0, 0.05]).value
        scan = p('tf_scan_xyz', [0.0, 0.0, 0.15]).value
        cam = p('tf_camera_xyz', [0.10, 0.0, 0.12]).value
        cam_pitch = math.radians(fp('tf_camera_pitch_deg', 0.0))  # positive = tilted down

        frames = [
            ('base_footprint', 'base_link', (0.0, 0.0, base_z), (0.0, 0.0, 0.0)),
            ('base_link', 'imu_link', imu, (0.0, 0.0, 0.0)),
            ('base_link', 'base_scan', scan, (0.0, 0.0, 0.0)),
            ('base_link', 'camera_link', cam, (0.0, cam_pitch, 0.0)),
            ('camera_link', 'camera_rgb_frame', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ('camera_rgb_frame', 'camera_rgb_optical_frame', (0.0, 0.0, 0.0),
             (-math.pi / 2, 0.0, -math.pi / 2)),
        ]
        now = self.get_clock().now().to_msg()
        msgs = []
        for parent, child, xyz, rpy in frames:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, xyz)
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = quaternion(*rpy)
            msgs.append(t)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(msgs)
        self.get_logger().info(f'Published {len(msgs)} static frames')


def main():
    rclpy.init()
    node = StaticFrames()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
