#!/usr/bin/env python3
"""All car bring-up nodes in ONE process: base driver, camera bridge, scan adapter and the
static TF tree. One process instead of four saves RAM (each rclpy process costs ~40 MB) and
DDS discovery traffic on the Jetson Nano. Parameter names are unique across the nodes, so a
single parameter set can be passed to the process.
"""
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

from crc_bringup.base_driver import BaseDriver
from crc_bringup.camera_node import CameraNode
from crc_bringup.scan_adapter import ScanAdapter
from crc_bringup.static_frames import StaticFrames


def main():
    rclpy.init()
    base = BaseDriver()
    nodes = [base, CameraNode(), ScanAdapter(), StaticFrames()]
    executor = SingleThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        base.stop_motors()
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
