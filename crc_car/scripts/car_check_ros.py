#!/usr/bin/env python3
"""ROS part of car_check.sh; runs inside the crc_bringup container (stdin of python3 -).
Listens to the driver topics for a few seconds and prints one result per line:
  PASS|WARN|FAIL|INFO <tab> check <tab> detail
--motion also drives the wheels (they must be lifted off the ground)."""
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Float32, Float32MultiArray
from tf2_msgs.msg import TFMessage

LISTEN = 5.0                      # seconds of listening for the rate checks
BATT_OK, BATT_LOW = 11.1, 10.5    # 3S pack: 12.6 V full, ~11.1 V nominal
STATIC_FRAMES = ('base_link', 'imu_link', 'base_scan', 'camera_link',
                 'camera_rgb_frame', 'camera_rgb_optical_frame')


def out(level, check, detail):
    print(f'{level}\t{check}\t{detail}', flush=True)


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Probe:
    def __init__(self, node):
        self.node = node
        self.stamps = {}   # receive times of the messages in the current window
        self.last = {}
        self.wheel = []
        self.recording = False
        self.static = set()
        self.dynamic = set()
        # Reliable QoS, as team code uses: best effort drops fragments of the 900 kB images.
        # The image arrives serialised (raw=True): no per-frame Python decoding cost.
        node.create_subscription(Image, '/camera/image_raw', self.cb('camera'), 5, raw=True)
        node.create_subscription(LaserScan, '/scan', self.cb('scan'), 10)
        node.create_subscription(Odometry, '/odom', self.cb('odom'), 10)
        node.create_subscription(Imu, '/imu', self.cb('imu', keep=50), 50)
        node.create_subscription(Float32, '/voltage', self.cb('voltage', keep=50), 10)
        node.create_subscription(Float32MultiArray, '/wheel_speed', self.on_wheel, 50)
        node.create_subscription(TFMessage, '/tf', self.on_tf, 50)
        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(TFMessage, '/tf_static', self.on_tf_static, latched)
        if os.environ.get('CRC_YOLO', '1') == '1':
            from vision_msgs.msg import Detection2DArray
            node.create_subscription(Detection2DArray, '/signs', self.cb('signs'), 10)

    def cb(self, key, keep=1):
        self.stamps[key] = []
        self.last[key] = []

        def f(msg):
            self.stamps[key].append(time.monotonic())
            buf = self.last[key]
            buf.append(msg)
            if len(buf) > keep:
                del buf[0]
        return f

    def on_wheel(self, msg):
        if self.recording:
            self.wheel.append((time.monotonic(), msg.data[0], msg.data[1]))

    def on_tf(self, msg):
        for t in msg.transforms:
            self.dynamic.add((t.header.frame_id, t.child_frame_id))

    def on_tf_static(self, msg):
        for t in msg.transforms:
            self.static.add(t.child_frame_id)

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)


def rate(stamps):
    # From the first to the last message, so late discovery does not lower the result
    return (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 else 0.0


def rate_check(probe, key, ok, low, name):
    hz = rate(probe.stamps.get(key, []))
    level = 'PASS' if hz >= ok else 'WARN' if hz >= low else 'FAIL'
    out(level, name, f'{hz:.1f} Hz (want >= {ok})')
    return hz > 0


def check_static(probe, seconds):
    for stamps in probe.stamps.values():
        stamps.clear()
    probe.spin_for(seconds)

    if rate_check(probe, 'camera', 25, 15, 'camera'):
        img = deserialize_message(probe.last['camera'][-1], Image)
        px = np.frombuffer(img.data, dtype=np.uint8)
        mean = float(px.mean()) if px.size else 0.0
        size_ok = px.size == img.height * img.step
        level = 'PASS' if size_ok and 10 < mean < 245 else 'WARN'
        out(level, 'camera image', f'{img.width}x{img.height} {img.encoding}, mean brightness '
            f'{mean:.0f}/255' + ('' if 10 < mean < 245 else ' (lens covered, too dark or too bright?)'))

    if os.environ.get('CRC_LIDAR', '1') == '1':
        if rate_check(probe, 'scan', 8, 5, 'lidar /scan'):
            r = np.asarray(probe.last['scan'][-1].ranges, dtype=np.float32)
            valid = np.isfinite(r)
            share = valid.mean()
            front = np.concatenate([r[-10:], r[:11]])
            front = front[np.isfinite(front)]
            ahead = f'{front.min():.2f} m' if front.size else 'nothing within range'
            out('PASS' if share >= 0.3 else 'WARN', 'lidar rays',
                f'{share * 100:.0f}% of {r.size} rays hit something; nearest ahead (+-10 deg): {ahead}')
    else:
        out('INFO', 'lidar /scan', 'off (CRC_LIDAR=0)')

    rate_check(probe, 'odom', 20, 10, 'odom')
    if probe.last['odom']:
        tw = probe.last['odom'][-1].twist.twist
        moving = abs(tw.linear.x) > 0.02 or abs(tw.angular.z) > 0.05
        out('WARN' if moving else 'PASS', 'odom at rest',
            f'v={tw.linear.x:+.3f} m/s w={tw.angular.z:+.3f} rad/s'
            + (' (car moving or a program is driving it)' if moving else ''))

    if rate_check(probe, 'imu', 20, 10, 'imu'):
        acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]
                        for m in probe.last['imu']])
        gyr = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]
                        for m in probe.last['imu']])
        a = acc.mean(axis=0)
        g = float(np.linalg.norm(a))
        flat = 8.8 < a[2] < 10.8 and g < 10.8
        out('PASS' if flat else 'WARN', 'imu gravity',
            f'a=({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}) m/s^2, |a|={g:.2f}'
            + ('' if flat else ' (want about +9.8 on z with the car standing flat)'))
        w = float(np.abs(gyr.mean(axis=0)).max())
        out('PASS' if w < 0.05 else 'WARN', 'imu gyro bias', f'{w:.3f} rad/s at rest (want < 0.05)')

    if probe.last['voltage']:
        v = float(np.mean([m.data for m in probe.last['voltage']]))
        level = 'PASS' if v >= BATT_OK else 'WARN' if v >= BATT_LOW else 'FAIL'
        out(level, 'battery', f'{v:.2f} V' + ('' if v >= BATT_OK else ' - charge the battery'))
    else:
        out('FAIL', 'battery', 'no /voltage (Yahboom board not answering?)')

    missing = [f for f in STATIC_FRAMES if f not in probe.static]
    out('FAIL' if missing else 'PASS', 'tf static',
        'missing ' + ', '.join(missing) if missing else f'{len(probe.static)} fixed frames')
    has_odom = ('odom', 'base_footprint') in probe.dynamic
    out('PASS' if has_odom else 'FAIL', 'tf odom', 'odom -> base_footprint'
        + ('' if has_odom else ' not published'))

    if 'signs' in probe.stamps:
        if not probe.stamps['signs']:
            out('WARN', 'signs', 'no /signs (no engine at CRC_YOLO_ENGINE? see run_car.sh bringup-logs)')
        else:
            rate_check(probe, 'signs', 8, 4, 'signs (yolo)')
    else:
        out('INFO', 'signs', 'sign detector off (CRC_YOLO=0)')


def check_motion(probe):
    node = probe.node
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    probe.spin_for(0.5)
    others = node.count_publishers('/cmd_vel') - 1
    if others > 0:
        out('FAIL', 'motion', f'{others} other /cmd_vel publisher(s) running; stop them first')
        return
    r = env_float('CRC_WHEEL_RADIUS', 0.0325)
    half = env_float('CRC_WHEEL_SEPARATION', 0.298) / 2.0
    # (name, vx m/s, wz rad/s): expected wheel speeds in rad/s follow from the kinematics
    steps = [('motion forward', 0.15, 0.0), ('motion spin left', 0.0, 1.5)]
    try:
        for name, vx, wz in steps:
            want = ((vx - wz * half) / r, (vx + wz * half) / r)
            probe.wheel.clear()
            probe.recording = True
            msg = Twist()
            msg.linear.x, msg.angular.z = vx, wz
            end = time.monotonic() + 2.5
            while time.monotonic() < end:
                pub.publish(msg)
                probe.spin_for(0.05)
            probe.recording = False
            tail = [s for s in probe.wheel if s[0] > end - 1.0]
            if not tail:
                out('FAIL', name, 'no /wheel_speed')
                continue
            got = (float(np.mean([s[1] for s in tail])), float(np.mean([s[2] for s in tail])))
            # Lifted wheels have no load, so with the floor-tuned feedforward they run faster
            # than commanded: this step checks direction and response only. Speed accuracy is
            # tuned on the floor with spin_test.py.
            signs_ok = all(math.copysign(1, g) == math.copysign(1, w) for g, w in zip(got, want))
            moving = all(abs(g) > 0.3 * abs(w) for g, w in zip(got, want))
            level = 'PASS' if signs_ok and moving else 'FAIL'
            hint = ('' if signs_ok else ' - wrong direction: check CRC_*_MOTOR / CRC_*_INVERT') + \
                   ('' if moving else ' - a wheel barely turns: motor cable, CRC_*_MOTOR, battery')
            out(level, name, f'left {got[0]:+.1f} (cmd {want[0]:+.1f}), right {got[1]:+.1f} '
                f'(cmd {want[1]:+.1f}) rad/s, unloaded{hint}')
    finally:
        stop = Twist()
        for _ in range(10):
            pub.publish(stop)
            probe.spin_for(0.05)
    # Free wheels coast for a moment: judge the mean speed 1.5-2 s after the stop command
    probe.wheel.clear()
    probe.recording = True
    probe.spin_for(1.5)
    probe.recording = False
    tail = [s for s in probe.wheel if s[0] > time.monotonic() - 0.5]
    w = max(abs(float(np.mean([s[1] for s in tail]))), abs(float(np.mean([s[2] for s in tail])))) \
        if tail else 0.0
    out('PASS' if w < 0.3 else 'FAIL', 'motion stop', f'{w:.2f} rad/s 1.5-2 s after stopping')


def main():
    rclpy.init()
    node = rclpy.create_node('car_check')
    probe = Probe(node)
    probe.spin_for(1.0)  # discovery and latched /tf_static
    check_static(probe, LISTEN)
    if '--motion' in sys.argv:
        check_motion(probe)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
