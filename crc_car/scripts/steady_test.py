#!/usr/bin/env python3
"""Constant-speed straight run: logs wheel speeds, PWM and IMU at 25 Hz and reports how even
the motion is. Usage: python3 steady_test.py [speed_m_s] [seconds]  (default 0.2 m/s, 4 s)"""
import math
import statistics as st
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, Int32MultiArray

V = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
SKIP = 1.0   # s of start-up excluded from the statistics

rclpy.init()
n = rclpy.create_node('steady_test')
last = {}
log = []
n.create_subscription(Float32MultiArray, '/wheel_speed', lambda m: last.__setitem__('spd', tuple(m.data)), 50)
n.create_subscription(Float32MultiArray, '/wheel_ref', lambda m: last.__setitem__('ref', tuple(m.data)), 50)
n.create_subscription(Int32MultiArray, '/wheel_pwm', lambda m: last.__setitem__('pwm', tuple(m.data)), 50)
n.create_subscription(Imu, '/imu', lambda m: last.__setitem__('imu', (m.linear_acceleration.x, m.angular_velocity.z)), 100)
pub = n.create_publisher(Twist, '/cmd_vel', 10)
t0 = time.time()
while (len(last) < 4 or pub.get_subscription_count() == 0) and time.time() - t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.05)

cmd = Twist()
cmd.linear.x = V
t0 = time.time()
next_log = t0
try:
    while time.time() - t0 < DUR:
        pub.publish(cmd)
        rclpy.spin_once(n, timeout_sec=0.01)
        if time.time() >= next_log:
            next_log += 0.04
            log.append((time.time() - t0,) + last['spd'] + last['pwm'] + last['imu'] + last['ref'])
finally:
    t1 = time.time()
    while time.time() - t1 < 1.0:
        pub.publish(Twist())
        rclpy.spin_once(n, timeout_sec=0.02)

R = 0.0331
steady = [r for r in log if r[0] >= SKIP]
print('time series (every 0.2 s): t | wheel L R rad/s | pwm L R | v m/s')
for r in log[::5]:
    print('  %4.1f | %5.2f %5.2f | %3d %3d | %.3f' % (r[0], r[1], r[2], r[3], r[4], R * (r[1] + r[2]) / 2))
ref = st.mean(r[7] for r in steady)
for i, name in ((1, 'left '), (2, 'right')):
    xs = [r[i] for r in steady]
    m = st.mean(xs)
    dev = [x - m for x in xs]
    crossings = sum(1 for a, b in zip(dev, dev[1:]) if a * b < 0)
    span = steady[-1][0] - steady[0][0]
    print('%s wheel: mean %.2f (ref %.2f, %+.0f %%)  ripple %.2f rad/s (%.0f %%)  min %.2f max %.2f  ~%.1f Hz' % (
        name, m, ref, 100 * (m - ref) / ref, st.pstdev(xs), 100 * st.pstdev(xs) / m, min(xs), max(xs),
        crossings / 2 / span if span > 0 else 0))
v = [R * (r[1] + r[2]) / 2 for r in steady]
ax = [r[5] for r in steady]
gz = [r[6] for r in steady]
print('forward speed: mean %.3f m/s (cmd %.2f), ripple %.3f m/s (%.0f %%)' % (st.mean(v), V, st.pstdev(v), 100 * st.pstdev(v) / st.mean(v)))
print('IMU: forward accel ripple %.2f m/s^2, yaw rate mean %+.3f ripple %.3f rad/s' % (st.pstdev(ax), st.mean(gz), st.pstdev(gz)))
