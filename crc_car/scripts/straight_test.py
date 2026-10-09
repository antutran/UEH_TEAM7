#!/usr/bin/env python3
"""Straight-line speed steps for checking the wheel speed controllers on the floor.
Open loop on heading (angular.z = 0), so any drift shows how well the two wheels match.
Usage: python3 straight_test.py   (needs ~1.5 m of free floor in front of the car)"""
import math
import statistics as st
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

STEPS = [(0.10, 2.5), (0.15, 2.0), (0.20, 1.5), (0.30, 1.2)]   # (m/s, s)

rclpy.init()
n = rclpy.create_node('straight_test')
state = {'odom': None, 'yaw': 0.0, 't_imu': None}
seg = {'spd': [], 'ref': []}


def on_imu(m):
    t = time.monotonic()
    if state['t_imu'] is not None:
        state['yaw'] += m.angular_velocity.z * (t - state['t_imu'])
    state['t_imu'] = t


n.create_subscription(Imu, '/imu', on_imu, 100)
n.create_subscription(Odometry, '/odom', lambda m: state.__setitem__('odom', m), 20)
n.create_subscription(Float32MultiArray, '/wheel_speed', lambda m: seg['spd'].append(tuple(m.data)), 50)
n.create_subscription(Float32MultiArray, '/wheel_ref', lambda m: seg['ref'].append(tuple(m.data)), 50)
pub = n.create_publisher(Twist, '/cmd_vel', 10)
t0 = time.time()
while (state['odom'] is None or state['t_imu'] is None or pub.get_subscription_count() == 0) and time.time() - t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.05)


def pos():
    p = state['odom'].pose.pose.position
    return p.x, p.y


start = pos()
print('  v_cmd | wheel L / R rad/s (ref)      | odom v  | segment  | heading drift (IMU)')
try:
    for v, dur in STEPS:
        cmd = Twist()
        cmd.linear.x = v
        p0, yaw0 = pos(), state['yaw']
        seg['spd'].clear()
        seg['ref'].clear()
        t0 = time.time()
        while time.time() - t0 < dur:
            if time.time() - t0 > 0.5 and not seg.get('rec'):
                seg['spd'].clear()
                seg['ref'].clear()
                seg['rec'] = True
            pub.publish(cmd)
            rclpy.spin_once(n, timeout_sec=0.02)
        seg['rec'] = False
        p1 = pos()
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        L = st.mean(s[0] for s in seg['spd'])
        R = st.mean(s[1] for s in seg['spd'])
        ref = st.mean(r[0] for r in seg['ref'])
        v_odom = 0.0325 * (L + R) / 2
        print('  %.2f  | %5.2f / %5.2f (%5.2f)       | %.3f   | %.2f m   | %+5.1f deg' % (
            v, L, R, ref, v_odom, d, math.degrees(state['yaw'] - yaw0)), flush=True)
finally:
    t0 = time.time()
    while time.time() - t0 < 1.0:
        pub.publish(Twist())
        rclpy.spin_once(n, timeout_sec=0.02)
end = pos()
print('total odom distance %.2f m, total heading drift %+.1f deg' % (
    math.hypot(end[0] - start[0], end[1] - start[1]), math.degrees(state['yaw'])))
