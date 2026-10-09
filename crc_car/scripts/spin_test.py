#!/usr/bin/env python3
"""Spin-in-place test for tuning the wheel speed controllers on the floor.
For each yaw rate: command it for HOLD s, then report over the last MEASURE s the wheel
reference, measured wheel speed (mean and ripple), PWM and the true yaw rate from the IMU.
Usage: python3 spin_test.py [w1 w2 ...]   (rad/s, default 0.5 1.0 2.0 -0.5 -1.0 -2.0)"""
import statistics as st
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, Int32MultiArray

HOLD, MEASURE = 3.0, 2.0
RATES = [float(x) for x in sys.argv[1:]] or [0.5, 1.0, 2.0, -0.5, -1.0, -2.0]

rclpy.init()
n = rclpy.create_node('spin_test')
buf = {'ref': [], 'spd': [], 'pwm': [], 'gz': []}
rec = [False]
add = lambda k: (lambda m: buf[k].append(m) if rec[0] else None)  # noqa: E731
n.create_subscription(Float32MultiArray, '/wheel_ref', add('ref'), 50)
n.create_subscription(Float32MultiArray, '/wheel_speed', add('spd'), 50)
n.create_subscription(Int32MultiArray, '/wheel_pwm', add('pwm'), 50)
n.create_subscription(Imu, '/imu', add('gz'), 100)
pub = n.create_publisher(Twist, '/cmd_vel', 10)
t0 = time.time()
while pub.get_subscription_count() == 0 and time.time() - t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.1)

print(' w_cmd | ref L/R rad/s | speed L/R mean (ripple)     | pwm L/R | IMU yaw rate')
try:
    for w in RATES:
        cmd = Twist()
        cmd.angular.z = w
        for k in buf:
            buf[k].clear()
        t0 = time.time()
        while time.time() - t0 < HOLD:
            rec[0] = time.time() - t0 > HOLD - MEASURE
            pub.publish(cmd)
            rclpy.spin_once(n, timeout_sec=0.02)
        rec[0] = False
        ref = [st.mean(m.data[i] for m in buf['ref']) for i in range(2)]
        spd = [[m.data[i] for m in buf['spd']] for i in range(2)]
        pwm = [st.mean(m.data[i] for m in buf['pwm']) for i in range(2)]
        gz = st.mean(m.angular_velocity.z for m in buf['gz'])
        print('%+5.1f | %+5.1f / %+5.1f | %+5.1f (%4.1f) / %+5.1f (%4.1f) | %+3.0f / %+3.0f | %+5.2f rad/s (%3.0f%% of cmd)' % (
            w, ref[0], ref[1], st.mean(spd[0]), st.pstdev(spd[0]), st.mean(spd[1]), st.pstdev(spd[1]),
            pwm[0], pwm[1], gz, 100.0 * gz / w), flush=True)
        stop_t = time.time()
        while time.time() - stop_t < 1.0:
            pub.publish(Twist())
            rclpy.spin_once(n, timeout_sec=0.02)
finally:
    for _ in range(5):
        pub.publish(Twist())
        time.sleep(0.05)
