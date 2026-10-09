#!/usr/bin/env python3
"""Compare wheel speed controller gains by smoothness: spin in place at +/-W rad/s for each
(kp, ki) pair and report the mean and ripple (stdev) of the IMU yaw rate and wheel speeds.
Gains are set live on /base_driver and restored to the first pair at the end."""
import statistics as st
import subprocess
import time

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

W = 1.0
HOLD, MEASURE = 3.0, 2.0
GAINS = [(3.0, 15.0), (2.0, 10.0), (1.5, 8.0), (1.0, 6.0), (2.0, 5.0)]


def set_param(name, value):
    subprocess.run(['ros2', 'param', 'set', '/base_driver', name, str(value)],
                   check=True, stdout=subprocess.DEVNULL)


rclpy.init()
n = rclpy.create_node('smooth_sweep')
buf = {'gz': [], 'spd': []}
rec = [False]
n.create_subscription(Imu, '/imu', lambda m: buf['gz'].append(m.angular_velocity.z) if rec[0] else None, 100)
n.create_subscription(Float32MultiArray, '/wheel_speed', lambda m: buf['spd'].append(tuple(m.data)) if rec[0] else None, 50)
pub = n.create_publisher(Twist, '/cmd_vel', 10)
t0 = time.time()
while pub.get_subscription_count() == 0 and time.time() - t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.05)

print('  kp    ki  | dir | IMU yaw rate mean (ripple)   | wheel L ripple | wheel R ripple')
try:
    for kp, ki in GAINS:
        set_param('kp', kp)
        set_param('ki', ki)
        for w in (W, -W):
            cmd = Twist()
            cmd.angular.z = w
            buf['gz'].clear()
            buf['spd'].clear()
            t0 = time.time()
            while time.time() - t0 < HOLD:
                rec[0] = time.time() - t0 > HOLD - MEASURE
                pub.publish(cmd)
                rclpy.spin_once(n, timeout_sec=0.02)
            rec[0] = False
            gz = buf['gz']
            print(' %4.1f %5.1f  | %+2.0f  | %+5.2f rad/s (%.3f = %4.1f %%) |   %.2f rad/s   |   %.2f rad/s' % (
                kp, ki, w, st.mean(gz), st.pstdev(gz), 100 * st.pstdev(gz) / abs(W),
                st.pstdev(s[0] for s in buf['spd']), st.pstdev(s[1] for s in buf['spd'])), flush=True)
            t0 = time.time()
            while time.time() - t0 < 0.8:
                pub.publish(Twist())
                rclpy.spin_once(n, timeout_sec=0.02)
finally:
    for _ in range(5):
        pub.publish(Twist())
        time.sleep(0.05)
    set_param('kp', GAINS[0][0])
    set_param('ki', GAINS[0][1])
