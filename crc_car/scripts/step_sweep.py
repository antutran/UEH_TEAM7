#!/usr/bin/env python3
"""Step response from standstill, spinning in place, for several controller settings.
Scores each setting by the true turn rate from the IMU: overshoot, settling time (to within
+/-15 %) and ripple once settled. Settings are applied live and the first one is restored."""
import statistics as st
import subprocess
import time

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

W, HOLD = 1.0, 3.0
SETTINGS = [  # (i_band, ki, kp, kd, speed_window)
    (3.0, 10.0, 2.0, 0.0, 0.12),
    (3.0, 10.0, 2.0, 0.0, 0.08),
    (3.0, 10.0, 2.0, 0.15, 0.08),
    (3.0, 10.0, 2.0, 0.3, 0.08),
    (3.0, 6.0, 1.5, 0.3, 0.08),
]
NAMES = ('i_band', 'ki', 'kp', 'kd', 'speed_window')


def setp(name, value):
    subprocess.run(['ros2', 'param', 'set', '/base_driver', name, str(value)], check=True, stdout=subprocess.DEVNULL)


rclpy.init()
n = rclpy.create_node('step_sweep')
trace = []
rec = [None]
n.create_subscription(Imu, '/imu', lambda m: trace.append((time.time() - rec[0], m.angular_velocity.z)) if rec[0] else None, 100)
pub = n.create_publisher(Twist, '/cmd_vel', 10)
t0 = time.time()
while pub.get_subscription_count() == 0 and time.time() - t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.05)

print(' i_band   ki   kp    kd  win | dir | peak (overshoot) | settled at | ripple after settling')
try:
    for setting in SETTINGS:
        for name, value in zip(NAMES, setting):
            setp(name, value)
        i_band, ki, kp, kd, win = setting
        for w in (W, -W):
            cmd = Twist(); cmd.angular.z = w
            trace.clear(); rec[0] = time.time()
            while time.time() - rec[0] < HOLD:
                pub.publish(cmd); rclpy.spin_once(n, timeout_sec=0.01)
            rec[0] = None
            ys = [(t, y / w) for t, y in trace]          # normalised: 1.0 = on target
            peak = max(y for _, y in ys)
            settled = next((t for t, _ in ys if all(abs(y2 - 1) < 0.15 for t2, y2 in ys if t2 >= t)), None)
            tail = [y for t, y in ys if settled is not None and t >= settled] or [y for t, y in ys if t > 1.5]
            print('  %4.1f %5.1f %4.1f %5.2f %4.2f | %+2.0f  |  %.2f (%+3.0f %%)   |  %s   | %.3f (%.0f %%)' % (
                i_band, ki, kp, kd, win, w, peak * abs(w), 100 * (peak - 1),
                '%.2f s' % settled if settled is not None else ' never', st.pstdev(tail) * abs(w), 100 * st.pstdev(tail)), flush=True)
            t1 = time.time()
            while time.time() - t1 < 1.0:
                pub.publish(Twist()); rclpy.spin_once(n, timeout_sec=0.02)
finally:
    for _ in range(5):
        pub.publish(Twist()); time.sleep(0.05)
    for name, value in zip(NAMES, SETTINGS[0]):
        setp(name, value)
