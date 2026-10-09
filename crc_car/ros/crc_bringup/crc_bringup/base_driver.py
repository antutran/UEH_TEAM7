#!/usr/bin/env python3
"""Base driver for the UEH CRC 2026 car: Yahboom ROS Robot Control Board V3.0 driving a
two-wheel differential base, with a per-wheel speed PID closed on the encoders.

Matches the topics and frames of the Gazebo simulation (TurtleBot3 Waffle):

Subscribes: /cmd_vel        geometry_msgs/Twist   (linear.x m/s, angular.z rad/s)
Publishes : /odom           nav_msgs/Odometry     wheel odometry, odom -> base_footprint
            /imu            sensor_msgs/Imu       acceleration + angular velocity (no orientation)
            /tf             odom -> base_footprint (publish_tf:=true)
            /voltage        std_msgs/Float32      battery voltage (V)
            /wheel_ticks    std_msgs/Int32MultiArray    [left, right] accumulated encoder counts
            /wheel_speed    std_msgs/Float32MultiArray  [left, right] measured rad/s
            /wheel_ref      std_msgs/Float32MultiArray  [left, right] commanded rad/s
            /wheel_pwm      std_msgs/Int32MultiArray    [left, right] PWM % sent to the board

Per-wheel control: PWM = feedforward(w_ref) + kp*e + ki*integral(e) + kd*d(meas)/dt,
with e = w_ref - w_meas. pid_enable:=false leaves only the feedforward (open loop).
"""
import fcntl
import math
import os
import termios
import threading
import time
from collections import deque

import rclpy
import serial
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray, Int32MultiArray
from tf2_ros import TransformBroadcaster

from crc_bringup.Rosmaster_Lib import Rosmaster


class WheelPid:
    """Speed PID for one wheel: rad/s in, PWM % out, with conditional integration (anti-windup)."""

    def __init__(self, kp, ki, kd, ff_gain, ff_offset, max_pwm, i_band, brake_pwm):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ff_gain, self.ff_offset, self.max_pwm = ff_gain, ff_offset, max_pwm
        # Integrate only while |e| < i_band, so the feedforward handles acceleration without overshoot
        self.i_band = i_band
        # On a stop command, brake actively (reverse P term) up to this PWM %
        self.brake_pwm = brake_pwm
        self.reset()

    def reset(self):
        self.integ = 0.0
        self.prev_meas = None
        self.braking = False

    def feedforward(self, w_ref):
        if abs(w_ref) < 1e-6:
            return 0.0
        return math.copysign(self.ff_offset + self.ff_gain * abs(w_ref), w_ref)

    def update(self, w_ref, w_meas, dt, use_pid=True):
        if abs(w_ref) < 1e-6:
            # Stop command: brake once until the wheel is nearly still, then cut PWM (no chatter)
            was_running = self.prev_meas is not None
            self.integ, self.prev_meas = 0.0, None
            if was_running and w_meas is not None:
                self.braking = math.copysign(1.0, w_meas)  # direction of rotation when braking starts
            if not use_pid or w_meas is None or w_meas * self.braking < 0.5:  # nearly stopped or past zero
                self.braking = False
            if not self.braking:
                return 0
            u = max(-self.brake_pwm, min(self.brake_pwm, -self.kp * w_meas))
            return int(round(u))
        self.braking = False
        u = self.feedforward(w_ref)
        if use_pid and w_meas is not None:
            e = w_ref - w_meas
            d = 0.0
            if self.prev_meas is not None and dt > 0:
                d = -(w_meas - self.prev_meas) / dt  # derivative on measurement: no kick on setpoint change
            self.prev_meas = w_meas
            u_try = u + self.kp * e + self.ki * (self.integ + e * dt) + self.kd * d
            saturated_same_dir = abs(u_try) > self.max_pwm and (u_try > 0) == (e > 0)
            if abs(e) < self.i_band and not saturated_same_dir:
                self.integ += e * dt
            u += self.kp * e + self.ki * self.integ + self.kd * d
        return int(round(max(-self.max_pwm, min(self.max_pwm, u))))


class BaseDriver(Node):
    def __init__(self):
        super().__init__('base_driver')
        p = self.declare_parameter
        # Numeric parameters accept ints too (e.g. counts_per_rev:=1320 from an env file)
        fp = lambda name, default: float(  # noqa: E731
            p(name, default, ParameterDescriptor(dynamic_typing=True)).value)
        port = p('board_port', '/dev/myserial').value
        self.left_motor = p('left_motor', 1).value             # board port Motor 1..4
        self.right_motor = p('right_motor', 4).value
        left_invert = p('left_invert', False).value            # reverse motor direction (mirrored mounting)
        right_invert = p('right_invert', False).value
        left_enc_invert = p('left_enc_invert', False).value    # reverse encoder sign (A/B swapped)
        right_enc_invert = p('right_enc_invert', False).value
        self.wheel_radius = fp('wheel_radius', 0.0325)          # 65 mm wheels
        self.wheel_separation = fp('wheel_separation', 0.298)   # left-to-right wheel centre distance
        # JGB37-520 1:30 (333 rpm): 11 PPR x 4 (quadrature) x 30 = 1320 counts per wheel revolution
        self.counts_per_rev = fp('counts_per_rev', 1320.0)
        self.max_pwm = p('max_pwm', 100).value
        self.max_linear = fp('max_linear', 1.0)           # m/s, clamp on /cmd_vel
        self.max_angular = fp('max_angular', 6.0)         # rad/s, clamp on /cmd_vel
        self.cmd_timeout = fp('cmd_timeout', 0.5)         # stop if /cmd_vel goes quiet this long
        # Acceleration limits on the commanded motion. A speed step makes the PI controller
        # overshoot (+39 %) and then undershoot (-30 %) for ~2 s, because the firmware's dead-zone
        # offset makes the start-up PWM jump; ramping the reference keeps the error small.
        self.max_accel = fp('max_accel', 0.8)            # m/s^2 when speeding up
        self.max_decel = fp('max_decel', 1.5)            # m/s^2 when slowing down
        self.max_ang_accel = fp('max_ang_accel', 6.0)    # rad/s^2
        # Wheel speed = count difference between two encoder frames >= speed_window apart
        # (the board sends one every 40 ms)
        # 0.08 s (two frames) cut the start-up overshoot from ~33 % to ~21 % versus 0.12 s
        self.speed_window = fp('speed_window', 0.08)
        self.use_pid = p('pid_enable', True).value
        kp = fp('kp', 3.0)            # PWM % per rad/s
        ki = fp('ki', 15.0)           # PWM % per rad
        kd = fp('kd', 0.0)            # PWM % per rad/s^2
        # Feedforward: PWM ~ ff_offset + ff_gain*|w| per wheel (motors are not identical).
        # 333 rpm motor: about 100 % / 34.9 rad/s no-load, so ~3 % per rad/s; the PID trims the rest.
        ff_gain = fp('ff_gain', 3.0)
        ff_gains = (fp('ff_gain_left', ff_gain), fp('ff_gain_right', ff_gain))
        ff_offset = fp('ff_offset', 0.0)
        # Integrate only while |e| < i_band (rad/s). Must stay wide: the board's firmware adds a
        # dead-zone offset to every motor command, so the feedforward is never exact and a narrow
        # band leaves a permanent speed error (measured 4 Oct 2026: 43% of the commanded turn rate).
        i_band = fp('i_band', 50.0)
        brake_pwm = p('brake_pwm', 20).value   # %, 0 = coast to a stop
        rate = fp('rate', 25.0)
        # The board reports acc z ~ -9.8 when level: its IMU z axis points down. REP-103 wants z up
        # (turning left = positive angular_velocity.z), so rotate 180 deg about x: (x, y, z) -> (x, -y, -z)
        self.imu_flip = p('imu_flip', True).value
        self.odom_frame = p('odom_frame', 'odom').value
        self.base_frame = p('base_frame', 'base_footprint').value
        self.imu_frame = p('imu_frame', 'imu_link').value
        self.publish_tf = p('publish_tf', True).value

        self.cmd_sign = (-1 if left_invert else 1, -1 if right_invert else 1)
        self.enc_sign = (self.cmd_sign[0] * (-1 if left_enc_invert else 1),
                         self.cmd_sign[1] * (-1 if right_enc_invert else 1))
        self.pids = [WheelPid(kp, ki, kd, g, ff_offset, self.max_pwm, i_band, brake_pwm) for g in ff_gains]

        # The board can enumerate late (seen ~20 s after boot): open it from a retry timer so the
        # process (shared with the camera and scan nodes) never blocks.
        self.port = port
        self.bot = None
        self.frames = deque(maxlen=50)  # (t_monotonic, (enc_m1..m4)) stamped on arrival
        self.frames_lock = threading.Lock()
        self.open_warned = 0.0
        self.open_timer = self.create_timer(0.5, self.try_open_board)

        self.cmd = (0.0, 0.0)
        self.v_ramp = 0.0
        self.w_ramp = 0.0
        self.last_cmd_time = self.get_clock().now()
        self.last_update = time.monotonic()
        self.prev_ticks = None
        self.stale_warned = False
        self.x = self.y = self.yaw = 0.0

        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.pub_voltage = self.create_publisher(Float32, 'voltage', 10)
        self.pub_imu = self.create_publisher(Imu, 'imu', 10)
        self.pub_ticks = self.create_publisher(Int32MultiArray, 'wheel_ticks', 10)
        self.pub_speed = self.create_publisher(Float32MultiArray, 'wheel_speed', 10)
        self.pub_ref = self.create_publisher(Float32MultiArray, 'wheel_ref', 10)
        self.pub_pwm = self.create_publisher(Int32MultiArray, 'wheel_pwm', 10)
        self.pub_odom = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.create_timer(1.0 / rate, self.update)
        # Controller gains can be tuned live: ros2 param set /base_driver kp 2.0
        self.add_on_set_parameters_callback(self.on_set_parameters)
        mode = f'PID kp={kp} ki={ki} kd={kd}' if self.use_pid else 'open loop'
        self.get_logger().info(
            f'Board {port}: left=Motor {self.left_motor}, right=Motor {self.right_motor}, '
            f'separation={self.wheel_separation} m, counts/rev={self.counts_per_rev}, '
            f'max_pwm={self.max_pwm}%, {mode}, ff left/right={ff_gains[0]}/{ff_gains[1]}*|w|. '
            'Waiting for /cmd_vel...')

    def try_open_board(self):
        """Open the board's serial port exclusively once it exists and is free."""
        port = self.port
        if os.path.exists(port):
            try:
                serial.Serial(port).close()  # probe first: a held port gives a clear error
                bot = Rosmaster(car_type=1, com=port)
                fcntl.ioctl(bot.ser.fileno(), termios.TIOCEXCL)  # keep the port exclusive
            except (serial.SerialException, OSError) as e:
                reason = f'cannot open {port}: {e} (is another program using it?)'
            else:
                # Timestamp every encoder frame as it arrives (in Rosmaster_Lib's serial thread),
                # so the speed uses the real spacing between frames instead of the timer period.
                parse = bot._Rosmaster__parse_data

                def parse_and_stamp(ext_type, ext_data):
                    parse(ext_type, ext_data)
                    if ext_type == bot.FUNC_REPORT_ENCODER:
                        with self.frames_lock:
                            self.frames.append((time.monotonic(), bot.get_motor_encoder()))
                bot._Rosmaster__parse_data = parse_and_stamp
                bot.create_receive_threading()
                bot.set_beep(50)
                self.bot = bot
                self.open_timer.cancel()
                self.get_logger().info(f'Yahboom board connected on {port}')
                return
        else:
            reason = f'{port} not present (board unplugged or still enumerating)'
        if time.monotonic() - self.open_warned > 5.0:
            self.get_logger().warn(f'Waiting for the Yahboom board: {reason}')
            self.open_warned = time.monotonic()

    TUNABLE = {
        'kp': ('kp', None), 'ki': ('ki', None), 'kd': ('kd', None),
        'ff_offset': ('ff_offset', None), 'i_band': ('i_band', None),
        'brake_pwm': ('brake_pwm', None), 'max_pwm': ('max_pwm', None),
        'max_accel': ('max_accel', 'node'), 'max_decel': ('max_decel', 'node'),
        'max_ang_accel': ('max_ang_accel', 'node'),
        'speed_window': ('speed_window', 'node'),
        'ff_gain_left': ('ff_gain', 0), 'ff_gain_right': ('ff_gain', 1),
    }

    def on_set_parameters(self, params):
        """Apply controller gain changes at run time (other parameters need a restart)."""
        for param in params:
            if param.name not in self.TUNABLE:
                continue
            attr, wheel = self.TUNABLE[param.name]
            value = float(param.value)
            if wheel == 'node':
                setattr(self, attr, value)
                self.get_logger().info(f'{param.name} -> {value}')
                continue
            for i, pid in enumerate(self.pids):
                if wheel is None or wheel == i:
                    setattr(pid, attr, value)
                    pid.integ = 0.0
            if param.name == 'max_pwm':
                self.max_pwm = value
            self.get_logger().info(f'{param.name} -> {value}')
        return SetParametersResult(successful=True)

    def on_cmd_vel(self, msg):
        v = max(-self.max_linear, min(self.max_linear, msg.linear.x))
        wz = max(-self.max_angular, min(self.max_angular, msg.angular.z))
        self.cmd = (v, wz)
        self.last_cmd_time = self.get_clock().now()

    def ramp(self, v, wz, dt):
        """Move the commanded (v, wz) towards the target within the acceleration limits."""
        dt = min(dt, 0.1)  # a stalled timer must not allow a jump
        speeding_up = abs(v) > abs(self.v_ramp) and v * self.v_ramp >= 0.0
        dv_max = (self.max_accel if speeding_up else self.max_decel) * dt
        self.v_ramp += max(-dv_max, min(dv_max, v - self.v_ramp))
        dw_max = self.max_ang_accel * dt
        self.w_ramp += max(-dw_max, min(dw_max, wz - self.w_ramp))
        return self.v_ramp, self.w_ramp

    def wheel_state(self):
        """-> (ticks [left, right], speed [left, right] in rad/s, or None when encoder data is missing)."""
        with self.frames_lock:
            frames = list(self.frames)
        if not frames:
            return None, None
        t1, enc1 = frames[-1]
        pick = lambda enc: (self.enc_sign[0] * enc[self.left_motor - 1],  # noqa: E731
                            self.enc_sign[1] * enc[self.right_motor - 1])
        ticks = pick(enc1)
        if time.monotonic() - t1 > 0.2:  # board stopped sending encoder frames (USB drop, power loss...)
            return ticks, None
        older = [f for f in frames if t1 - f[0] >= self.speed_window]
        if not older:
            return ticks, None
        t0, enc0 = older[-1]
        ticks0 = pick(enc0)
        rad_per_tick = 2.0 * math.pi / self.counts_per_rev
        speed = [(ticks[i] - ticks0[i]) * rad_per_tick / (t1 - t0) for i in range(2)]
        return ticks, speed

    def update(self):
        if self.bot is None:
            return
        now = self.get_clock().now()
        t = time.monotonic()
        dt, self.last_update = t - self.last_update, t
        v, wz = self.cmd
        if (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout:
            v, wz = 0.0, 0.0
        v, wz = self.ramp(v, wz, dt)
        half = self.wheel_separation / 2.0
        ref = [(v - wz * half) / self.wheel_radius, (v + wz * half) / self.wheel_radius]

        ticks, speed = self.wheel_state()
        if self.use_pid and speed is None and any(abs(r) > 1e-6 for r in ref):
            # Without a speed measurement the integrator would wind up to full PWM: stop instead
            if not self.stale_warned:
                self.get_logger().warn('No encoder data from the board, stopping the motors')
                self.stale_warned = True
            ref = [0.0, 0.0]
        elif speed is not None:
            self.stale_warned = False

        out = [self.pids[i].update(ref[i], speed[i] if speed else None, dt, self.use_pid)
               for i in range(2)]
        pwm = [0, 0, 0, 0]
        pwm[self.left_motor - 1] = self.cmd_sign[0] * out[0]
        pwm[self.right_motor - 1] = self.cmd_sign[1] * out[1]
        self.bot.set_motor(*pwm)

        self.pub_ref.publish(Float32MultiArray(data=[float(r) for r in ref]))
        self.pub_pwm.publish(Int32MultiArray(data=out))
        self.publish_state(now, ticks, speed)

    def publish_state(self, now, ticks, speed):
        stamp = now.to_msg()
        self.pub_voltage.publish(Float32(data=float(self.bot.get_battery_voltage())))

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = self.imu_frame
        imu.orientation_covariance[0] = -1.0  # orientation not provided
        ax, ay, az = self.bot.get_accelerometer_data()
        gx, gy, gz = self.bot.get_gyroscope_data()
        if self.imu_flip:
            ay, az, gy, gz = -ay, -az, -gy, -gz
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = ax, ay, az
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gx, gy, gz
        self.pub_imu.publish(imu)

        if ticks is None:
            return
        self.pub_ticks.publish(Int32MultiArray(data=list(ticks)))

        rad_per_tick = 2.0 * math.pi / self.counts_per_rev
        if self.prev_ticks is not None:
            # Integrate the pose step by step, independent of the encoder frame timing
            d_left = (ticks[0] - self.prev_ticks[0]) * rad_per_tick
            d_right = (ticks[1] - self.prev_ticks[1]) * rad_per_tick
            ds = self.wheel_radius * (d_left + d_right) / 2.0
            dyaw = self.wheel_radius * (d_right - d_left) / self.wheel_separation
            self.x += ds * math.cos(self.yaw + dyaw / 2.0)
            self.y += ds * math.sin(self.yaw + dyaw / 2.0)
            self.yaw += dyaw
        self.prev_ticks = ticks
        if speed is None:
            return
        w_left, w_right = speed
        self.pub_speed.publish(Float32MultiArray(data=[float(w_left), float(w_right)]))

        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.wheel_radius * (w_left + w_right) / 2.0
        odom.twist.twist.angular.z = self.wheel_radius * (w_right - w_left) / self.wheel_separation
        self.pub_odom.publish(odom)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

    def stop_motors(self):
        if self.bot is None:
            return
        try:
            self.bot.set_motor(0, 0, 0, 0)
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass


def main():
    rclpy.init()
    node = BaseDriver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_motors()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
