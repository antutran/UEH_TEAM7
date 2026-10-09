"""UEH CRC 2026 car bring-up.

Starts two processes:
  * crc_bringup/bringup   base driver + camera bridge + scan adapter + static TF (one process)
  * sllidar_ros2          RPLIDAR C1 driver, publishing /scan_raw for the scan adapter

Per-car settings come from environment variables (the bring-up container loads ~/crc_car.env):
  CRC_LEFT_MOTOR, CRC_RIGHT_MOTOR       Yahboom board motor ports (default 1 and 4)
  CRC_LEFT_INVERT, CRC_RIGHT_INVERT     1 to reverse a wheel (default 0)
  CRC_WHEEL_SEPARATION                  wheel centre distance in metres (default 0.298)
  CRC_WHEEL_RADIUS                      effective wheel radius in metres (default 0.0325)
  CRC_COUNTS_PER_REV                    encoder counts per wheel revolution (default 1320)
  CRC_MAX_PWM                           PWM limit in percent (default 100)
  CRC_KP, CRC_KI, CRC_I_BAND            wheel speed PI gains (default 3.0, 15.0, 50)
  CRC_FF_GAIN_LEFT, CRC_FF_GAIN_RIGHT   feedforward PWM % per rad/s (default 3.0)
  CRC_SPEED_WINDOW                      wheel speed measurement window in s (default 0.08)
  CRC_MAX_ACCEL, CRC_MAX_DECEL          linear acceleration limits in m/s^2 (default 0.8, 1.5)
  CRC_MAX_ANG_ACCEL                     angular acceleration limit in rad/s^2 (default 6.0)
  CRC_SCAN_YAW_OFFSET_DEG               LiDAR mounting rotation (default 0)
  CRC_CAMERA_HFOV_DEG                   camera horizontal field of view (default 62.2)
  CAMERA_WIDTH, CAMERA_HEIGHT           must match camera_stream.sh (default 640x480)
  CRC_LIDAR                             1 to start the LiDAR driver, 0 to skip it (default 1)
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node


def env(name, default, cast=str):
    value = os.environ.get(name, '')
    return cast(value) if value != '' else default


def generate_launch_description():
    params = {
        'board_port': '/dev/myserial',
        'left_motor': env('CRC_LEFT_MOTOR', 1, int),
        'right_motor': env('CRC_RIGHT_MOTOR', 4, int),
        'left_invert': env('CRC_LEFT_INVERT', '0') in ('1', 'true', 'True', 'yes'),
        'right_invert': env('CRC_RIGHT_INVERT', '0') in ('1', 'true', 'True', 'yes'),
        'wheel_separation': env('CRC_WHEEL_SEPARATION', 0.298, float),
        'wheel_radius': env('CRC_WHEEL_RADIUS', 0.0325, float),
        'counts_per_rev': env('CRC_COUNTS_PER_REV', 1320.0, float),
        'max_pwm': env('CRC_MAX_PWM', 100, int),
        'kp': env('CRC_KP', 3.0, float),
        'ki': env('CRC_KI', 15.0, float),
        'i_band': env('CRC_I_BAND', 50.0, float),
        'ff_gain_left': env('CRC_FF_GAIN_LEFT', 3.0, float),
        'ff_gain_right': env('CRC_FF_GAIN_RIGHT', 3.0, float),
        'speed_window': env('CRC_SPEED_WINDOW', 0.08, float),
        'max_accel': env('CRC_MAX_ACCEL', 0.8, float),
        'max_decel': env('CRC_MAX_DECEL', 1.5, float),
        'max_ang_accel': env('CRC_MAX_ANG_ACCEL', 6.0, float),
        'camera_width': env('CAMERA_WIDTH', 640, int),
        'camera_height': env('CAMERA_HEIGHT', 480, int),
        'camera_hfov_deg': env('CRC_CAMERA_HFOV_DEG', 62.2, float),
        'scan_yaw_offset_deg': env('CRC_SCAN_YAW_OFFSET_DEG', 0.0, float),
    }
    actions = [
        # No `name=` here: it would rename every node inside the shared process to the same name
        Node(package='crc_bringup', executable='bringup',
             parameters=[params], output='screen', respawn=True, respawn_delay=2.0),
    ]
    if env('CRC_LIDAR', 1, int):
        actions.append(Node(
            package='sllidar_ros2', executable='sllidar_node', name='sllidar_node',
            parameters=[{
                'channel_type': 'serial',
                'serial_port': '/dev/rplidar',
                'serial_baudrate': 460800,
                'frame_id': 'base_scan',
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'Standard',
            }],
            remappings=[('scan', 'scan_raw')],
            output='screen', respawn=True, respawn_delay=3.0))
    return LaunchDescription(actions)
