"""Default settings of the race template (= the car). config/race_sim.yaml and
config/race_car.yaml override them; no ROS needed, so the tools can use this file too."""

DEFAULTS = {
    # Camera geometry (ground frame: x forward from the wheel axle, y left, metres)
    'cam_height': 0.12, 'cam_x': 0.10, 'cam_y': 0.0, 'cam_pitch_deg': 0.0,
    'camera_matrix': [],          # 9 numbers from tools/calib_intrinsics.py; [] = /camera/camera_info
    'dist_coeffs': [], 'fisheye': False,
    'birdseye_points': [],        # 8 numbers (u, v) x 4 from tools/calib_birdseye.py; [] = geometry
    'birdseye_ground': [],        # 8 numbers (x, y) x 4, the same points on the ground
    'be_x_min': 0.25, 'be_x_max': 0.95, 'be_half_width': 0.60, 'be_ppm': 150.0,
    # Lane
    'lane_width': 0.425,         # distance between the CENTRES of the two lines (m)
    'line_width': 0.025, 'min_contrast': 25,
    # Driving
    'max_speed': 0.25, 'min_speed': 0.10, 'lookahead': 0.45, 'curve_slowdown': 1.0,
    'max_turn': 2.5, 'lost_timeout': 0.5,
    'blind_distance': 1.0,        # lines lost (zebra, junction, ramp): drive straight this far
    # START light
    'wait_for_green': True, 'green_frames': 3,
    'light_roi': [0.0, 0.0, 1.0, 0.6], 'light_min_area': 20, 'light_max_area': 5000,
    # Obstacles (LiDAR)
    'stop_distance': 0.35, 'corridor_half': 0.15, 'lidar_x': 0.0,
    'body_front': 0.12,           # ignore LiDAR points closer than this (own body, ramp edges)
    # Overtaking (only after a highway-entry sign)
    'overtake': True, 'overtake_wait': 1.5, 'overtake_length': 0.9, 'highway_speed': 0.35,
    # Signs from /signs: a sign counts when its box is at least this tall (= close enough)
    'sign_min_height': 40, 'sign_cooldown': 6.0,
    'stop_classes': ['stop'], 'stop_time': 3.0,
    'slow_classes': ['pedestrian', 'tunnel', 'hump', 'bus', 'uneven_road'],
    'slow_factor': 0.6, 'slow_time': 4.0,
    'left_classes': ['turn_left'], 'right_classes': ['turn_right'],
    'straight_classes': ['go_straight'], 'branch_time': 3.0,
    'hw_enter_classes': ['hw_enter'], 'hw_exit_classes': ['hw_exit'],
    'red_classes': ['red_light'], 'green_classes': ['green_light'], 'red_clear_time': 1.0,
}
