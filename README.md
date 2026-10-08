# UEH CRC 2026 Simulation Round

**Participant:** Nguyen Tuan Kiet
**School:** University of Economics Ho Chi Minh City (UEH)
**Faculty:** Institute of Intelligent and Interactive Technologies

## Solution

This repository contains my solution for the UEH Creative Robot Contest 2026
Simulation Round.

The solution is implemented in the `crc_sim` ROS 2 package and runs through
the `starter` node.

The simulator must already be running before starting the solution.

## Run the Solution

From a shell inside the provided simulation environment, run:

```bash
ros2 run crc_sim starter
```

This is the only command required to start the submitted solution.

## Dependencies

No additional external model files are required.

The solution uses the ROS 2 and Python dependencies provided by the official
simulation environment, including ROS 2 Humble, OpenCV, NumPy, and `cv_bridge`.

## Debug Camera Views

The following commands are optional and are used only for visualization and
debugging. They are NOT required to run the submitted solution.

### Lane Tracking Debug View

```bash
ros2 run rqt_image_view rqt_image_view /lane_debug/image
```

This view shows the lane detection and lane-following debug output.

### Perception Debug View

```bash
ros2 run rqt_image_view rqt_image_view /detection_debug/image
```

This view shows traffic-sign, traffic-light, pedestrian, and perception debug
output.

### Raw Robot Camera

```bash
ros2 run rqt_image_view rqt_image_view /camera/image_raw
```

## Notes

- Debug image topics are visualization outputs only.
- Debug viewers are not required for the controller to operate.
- The simulator must be started before running `ros2 run crc_sim starter`.
- The solution uses the robot camera, LiDAR, odometry, and IMU interfaces used
  by the submitted controller.
