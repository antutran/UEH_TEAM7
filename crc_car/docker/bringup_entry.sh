#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Entry point of the crc_bringup container. Runs the two bring-up processes
# directly (not through `ros2 launch` or `ros2 run`, each of which keeps an
# extra ~30-50 MB Python process alive on the Nano). Each process is restarted
# if it exits. Settings come from the environment (~/crc_car.env, see
# ros/crc_bringup/launch/bringup.launch.py).
# ---------------------------------------------------------------------------
# The ROS setup scripts reference unset variables, so enable `set -u` only afterwards
source /opt/ros/humble/setup.bash
source /opt/crc_ws/install/setup.bash
set -u

e() { local v="${!1:-}"; echo "${v:-$2}"; }
# 1/true/yes -> true, anything else -> false (ROS bool parameters)
b() { case "$(e "$1" "$2")" in 1|true|True|yes) echo true;; *) echo false;; esac; }

BRINGUP_ARGS=(--ros-args
  -p board_port:=/dev/myserial
  -p left_motor:="$(e CRC_LEFT_MOTOR 1)"
  -p right_motor:="$(e CRC_RIGHT_MOTOR 4)"
  -p left_invert:="$(b CRC_LEFT_INVERT 0)"
  -p right_invert:="$(b CRC_RIGHT_INVERT 0)"
  -p wheel_separation:="$(e CRC_WHEEL_SEPARATION 0.298)"
  -p wheel_radius:="$(e CRC_WHEEL_RADIUS 0.0325)"
  -p counts_per_rev:="$(e CRC_COUNTS_PER_REV 1320.0)"
  -p max_pwm:="$(e CRC_MAX_PWM 100)"
  -p kp:="$(e CRC_KP 3.0)"
  -p ki:="$(e CRC_KI 15.0)"
  -p i_band:="$(e CRC_I_BAND 50)"
  -p ff_gain_left:="$(e CRC_FF_GAIN_LEFT 3.0)"
  -p ff_gain_right:="$(e CRC_FF_GAIN_RIGHT 3.0)"
  -p speed_window:="$(e CRC_SPEED_WINDOW 0.08)"
  -p max_accel:="$(e CRC_MAX_ACCEL 0.8)"
  -p max_decel:="$(e CRC_MAX_DECEL 1.5)"
  -p max_ang_accel:="$(e CRC_MAX_ANG_ACCEL 6.0)"
  -p camera_width:="$(e CAMERA_WIDTH 640)"
  -p camera_height:="$(e CAMERA_HEIGHT 480)"
  -p camera_hfov_deg:="$(e CRC_CAMERA_HFOV_DEG 62.2)"
  -p scan_yaw_offset_deg:="$(e CRC_SCAN_YAW_OFFSET_DEG 0.0)")

LIDAR_ARGS=(--ros-args -r scan:=scan_raw
  -p channel_type:=serial -p serial_port:=/dev/rplidar -p serial_baudrate:=460800
  -p frame_id:=base_scan -p inverted:=false -p angle_compensate:=true -p scan_mode:=Standard)

keep_running() {  # name, delay, command...
  local name=$1 delay=$2; shift 2
  while true; do
    "$@"
    echo "[crc_bringup] $name exited with code $?, restarting in ${delay}s" >&2
    sleep "$delay"
  done
}

trap 'kill 0' TERM INT
echo "[crc_bringup] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
LIB=/opt/crc_ws/install/lib
keep_running bringup 2 "$LIB/crc_bringup/bringup" "${BRINGUP_ARGS[@]}" &
# Sign detector on the GPU, only when an engine is present (build it with trtexec on the car)
YOLO_ENGINE="$(e CRC_YOLO_ENGINE /engines/signs.engine)"
if [ "$(e CRC_YOLO 1)" = 1 ] && [ -f "$YOLO_ENGINE" ]; then
  keep_running yolo_trt 3 "$LIB/crc_yolo/yolo_trt" --ros-args \
    -p engine:="$YOLO_ENGINE" -p conf:="$(e CRC_YOLO_CONF 0.4)" -p max_rate:="$(e CRC_YOLO_RATE 15)" &
else
  echo "[crc_bringup] sign detector off (CRC_YOLO=$(e CRC_YOLO 1), engine $YOLO_ENGINE missing?)"
fi
if [ "$(e CRC_LIDAR 1)" = 1 ]; then
  # Wait for the LiDAR to enumerate instead of crash-looping the driver
  keep_running sllidar 3 bash -c 'until [ -e /dev/rplidar ]; do sleep 1; done; exec "$@"' _ \
    "$LIB/sllidar_ros2/sllidar_node" "${LIDAR_ARGS[@]}" &
fi
wait
