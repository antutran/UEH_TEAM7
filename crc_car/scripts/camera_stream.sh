#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Stream the CSI camera (IMX219 on CAM0) as raw RGBA frames over TCP to
# 127.0.0.1:5600 for crc_bringup/camera_node. Runs on the Jetson host because
# nvarguscamerasrc needs the host Argus daemon. Scaling, flipping and colour
# conversion run on the VIC hardware (nvvidconv), so the CPU only copies bytes.
# A slow reader gets the newest frame instead of a growing backlog (low latency).
# Started at boot by crc-camera.service; settings come from ~/crc_car.env:
#   CAMERA_FLIP    nvvidconv flip-method: 0 = none, 2 = rotate 180 (camera upside down)
#   CAMERA_WIDTH / CAMERA_HEIGHT / CAMERA_FPS   output size and rate (default 640x480 @ 30)
#   CAMERA_PORT    TCP port (default 5600)
#   CAMERA_CROP    fraction of the sensor kept, centred (default 1 = full view). With the
#                  IMX219-200 fisheye (200 deg diagonal), 0.385 gives the same view as the
#                  IMX219-77 lens (77/200): the angle is roughly proportional to the radius.
# ---------------------------------------------------------------------------
set -u
ENV_FILE="${CRC_ENV:-$HOME/crc_car.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
FLIP="${CAMERA_FLIP:-0}"
W="${CAMERA_WIDTH:-640}"
H="${CAMERA_HEIGHT:-480}"
FPS="${CAMERA_FPS:-30}"
PORT="${CAMERA_PORT:-5600}"
CROP="${CAMERA_CROP:-1}"

# Centred crop rectangle in sensor pixels (nvvidconv takes the rectangle's coordinates)
SW=1640; SH=1232
read -r CW CH < <(awk -v c="$CROP" -v w="$SW" -v h="$SH" \
  'BEGIN { if (c <= 0 || c > 1) c = 1; printf "%d %d\n", int(w * c / 2) * 2, int(h * c / 2) * 2 }')
CL=$(( (SW - CW) / 2 )); CT=$(( (SH - CH) / 2 ))
CROP_ARGS=""
[ "$CW" -lt "$SW" ] && CROP_ARGS="left=$CL right=$((CL + CW)) top=$CT bottom=$((CT + CH))"

# 1640x1232 is the full-field-of-view IMX219 mode
exec gst-launch-1.0 -q nvarguscamerasrc sensor-id=0 \
  ! "video/x-raw(memory:NVMM),width=$SW,height=$SH,framerate=30/1" \
  ! nvvidconv flip-method="$FLIP" $CROP_ARGS \
  ! "video/x-raw,width=$W,height=$H,format=RGBA" \
  ! videorate drop-only=true ! "video/x-raw,framerate=$FPS/1" \
  ! tcpserversink host=127.0.0.1 port="$PORT" sync=false \
      buffers-soft-max=2 recover-policy=latest
