#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Watch a UEH CRC 2026 car from a Linux laptop: camera in rqt_image_view, rqt,
# rviz2 or a ROS 2 shell, all inside the crc_sim:humble image (no ROS install
# needed on the laptop). Runs on the LAPTOP, not on the car.
#
# Why: some Wi-Fi networks (e.g. UEH-B2-MakerSpace) drop multicast, so ROS 2
# discovery finds nothing - no error, just no topics - although unicast works.
# This script writes a Fast DDS profile that contacts the car by IP (initial
# peers). Nothing changes on the car.
#
#   bash view_car.sh <car-ip>                  # camera (/camera/image_raw/compressed)
#   bash view_car.sh <car-ip> image <topic>    # another image topic, e.g.
#                                              #   /red_follower/debug/compressed
#   bash view_car.sh <car-ip> rqt              # full rqt
#   bash view_car.sh <car-ip> rviz             # rviz2
#   bash view_car.sh <car-ip> check            # list topics + camera rate (no GUI)
#   bash view_car.sh <car-ip> sh               # ROS 2 shell (ros2 topic echo ...)
#   bash view_car.sh <car-ip> profile          # only write the profile; prints how to
#                                              #   use it with a native ROS 2 Humble install
#   bash view_car.sh stop                      # close the viewer
#
# Several cars at once: comma-separated IPs, e.g. 172.21.210.125,172.21.210.126
# Environment overrides:
#   DOMAIN   ROS_DOMAIN_ID of the car (default 30; see run_car.sh domain on the car)
#   IMAGE    Docker image with ROS 2 Humble + rqt/rviz2 (default crc_sim:humble)
# ---------------------------------------------------------------------------
set -u

IMAGE="${IMAGE:-crc_sim:humble}"
DOMAIN="${DOMAIN:-30}"
NAME=crc_view
PROFILE_DIR="$HOME/.config/crc"

usage() { sed -n '/^#   bash/,/^# Several/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 1; }

if [ "${1:-}" = stop ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 && echo "Viewer closed." || echo "No viewer running."
  exit 0
fi

CARS="${1:-}"; [ -n "$CARS" ] || usage
shift
CMD="${1:-image}"; [ $# -gt 0 ] && shift

IFS=',' read -r -a IPS <<< "$CARS"
for ip in "${IPS[@]}"; do
  [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || { echo "Not an IPv4 address: $ip"; exit 1; }
done

# --- Fast DDS profile: unicast discovery of the car(s) ----------------------
# An initial peer without a port is tried on the ports of participant IDs
# 0..maxInitialPeersRange-1. The car runs one participant per ROS process
# (5 drivers + team program + tools), more than the default range of 4.
# maxInitialPeersRange is a UDPv4 transport setting, so the transports are
# declared explicitly (UDP + shared memory, as the defaults). Fast DDS ignores
# the WHOLE file on any XML error, so keep this exact structure.
mkdir -p "$PROFILE_DIR"
PROFILE="$PROFILE_DIR/fastdds_peers_${CARS//,/_}.xml"
{
  cat <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!-- Written by view_car.sh: reach the car(s) by unicast when the Wi-Fi drops multicast -->
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <transport_descriptors>
    <transport_descriptor>
      <transport_id>crc_udp</transport_id>
      <type>UDPv4</type>
      <maxInitialPeersRange>32</maxInitialPeersRange>
    </transport_descriptor>
    <transport_descriptor>
      <transport_id>crc_shm</transport_id>
      <type>SHM</type>
    </transport_descriptor>
  </transport_descriptors>
  <participant profile_name="crc_car_peers" is_default_profile="true">
    <rtps>
      <userTransports>
        <transport_id>crc_udp</transport_id>
        <transport_id>crc_shm</transport_id>
      </userTransports>
      <useBuiltinTransports>false</useBuiltinTransports>
      <builtin>
        <initialPeersList>
EOF
  for ip in "${IPS[@]}"; do
    echo "          <locator><udpv4><address>$ip</address></udpv4></locator>"
  done
  cat <<'EOF'
        </initialPeersList>
      </builtin>
    </rtps>
  </participant>
</profiles>
EOF
} > "$PROFILE"

if [ "$CMD" = profile ]; then
  echo "Profile: $PROFILE"
  echo "With ROS 2 Humble installed on this laptop, in each terminal:"
  echo "  export ROS_DOMAIN_ID=$DOMAIN"
  echo "  export FASTRTPS_DEFAULT_PROFILES_FILE=$PROFILE"
  echo "  ros2 daemon stop   # the daemon keeps its old discovery settings"
  exit 0
fi

for ip in "${IPS[@]}"; do
  ping -c 1 -W 2 "$ip" >/dev/null 2>&1 || echo "Warning: $ip does not answer ping (same Wi-Fi? car on?)"
done
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "Image $IMAGE missing: build it with the simulation pack (run_docker.sh build)"; exit 1; }

# --- GUI access (same approach as the simulation pack's run_docker.sh) -----
find_xauth() {
  local f rt="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  [ -n "${XAUTHORITY:-}" ] && [ -f "$XAUTHORITY" ] && { echo "$XAUTHORITY"; return; }
  f=$(ls "$rt"/.mutter-Xwaylandauth* 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; return; }
  [ -f "$rt/gdm/Xauthority" ] && { echo "$rt/gdm/Xauthority"; return; }
  [ -f "$HOME/.Xauthority" ] && { echo "$HOME/.Xauthority"; return; }
  echo ""
}

ARGS=(--net=host --ipc=host
      -e ROS_DOMAIN_ID="$DOMAIN"
      -e FASTRTPS_DEFAULT_PROFILES_FILE=/crc_peers.xml
      -v "$PROFILE":/crc_peers.xml:ro)
GUI=(-e "DISPLAY=${DISPLAY:-:0}" -e XDG_RUNTIME_DIR=/tmp/rt -e QT_X11_NO_MITSHM=1
     -e LIBGL_ALWAYS_SOFTWARE=1 -v /tmp/.X11-unix:/tmp/.X11-unix)
XA=$(find_xauth)
[ -n "$XA" ] && GUI+=(-e XAUTHORITY=/tmp/.Xauth -v "$XA":/tmp/.Xauth:ro)

SRC='mkdir -p /tmp/rt && chmod 700 /tmp/rt && source /opt/ros/humble/setup.bash'

# Start a GUI tool in the background and wait until its window exists
gui() {
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" "${ARGS[@]}" "${GUI[@]}" "$IMAGE" bash -c "$SRC && $1" >/dev/null \
    || exit 1
  sleep 4
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then
    echo "The viewer exited right away:"; docker logs "$NAME" 2>&1 | tail -n 15; exit 1
  fi
  echo "Viewer open on ${DISPLAY:-:0} (ROS_DOMAIN_ID=$DOMAIN, car ${CARS})."
  echo "  Window hidden? Alt+Tab.   Empty image? pick the topic in the list and press refresh."
  if command -v view_car >/dev/null 2>&1; then
    echo "  Close: view_car stop"
  else
    echo "  Close: bash $(printf '%q' "$0") stop"
  fi
}

case "$CMD" in
  image)
    gui "ros2 run rqt_image_view rqt_image_view ${1:-/camera/image_raw/compressed}"
    ;;
  rqt)
    gui "rqt"
    ;;
  rviz)
    # Ready-made layout: LiDAR, odometry, TF and the compressed camera, seen from above.
    # Copied next to the Fast DDS profile because Docker cannot mount cloud-drive paths.
    RVIZ_CFG="$PROFILE_DIR/crc_car.rviz"
    cp "$(dirname "$0")/../config/crc_car.rviz" "$RVIZ_CFG" 2>/dev/null || true
    if [ -f "$RVIZ_CFG" ]; then
      GUI+=(-v "$RVIZ_CFG:/crc_car.rviz:ro")
      gui "rviz2 -d /crc_car.rviz"
    else
      gui "rviz2"
    fi
    ;;
  check)
    docker run --rm "${ARGS[@]}" "$IMAGE" bash -c "$SRC
      topics=\$(ros2 topic list --no-daemon 2>/dev/null)
      n=\$(printf '%s\n' \"\$topics\" | grep -c '^/camera\|^/odom\|^/scan')
      if [ \"\$n\" -eq 0 ]; then
        echo 'No car topics found. Check: same Wi-Fi, car IP, ROS_DOMAIN_ID (DOMAIN=...).'; exit 1
      fi
      printf '%s\n' \"\$topics\"
      echo '--- camera rate:'
      timeout 6 ros2 topic hz /camera/image_raw/compressed 2>/dev/null | grep -m1 average \
        || echo 'no camera images (is crc-camera running on the car?)'"
    ;;
  sh)
    docker run --rm -it "${ARGS[@]}" "${GUI[@]}" "$IMAGE" bash -c "$SRC && exec bash"
    ;;
  *)
    usage
    ;;
esac
