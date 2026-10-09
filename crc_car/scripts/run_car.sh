#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the UEH CRC 2026 ROS 2 Humble environment on a Jetson Nano (JetPack 4.6).
#
# Two containers share the host network and IPC (fast shared-memory DDS):
#   crc_bringup  car drivers, started at boot by Docker: /cmd_vel, /odom, /imu,
#                /camera/image_raw, /camera/camera_info, /scan, /tf (as in simulation)
#   crc_car      the team's code (GPU-enabled), started on demand
#
#   bash run_car.sh build            # build the image (needs Internet)
#   bash run_car.sh bringup          # (re)start the drivers with the settings in ~/crc_car.env
#                                    #   (settings changes need only this, no rebuild;
#                                    #    DEV=1 also runs the driver code from this checkout)
#   bash run_car.sh bringup-stop     # stop the drivers (e.g. to use the serial ports directly)
#   bash run_car.sh bringup-logs     # follow the driver log
#   bash run_car.sh domain <id>      # set ROS_DOMAIN_ID for both containers and restart the drivers
#   bash run_car.sh up               # start the team container in the background
#   bash run_car.sh compile          # colcon build the team workspace
#   bash run_car.sh sh               # open a shell in the team container (ROS sourced)
#   bash run_car.sh run "<command>"  # run one command attached to this terminal (stops with SSH)
#   bash run_car.sh start "<command>"  # run the team program in the background: it keeps running
#                                    #   when SSH drops or the laptop is closed; replaces a running one
#   bash run_car.sh stop             # stop the background program (Ctrl+C, then harder)
#   bash run_car.sh logs             # follow its output (Ctrl+C leaves the program running)
#   bash run_car.sh frames [period_s] [count]  # save camera JPEGs for a dataset (default every
#                                    #   0.5 s until Ctrl+C) to /data/datasets/frames_<date>_<time>
#   bash run_car.sh gpu-test [file.engine]  # check GPU/TensorRT inside the team container
#   bash run_car.sh status           # status of both containers, the program and the topics
#   bash run_car.sh down             # stop and remove the team container
#
# Environment overrides (e.g. TEAM=team05 bash run_car.sh up):
#   TEAM     team name; each team gets its own build volume  (default: default)
#   SRC      team src directory on the Jetson                 (default: ~/crc_ws/src)
#   ENGINES  TensorRT engine directory, mounted read-only     (default: ~/crc_engines)
#   CRC_ENV  per-car settings file                            (default: ~/crc_car.env)
#   IMAGE    image name                                       (default: crc_car:humble)
# ---------------------------------------------------------------------------
set -u

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SELF")"
IMAGE="${IMAGE:-crc_car:humble}"
NAME=crc_car
BRINGUP=crc_bringup
TEAM="${TEAM:-default}"
SRC="${SRC:-$HOME/crc_ws/src}"
ENGINES="${ENGINES:-$HOME/crc_engines}"
CRC_ENV="${CRC_ENV:-$HOME/crc_car.env}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

[ -f "$CRC_ENV" ] || cp "$ROOT/config/crc_car.env" "$CRC_ENV" 2>/dev/null || true
DOMAIN="$(sed -n 's/^ROS_DOMAIN_ID=//p' "$CRC_ENV" 2>/dev/null | tail -1)"
DOMAIN="${DOMAIN:-30}"

# --- GPU: bind-mount the host CUDA/cuDNN/TensorRT libraries (read-only) -----
# The Tegra driver part (/dev/nvhost-*, libcuda, ...) is mounted by
# nvidia-container-runtime from /etc/nvidia-container-runtime/host-files-for-container.d/l4t.csv.
gpu_args() {
  local f
  printf '%s\n' --runtime nvidia
  for f in $(dpkg -L libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 libcudnn8 2>/dev/null \
             | grep '^/usr/lib/aarch64-linux-gnu/lib.*\.so'); do
    printf '%s\n' -v "$f:$f:ro"
  done
  printf '%s\n' -v /usr/local/cuda-10.2:/usr/local/cuda-10.2:ro
  [ -x "$TRTEXEC" ] && printf '%s\n' -v "$TRTEXEC:/usr/local/bin/trtexec:ro"
  return 0
}

check_host() {
  docker info >/dev/null 2>&1 || { echo "Cannot reach Docker (is the user in the docker group? log in again)"; exit 1; }
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Image $IMAGE missing: bash run_car.sh build"; exit 1; }
}

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
need_running() { running "$NAME" || { echo "Team container is not running. Start it with: bash run_car.sh up"; exit 1; }; }

start_bringup() {
  check_host
  docker rm -f "$BRINGUP" >/dev/null 2>&1
  # /dev is shared so the USB serial ports (/dev/myserial, /dev/rplidar) survive unplug/replug;
  # the cgroup rules allow only ttyUSB (major 188) and ttyACM (major 166) devices.
  # DEV=1: run the bring-up code straight from this checkout instead of the copy baked into
  # the image, so Python and entry-script edits only need `bash run_car.sh bringup` (no build).
  local dev_args=()
  if [ "${DEV:-0}" = 1 ]; then
    local pkg
    pkg=$(docker run --rm "$IMAGE" bash -c 'ls -d /opt/crc_ws/install/lib/python3*/site-packages/crc_bringup')
    dev_args=(-v "$ROOT/ros/crc_bringup/crc_bringup:$pkg:ro"
              -v "$ROOT/docker/bringup_entry.sh:/opt/crc_ws/bringup_entry.sh:ro")
  fi
  # GPU (CUDA/TensorRT from the host) and the engines are needed by the sign detector (crc_yolo)
  mkdir -p "$ENGINES"
  local gpu
  mapfile -t gpu < <(gpu_args)
  docker run -d --name "$BRINGUP" --restart unless-stopped "${gpu[@]}" \
    -v "$ENGINES":/engines:ro \
    --net=host --ipc=host --env-file "$CRC_ENV" \
    -v /dev:/dev --device-cgroup-rule 'c 188:* rmw' --device-cgroup-rule 'c 166:* rmw' \
    "${dev_args[@]}" "$IMAGE" bash /opt/crc_ws/bringup_entry.sh >/dev/null || exit 1
  echo "Started $BRINGUP (ROS_DOMAIN_ID=$DOMAIN, settings: $CRC_ENV)$([ "${DEV:-0}" = 1 ] && echo ', DEV: code from '"$ROOT"). Log: bash run_car.sh bringup-logs"
}

team_up() {
  check_host
  docker rm -f "$NAME" >/dev/null 2>&1
  mkdir -p "$SRC" "$ENGINES"
  local ga
  mapfile -t ga < <(gpu_args)
  # TF card data disk (setup_data_card.sh): datasets, models, logs, rosbags
  local data_args=()
  mountpoint -q /data && data_args=(-v /data:/data)
  docker run -d --name "$NAME" "${ga[@]}" --net=host --ipc=host \
    -e ROS_DOMAIN_ID="$DOMAIN" \
    -v "$SRC":/ws/src -v "$ENGINES":/engines:ro -v "crc_build_$TEAM":/ws_build \
    "${data_args[@]}" "$IMAGE" sleep infinity >/dev/null || exit 1
  echo "Started container $NAME  (TEAM=$TEAM, ROS_DOMAIN_ID=$DOMAIN)"
  echo "  src     : $SRC -> /ws/src"
  echo "  engines : $ENGINES -> /engines (read-only)"
  if [ ${#data_args[@]} -gt 0 ]; then echo "  data    : /data -> /data (TF card)"; else echo "  data    : /data not mounted (no TF data card)"; fi
  running "$BRINGUP" || echo "  warning : drivers ($BRINGUP) not running: bash run_car.sh bringup"
}

# --- Background team program (start/stop/logs) ---------------------------------
# The program runs inside the team container under `docker exec -d`, so it belongs to
# the Docker daemon, not to the SSH session. It gets its own process group (setsid),
# whose id is kept in $PROG_PID; stop signals the whole group, which also reaches the
# nodes started by `ros2 launch`. Output goes to a log on the TF card when there is one
# (the eMMC is small); the previous run's log is kept as *.prev.log.
PROG_PID=/tmp/team_program.pid
LOG_DIR_CMD='if [ -d /data ] && [ -w /data ]; then echo /data/logs; else echo /ws_build/logs; fi'

prog_running() {
  running "$NAME" && docker exec "$NAME" bash -c \
    "[ -s $PROG_PID ] && kill -0 -- -\$(cat $PROG_PID) 2>/dev/null"
}

prog_stop() {
  # SIGINT first (what Ctrl+C sends; ros2 launch shuts its nodes down cleanly), then
  # SIGTERM, then SIGKILL. The drivers stop the wheels 0.5 s after /cmd_vel goes quiet.
  docker exec "$NAME" bash -c "
    [ -s $PROG_PID ] || exit 0
    g=\$(cat $PROG_PID)
    for sig in INT TERM KILL; do
      kill -0 -- -\$g 2>/dev/null || break
      kill -\$sig -- -\$g 2>/dev/null
      for i in \$(seq 20); do kill -0 -- -\$g 2>/dev/null || break; sleep 0.5; done
    done
    rm -f $PROG_PID"
}

SRC_ROS='source /opt/ros/humble/setup.bash && source /opt/crc_ws/install/setup.bash && { [ -f /ws_build/install/setup.bash ] && source /ws_build/install/setup.bash || true; }'

case "${1:-}" in

  build)
    # Tag the build stage too: untagged intermediate images are deleted by `docker image prune`,
    # which would throw away the cached sllidar_ros2 compile (~2 min on the Nano). With the tag,
    # a code-only change rebuilds in ~15 s and old images can be pruned safely.
    docker build --target build -t "${IMAGE%%:*}:build" -f "$ROOT/docker/Dockerfile" "$ROOT" >/dev/null &&
      docker build -t "$IMAGE" -f "$ROOT/docker/Dockerfile" "$ROOT" &&
      docker image prune -f >/dev/null
    ;;

  bringup)
    start_bringup
    ;;

  bringup-stop)
    docker rm -f "$BRINGUP" >/dev/null 2>&1 && echo "Stopped $BRINGUP." || echo "No container named $BRINGUP."
    ;;

  bringup-logs)
    docker logs -f --tail 50 "$BRINGUP"
    ;;

  domain)
    [[ "${2:-}" =~ ^[0-9]+$ ]] && [ "$2" -le 101 ] || { echo "Usage: bash run_car.sh domain <0-101>"; exit 1; }
    if grep -q '^ROS_DOMAIN_ID=' "$CRC_ENV"; then
      sed -i "s/^ROS_DOMAIN_ID=.*/ROS_DOMAIN_ID=$2/" "$CRC_ENV"
    else
      echo "ROS_DOMAIN_ID=$2" >> "$CRC_ENV"
    fi
    DOMAIN=$2
    start_bringup
    running "$NAME" && echo "Restart the team container to apply it: bash run_car.sh up"
    ;;

  up)
    team_up
    ;;

  compile)
    need_running
    docker exec "$NAME" bash -c "source /opt/ros/humble/setup.bash && source /opt/crc_ws/install/setup.bash && \
      cd /ws && colcon build --symlink-install --build-base /ws_build/build --install-base /ws_build/install"
    ;;

  sh)
    need_running
    docker exec -it "$NAME" bash -c "$SRC_ROS && exec bash"
    ;;

  run)
    need_running
    shift; [ $# -ge 1 ] || { echo 'Usage: bash run_car.sh run "<command>"'; exit 1; }
    docker exec -it "$NAME" bash -c "$SRC_ROS && $*"
    ;;

  start)
    shift; [ $# -ge 1 ] || { echo 'Usage: bash run_car.sh start "<command>"   e.g. "ros2 launch my_pkg race.launch.py"'; exit 1; }
    running "$NAME" || team_up
    if prog_running; then echo "Stopping the program that is already running..."; prog_stop; fi
    LOG_DIR=$(docker exec "$NAME" bash -c "$LOG_DIR_CMD")
    LOG="$LOG_DIR/team_$TEAM.log"
    # The command travels in an environment variable, so it needs no extra quoting.
    docker exec -d -e TEAM_CMD="$*" -e RUN_LOG="$LOG" -e ROS_LOG_DIR="$LOG_DIR/ros" "$NAME" bash -c "
      mkdir -p '$LOG_DIR' && { [ -f \"\$RUN_LOG\" ] && mv -f \"\$RUN_LOG\" \"\${RUN_LOG%.log}.prev.log\"; }
      exec >>\"\$RUN_LOG\" 2>&1
      chown $(id -u):$(id -g) \"\$RUN_LOG\" 2>/dev/null
      $SRC_ROS
      echo \"=== [run_car] \$(date '+%F %T') start: \$TEAM_CMD\"
      setsid -w bash -c 'echo \$\$ > $PROG_PID; exec bash -c \"\$TEAM_CMD\"'
      code=\$?
      rm -f $PROG_PID
      echo \"=== [run_car] \$(date '+%F %T') exited with code \$code\"" || exit 1
    sleep 3.5
    if prog_running; then
      echo "Program started in the background (it keeps running after SSH disconnects)."
    else
      echo "The program exited within 2 s; last lines of the log:"
    fi
    echo "  log : ${LOG/#\/ws_build/crc_build_$TEAM volume:}    follow: bash run_car.sh logs    stop: bash run_car.sh stop"
    prog_running || docker exec "$NAME" tail -n 15 "$LOG"
    ;;

  stop)
    if prog_running; then prog_stop && echo "Program stopped."; else echo "No background program running."; fi
    ;;

  logs)
    need_running
    LOG="$(docker exec "$NAME" bash -c "$LOG_DIR_CMD")/team_$TEAM.log"
    docker exec "$NAME" test -f "$LOG" || { echo "No log yet ($LOG). Start a program: bash run_car.sh start \"<command>\""; exit 1; }
    prog_running || echo "(program not running; showing the last log)"
    docker exec -it "$NAME" tail -n 50 -F "$LOG"
    ;;

  frames)
    running "$NAME" || team_up >/dev/null
    shift
    docker cp "$SELF/save_frames.py" "$NAME":/tmp/save_frames.py >/dev/null || exit 1
    docker exec -it "$NAME" bash -c "$SRC_ROS && python3 /tmp/save_frames.py $(printf '%q ' "$@")"
    ;;

  gpu-test)
    need_running
    ENG="${2:-}"
    if [ -z "$ENG" ]; then ENG=$(ls "$ENGINES"/*.engine 2>/dev/null | head -1); fi
    [ -n "$ENG" ] && [ -f "$ENG" ] || { echo "No .engine file in $ENGINES (build one on the host with trtexec first)"; exit 1; }
    docker exec "$NAME" bash -c "ls /dev/nvhost-gpu /dev/nvmap >/dev/null && echo 'GPU: /dev/nvhost-gpu and /dev/nvmap present' && \
      trtexec --loadEngine=/engines/$(basename "$ENG") --duration=10 2>&1 | grep -E 'Throughput|GPU Compute Time: min|PASSED|FAILED|\[E\]'"
    ;;

  status)
    docker ps -a --filter "name=^(${NAME}|${BRINGUP})$" --format '{{.Names}}  {{.Status}}  {{.Image}}'
    echo "ROS_DOMAIN_ID=$DOMAIN  (settings: $CRC_ENV)"
    if prog_running; then
      echo "Background program: running  ($(docker exec "$NAME" bash -c "ps -o args= -p \$(cat $PROG_PID)" | cut -c1-80))"
    else
      echo "Background program: none"
    fi
    for c in "$NAME" "$BRINGUP"; do
      if running "$c"; then
        docker exec "$c" bash -c "$SRC_ROS && timeout 10 ros2 topic list" 2>/dev/null | sed 's/^/  topic: /'
        break
      fi
    done
    ;;

  down)
    docker rm -f "$NAME" >/dev/null 2>&1 && echo "Stopped $NAME." || echo "No container named $NAME."
    ;;

  *)
    awk 'NR > 2 && /^# ---/ { exit } NR > 2' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
