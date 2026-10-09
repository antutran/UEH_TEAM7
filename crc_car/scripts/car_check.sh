#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# UEH CRC 2026 car self-test. Run on the Jetson (no sudo needed), e.g. after cloning,
# before a practice session or when something misbehaves:
#
#   bash car_check.sh            # system, devices, drivers and sensor topics (~10 s)
#   bash car_check.sh --motion   # also drives both wheels for ~6 s: LIFT THE WHEELS FIRST
#
# Each line is PASS, WARN (works, but look at it) or FAIL (fix before racing).
# Exit code: 0 when nothing failed, 1 otherwise.
# ---------------------------------------------------------------------------
set -u

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRC_ENV="${CRC_ENV:-$HOME/crc_car.env}"
ENGINES="${ENGINES:-$HOME/crc_engines}"
IMAGE="${IMAGE:-crc_car:humble}"
BRINGUP=crc_bringup
TEAM_CONTAINER=crc_car
MOTION=0
case "${1:-}" in
  --motion) MOTION=1 ;;
  "") ;;
  *) awk 'NR > 2 && /^# ---/ { exit } NR > 2' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

N_PASS=0 N_WARN=0 N_FAIL=0
if [ -t 1 ]; then G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' B=$'\e[1m' Z=$'\e[0m'; else G= Y= R= B= Z=; fi
report() {  # level check detail...
  local level=$1 check=$2; shift 2
  case "$level" in
    PASS) N_PASS=$((N_PASS + 1)); printf '%s%-4s%s  %-14s %s\n' "$G" PASS "$Z" "$check" "$*" ;;
    WARN) N_WARN=$((N_WARN + 1)); printf '%s%-4s%s  %-14s %s\n' "$Y" WARN "$Z" "$check" "$*" ;;
    FAIL) N_FAIL=$((N_FAIL + 1)); printf '%s%-4s%s  %-14s %s\n' "$R" FAIL "$Z" "$check" "$*" ;;
    *)    printf '%-4s  %-14s %s\n' INFO "$check" "$*" ;;
  esac
}
section() { printf '\n%s== %s ==%s\n' "$B" "$1" "$Z"; }
env_get() { sed -n "s/^$1=//p" "$CRC_ENV" 2>/dev/null | tail -1; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

echo "${B}UEH CRC 2026 car check${Z}  $(hostname)  $(date '+%F %T')"

# --- System -------------------------------------------------------------------
section System
rel=$(head -1 /etc/nv_tegra_release 2>/dev/null)
if [[ "$rel" == *"R32 (release), REVISION: 7.5"* ]]; then report PASS L4T "R32.7.5 (JetPack 4.6.5)"
else report WARN L4T "unexpected release: ${rel:-unknown}"; fi
held=$(apt-mark showhold 2>/dev/null | grep -c '^nvidia-l4t')
if [ "$held" -ge 20 ]; then report PASS "l4t hold" "$held nvidia-l4t packages held"
else report WARN "l4t hold" "only $held nvidia-l4t packages held (apt upgrade could break boot)"; fi
year=$(date +%Y)
if [ "$year" -ge 2026 ]; then report PASS clock "$(date '+%F %T %Z')"
else report FAIL clock "$(date '+%F') - no RTC battery; HTTPS/apt fail. From the laptop: ssh jetson@<car> \"sudo date -s @\$(date +%s)\""; fi
mode=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: //p')
if [ "$mode" = MAXN ]; then report PASS power "nvpmodel MAXN (10 W)"
else report WARN power "nvpmodel ${mode:-unknown}; full speed needs MAXN (sudo nvpmodel -m 0)"; fi
tmax=0 tdesc=
for z in /sys/class/thermal/thermal_zone*; do
  case "$(cat "$z/type" 2>/dev/null)" in
    CPU-therm|GPU-therm)
      t=$(( $(cat "$z/temp") / 1000 )); tdesc+="$(cat "$z/type") ${t}C  "
      [ "$t" -gt "$tmax" ] && tmax=$t ;;
  esac
done
if [ "$tmax" -lt 70 ]; then report PASS temperature "$tdesc"
elif [ "$tmax" -lt 85 ]; then report WARN temperature "$tdesc(hot: check the fan)"
else report FAIL temperature "$tdesc(throttling: check the fan)"; fi
avail=$(awk '/MemAvailable/ { print int($2 / 1024) }' /proc/meminfo)
report INFO memory "${avail} MB RAM available, swap $(awk '/SwapTotal/ { print int($2 / 1024) }' /proc/meminfo) MB"

# --- Storage ------------------------------------------------------------------
section Storage
free_mb=$(df -BM --output=avail / | tail -1 | tr -dc 0-9)
if [ "$free_mb" -ge 2048 ]; then report PASS eMMC "$((free_mb / 1024)) GB free"
elif [ "$free_mb" -ge 1024 ]; then report WARN eMMC "${free_mb} MB free (docker image prune -f; move data to /data)"
else report FAIL eMMC "${free_mb} MB free - the system may stop working"; fi
if mountpoint -q /data; then
  report PASS "data card" "/data $(df -h --output=avail /data | tail -1 | tr -d ' ') free"
else
  report WARN "data card" "/data not mounted: logs and datasets go to the eMMC (setup_data_card.sh)"
fi

# --- Network ------------------------------------------------------------------
section Network
wifi=$(nmcli -t -f DEVICE,STATE,CONNECTION dev 2>/dev/null | awk -F: '$1 == "wlan0" { print $2 ":" $3 }')
ip4=$(ip -4 -o addr show wlan0 2>/dev/null | awk '{ print $4 }' | cut -d/ -f1 | head -1)
if [ "${wifi%%:*}" = connected ] && [ -n "$ip4" ]; then report PASS wifi "${wifi#*:}  IP $ip4"
else report WARN wifi "wlan0 ${wifi:-missing}; USB still works: ssh jetson@192.168.55.1"; fi
ps_state=$(iw dev wlan0 get power_save 2>/dev/null | awk '{ print $NF }')
if [ "$ps_state" = off ]; then report PASS "wifi powersave" off
elif [ -n "$ps_state" ]; then report WARN "wifi powersave" "$ps_state (SSH drops; see setup_car.sh)"; fi
report INFO "ROS domain" "ROS_DOMAIN_ID=$(env_get ROS_DOMAIN_ID)"

# --- Devices ------------------------------------------------------------------
section Devices
if [ -e /dev/myserial ]; then report PASS "yahboom board" "/dev/myserial -> $(readlink /dev/myserial)"
else report FAIL "yahboom board" "/dev/myserial missing (USB cable, board power switch, udev rules)"; fi
if [ "$(env_get CRC_LIDAR)" = 0 ]; then report INFO lidar "off (CRC_LIDAR=0)"
elif [ -e /dev/rplidar ]; then report PASS lidar "/dev/rplidar -> $(readlink /dev/rplidar)"
else report FAIL lidar "/dev/rplidar missing (USB cable or adapter board)"; fi
if [ -e /dev/video0 ]; then report PASS "camera sensor" "/dev/video0 (CSI)"
else report FAIL "camera sensor" "no /dev/video0: check the CSI ribbon cable (power off first)"; fi
if systemctl is-active -q crc-camera; then report PASS "camera service" "crc-camera active"
else report FAIL "camera service" "crc-camera $(systemctl is-active crc-camera) (journalctl -u crc-camera)"; fi

# --- Docker and drivers ---------------------------------------------------------
section Drivers
if ! docker info >/dev/null 2>&1; then
  report FAIL docker "cannot reach Docker (user not in the docker group?)"
else
  if docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then report PASS docker "nvidia runtime present"
  else report FAIL docker "nvidia runtime missing (nvidia-docker2)"; fi
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    report PASS image "$IMAGE built $(docker image inspect -f '{{.Created}}' "$IMAGE" | cut -c1-10)"
  else report FAIL image "$IMAGE missing: bash run_car.sh build"; fi
  if running "$BRINGUP"; then
    report PASS "$BRINGUP" "running since $(docker inspect -f '{{.State.StartedAt}}' "$BRINGUP" | cut -c1-19 | tr T ' ') UTC"
    restarts=$(docker logs --since 2m "$BRINGUP" 2>&1 | grep -c 'restarting in')
    if [ "$restarts" -eq 0 ]; then report PASS "driver restarts" "none in the last 2 min"
    else report WARN "driver restarts" "$restarts in the last 2 min: bash run_car.sh bringup-logs"; fi
  else
    report FAIL "$BRINGUP" "not running: bash run_car.sh bringup"
  fi
fi
if [ "$(env_get CRC_YOLO)" != 0 ]; then
  eng=$(env_get CRC_YOLO_ENGINE); eng_host="$ENGINES/$(basename "${eng:-signs.engine}")"
  if [ -f "$eng_host" ]; then
    report PASS "sign engine" "$eng_host ($(du -h "$eng_host" | cut -f1))"
    [ -f "${eng_host%.engine}.labels" ] || report WARN "sign labels" "no ${eng_host%.engine}.labels (classes reported as class_<id>)"
  else
    report WARN "sign engine" "$eng_host missing: sign detector off (build_engine.sh, or CRC_YOLO=0)"
  fi
fi
prog=0
if running "$TEAM_CONTAINER" && docker exec "$TEAM_CONTAINER" test -s /tmp/team_program.pid; then
  prog=1; report INFO "team program" "running in the background (bash run_car.sh stop)"
fi

# --- Sensor topics (inside the driver container) ---------------------------------
if running "$BRINGUP"; then
  section "Sensor topics"
  args=()
  if [ "$MOTION" = 1 ]; then
    if [ "$prog" = 1 ]; then
      report FAIL motion "a team program is running; stop it first: bash run_car.sh stop"
    else
      read -r -p "The wheels will turn for about 6 s. Are they lifted off the ground? [y/N] " ans
      if [[ "$ans" =~ ^[Yy] ]]; then args=(--motion); else report INFO motion "skipped"; fi
    fi
  fi
  # Tab-separated "level check detail" lines; anything else (e.g. a traceback) is a failure
  while IFS=$'\t' read -r level check detail; do
    case "$level" in
      PASS|WARN|FAIL|INFO) report "$level" "$check" "$detail" ;;
      *) report FAIL checker "$level${check:+ $check}${detail:+ $detail}" ;;
    esac
  done < <(docker exec -i "$BRINGUP" bash -c \
             'source /opt/ros/humble/setup.bash && source /opt/crc_ws/install/setup.bash && exec python3 - "$@"' \
             _ "${args[@]}" < "$SELF/car_check_ros.py" 2>&1)
fi

printf '\n%sResult:%s %s%d PASS%s, %s%d WARN%s, %s%d FAIL%s\n' "$B" "$Z" "$G" "$N_PASS" "$Z" "$Y" "$N_WARN" "$Z" "$R" "$N_FAIL" "$Z"
[ "$N_FAIL" -eq 0 ]
