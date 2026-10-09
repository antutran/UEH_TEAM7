#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Post-flash setup of a UEH CRC 2026 car (Jetson Nano, L4T R32.7.5).
# Run once on the Jetson, with Internet access, from the copied package:
#
#   sudo bash ~/crc_car/scripts/setup_car.sh
#
# What it does:
#   1. Holds every installed nvidia-l4t-* package (the apt repo offers 32.7.6;
#      upgrading the userland past the 32.7.5 kernel/dtb would break CUDA).
#   2. Installs the lean GPU stack: CUDA 10.2 runtime libs, cuDNN 8.2,
#      TensorRT 8.2 (+ trtexec) and nvidia-docker2 (~2.1 GB).
#   3. Adds the login user to the docker, dialout, i2c and gpio groups.
#   4. Installs the udev rules (/dev/myserial, /dev/rplidar).
#   5. Builds the crc_car:humble Docker image (drivers included, about 10 minutes).
#   6. Installs the per-car settings (~/crc_car.env) and the crc-camera service,
#      starts the crc_bringup driver container (restarts with Docker at boot).
#   7. Cleans the apt cache and prints a summary.
# ---------------------------------------------------------------------------
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SELF")"
CAR_USER="${SUDO_USER:-jetson}"

die() { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

[ "$(id -u)" = 0 ] || die "run this script with sudo"
grep -q "R32 (release), REVISION: 7" /etc/nv_tegra_release || die "this script targets L4T R32.7 (JetPack 4.6)"
[ "$(findmnt -n -o SOURCE /)" = /dev/mmcblk0p1 ] || echo "Note: root is not on eMMC ($(findmnt -n -o SOURCE /))"
getent hosts repo.download.nvidia.com >/dev/null || die "no Internet/DNS (connect Wi-Fi or Ethernet first)"
# A wrong clock breaks HTTPS certificate checks for apt and docker pull.
# The board has no RTC battery, so give NTP a moment, then insist on a sane year.
if [ "$(date +%Y)" -lt 2025 ]; then
  echo "Clock is $(date); waiting up to 30 s for NTP..."
  timedatectl set-ntp true 2>/dev/null || true
  for i in $(seq 1 30); do [ "$(date +%Y)" -ge 2025 ] && break; sleep 1; done
fi
[ "$(date +%Y)" -ge 2025 ] || die "clock is still wrong ($(date)). Set it from your PC, then rerun:
  ssh -t <user>@<car> \"sudo date -s @\$(date +%s) && sudo hwclock -w\""

step "1/7 Hold nvidia-l4t-* packages"
mapfile -t L4T_PKGS < <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 'nvidia-l4t-*' | awk '$1 ~ /^[hi]i/ {print $2}')
apt-mark hold "${L4T_PKGS[@]}" >/dev/null
echo "held ${#L4T_PKGS[@]} packages"

step "2/7 Install CUDA/cuDNN/TensorRT runtime and nvidia-docker2"
# Use HTTPS for the Ubuntu mirrors: captive-portal networks (e.g. campus Wi-Fi)
# rewrite plain HTTP to a login page, which breaks apt, while HTTPS passes.
for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
  [ -f "$f" ] && grep -q 'http://ports.ubuntu.com' "$f" && \
    sed -i.bak 's#http://ports.ubuntu.com#https://ports.ubuntu.com#g' "$f"
done
apt-get update -qq
apt-get install -y --no-install-recommends \
  libnvinfer8 libnvinfer-plugin8 libnvonnxparsers8 libnvparsers8 libnvinfer-bin \
  libcudnn8 cuda-cudart-10-2 cuda-libraries-10-2 nvidia-docker2
systemctl restart docker

step "3/7 Groups for $CAR_USER"
for g in docker dialout i2c gpio; do
  getent group "$g" >/dev/null && usermod -aG "$g" "$CAR_USER"
done
echo "$CAR_USER: $(id -nG "$CAR_USER")"

step "4/7 udev rules"
bash "$SELF/install_udev.sh"

step "5/7 Build the crc_car:humble image"
# Some captive-portal networks let wget through but reset dockerd's registry
# connections (pull fails with EOF). The base image can then be copied from a PC.
BASE=ros:humble-ros-base
if ! docker image inspect "$BASE" >/dev/null 2>&1 && ! docker pull "$BASE"; then
  die "cannot pull $BASE on this network. Copy it from a PC that can, then rerun:
  docker pull --platform linux/arm64 $BASE && docker tag $BASE crc/ros-humble-base:arm64
  docker save --platform linux/arm64 crc/ros-humble-base:arm64 | gzip -1 | ssh <user>@<car> 'gunzip | docker load && docker tag crc/ros-humble-base:arm64 $BASE'"
fi
docker build -t crc_car:humble -f "$ROOT/docker/Dockerfile" "$ROOT"

step "6/7 Car settings, camera service and drivers"
# Wi-Fi power save makes the Intel 8265 sleep through ARP requests: the car then cannot be
# reached (ping/ssh/ROS) until it sends something itself. Disable it for good and right now.
# conf.d is read in alphabetical order and the last value wins: the file must sort after
# Ubuntu's default-wifi-powersave-on.conf.
rm -f /etc/NetworkManager/conf.d/99-crc-wifi-powersave-off.conf
cat > /etc/NetworkManager/conf.d/zz-crc-wifi-powersave-off.conf <<'NMCONF'
[connection]
wifi.powersave = 2
NMCONF
iw dev wlan0 set power_save off 2>/dev/null || true
echo "Wi-Fi power save: $(iw dev wlan0 get power_save 2>/dev/null | awk '{print $3}')"
HOME_DIR="$(getent passwd "$CAR_USER" | cut -d: -f6)"
ENV_FILE="$HOME_DIR/crc_car.env"
if [ ! -f "$ENV_FILE" ]; then
  install -m 0644 -o "$CAR_USER" -g "$CAR_USER" "$ROOT/config/crc_car.env" "$ENV_FILE"
  echo "created $ENV_FILE (edit it for this car)"
else
  echo "keeping existing $ENV_FILE"
fi
chmod +x "$ROOT/scripts/camera_stream.sh"
cat > /etc/systemd/system/crc-camera.service <<UNIT
[Unit]
Description=UEH CRC camera stream (CSI IMX219 -> raw RGBA on 127.0.0.1:5600)
After=nvargus-daemon.service
Wants=nvargus-daemon.service

[Service]
User=$CAR_USER
Environment=CRC_ENV=$ENV_FILE
# Run through bash: cloud-synced copies of the script can lose the executable bit
ExecStart=/bin/bash $ROOT/scripts/camera_stream.sh
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now crc-camera.service
systemctl restart crc-camera.service
sudo -u "$CAR_USER" CRC_ENV="$ENV_FILE" bash "$ROOT/scripts/run_car.sh" bringup

step "7/7 Clean up"
apt-get clean
mkdir -p "$HOME_DIR/crc_ws/src" "$HOME_DIR/crc_engines"
chown "$CAR_USER:$CAR_USER" "$HOME_DIR/crc_ws" "$HOME_DIR/crc_ws/src" "$HOME_DIR/crc_engines"

echo
echo "=== Summary ==="
echo "L4T      : $(head -1 /etc/nv_tegra_release | cut -d, -f1,2)"
echo "TensorRT : $(dpkg-query -W -f='${Version}' libnvinfer8)"
echo "Docker   : $(docker --version | cut -d, -f1), runtimes: $(docker info 2>/dev/null | awk -F': ' '/Runtimes/{print $2}')"
echo "Image    : $(docker images crc_car:humble --format '{{.Repository}}:{{.Tag}} {{.Size}}')"
echo "Disk     : $(df -h --output=avail / | tail -1 | tr -d ' ') free on /"
echo
echo "Camera   : $(systemctl is-active crc-camera)"
echo "Drivers  : $(docker inspect -f '{{.State.Status}}' crc_bringup 2>/dev/null || echo missing)"
echo
echo "Log out and back in (or reboot) so the new groups apply, then:"
echo "  bash ~/crc_car/scripts/run_car.sh status"
