#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Turn a fully set-up car into the "golden" master before reading its eMMC back
# with clone_golden.sh. Run on the car:
#
#   sudo bash ~/crc_car/scripts/prepare_golden.sh
#
# Keeps: GPU libraries, the crc_car:humble image and the crc_bringup container (it
#        starts the drivers on every boot), ~/crc_car, ~/crc_car.env, ~/crc_engines,
#        Wi-Fi profiles, authorized SSH keys.
# Removes: team/test containers and build volumes, the image build cache (crc_car:build,
#          ~1.4 GB; the next `run_car.sh build` recreates it), the team sources in
#          ~/crc_ws/src, apt cache, logs, shell history, DHCP leases, SSH host keys and the
#          machine-id (regenerated on each clone's first boot by crc-ssh-hostkeys.service
#          and systemd).
# Fixes: Wi-Fi power save conf name (older setups used a name that did not take effect).
# The board powers off at the end; put it in recovery mode for clone_golden.sh.
# ---------------------------------------------------------------------------
set -euo pipefail

CAR_USER="${SUDO_USER:-jetson}"
HOME_DIR="$(getent passwd "$CAR_USER" | cut -d: -f6)"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run this script with sudo"
[ -f /etc/systemd/system/crc-ssh-hostkeys.service ] || \
  die "crc-ssh-hostkeys.service missing: clones would boot without SSH host keys"
docker image inspect crc_car:humble >/dev/null 2>&1 || die "crc_car:humble image missing (run setup_car.sh first)"
[ "$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' crc_bringup 2>/dev/null)" = unless-stopped ] || \
  die "crc_bringup container missing: run 'bash run_car.sh bringup' as $CAR_USER first"

echo "This removes per-board identity and powers the board off. Type YES to continue:"
read -r ans; [ "$ans" = YES ] || { echo "Cancelled."; exit 1; }

echo "Removing team/test containers, build volumes and the image build cache..."
# Everything except the drivers: crc_bringup must stay so every clone starts them at boot
docker ps -a --format '{{.Names}}' | { grep -vx crc_bringup || true; } | xargs -r docker rm -f >/dev/null
docker volume ls -q | grep '^crc_build_' | xargs -r docker volume rm >/dev/null || true
docker image rm crc_car:build >/dev/null 2>&1 || true
docker image prune -f >/dev/null
rm -rf "$HOME_DIR/crc_ws/src"/* "$HOME_DIR/crc_ws/logs"

echo "Wi-Fi power save off (conf name must sort after default-wifi-powersave-on.conf)..."
rm -f /etc/NetworkManager/conf.d/99-crc-wifi-powersave-off.conf
printf '[connection]\nwifi.powersave = 2\n' > /etc/NetworkManager/conf.d/zz-crc-wifi-powersave-off.conf

echo "Cleaning caches, logs and history..."
apt-get clean
rm -rf "$HOME_DIR/trt_test" "$HOME_DIR/crc_build.log" "$HOME_DIR/.cache"/*
rm -f "$HOME_DIR/.bash_history" /root/.bash_history
journalctl --rotate >/dev/null 2>&1 || true
journalctl --vacuum-time=1s >/dev/null 2>&1 || true
find /var/log -type f \( -name '*.gz' -o -name '*.[0-9]' \) -delete
find /var/log -type f -exec truncate -s 0 {} +
rm -f /var/lib/NetworkManager/*.lease /var/lib/dhcp/*.leases
rm -rf /tmp/* /var/tmp/*

echo "Removing per-board identity..."
rm -f /etc/ssh/ssh_host_*
: > /etc/machine-id

echo "Disk usage now: $(df -h --output=used / | tail -1 | tr -d ' ') used on /"
echo "Powering off. Next: jumper FC REC to GND, power on, Micro-USB to the PC,"
echo "then on the PC: sudo bash clone_golden.sh"
sync
poweroff
