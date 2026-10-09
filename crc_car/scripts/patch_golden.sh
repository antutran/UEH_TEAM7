#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Update the golden image in place with the current jetson_ros2 checkout, so every car
# flashed from it is complete without a second step (no update_car.sh, no re-read of the
# golden car). Run on the PC:
#
#   rsync -a --delete <this jetson_ros2 folder>/ ~/jetson/crc_car/
#        (root cannot read the cloud-drive folder, so the script runs from that copy), then:
#   sudo bash ~/jetson/crc_car/scripts/patch_golden.sh      # image dir: ~/jetson/golden
#
# What it changes inside the image (raw file, loop-mounted):
#   /home/jetson/crc_car       <- this checkout (scripts, examples/crc_team, docs, ...)
#   /home/jetson/.bashrc       <- team shortcuts `car` and `car-check` (once)
#   /home/jetson/crc_car/GOLDEN_VERSION  <- date of this patch
# Then it rebuilds the sparse golden_app.img that flash_car.sh writes to the cars.
# Docker images and the per-car settings (~/crc_car.env) are not touched.
# ---------------------------------------------------------------------------
set -euo pipefail

HOST_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
OUT="${OUT:-$HOST_HOME/jetson/golden}"
L4T="${L4T:-$HOST_HOME/jetson/Linux_for_Tegra}"
RAW="$OUT/golden_app.img.raw"
IMG="$OUT/golden_app.img"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MNT=""

die() { echo "ERROR: $*" >&2; exit 1; }
cleanup() { [ -n "$MNT" ] && { mountpoint -q "$MNT" && umount "$MNT"; rmdir "$MNT"; }; true; }
trap cleanup EXIT

[ "$(id -u)" = 0 ] || die "run this script with sudo"
cd /   # the caller's directory may be a cloud drive that root cannot read
[ -s "$RAW" ] || die "raw golden image not found: $RAW (made by clone_golden.sh)"
[ -x "$L4T/bootloader/mksparse" ] || die "mksparse not found in $L4T/bootloader"
pgrep -x flash.sh >/dev/null && die "flash.sh is running: wait until the current car is flashed"
losetup -j "$RAW" | grep -q . && die "$RAW is already attached to a loop device"

MNT="$(mktemp -d /tmp/golden.XXXX)"
mount -o loop "$RAW" "$MNT"
HOME_IMG="$MNT/home/jetson"
[ -d "$HOME_IMG" ] || die "no /home/jetson in the image"

echo "Copying $ROOT -> /home/jetson/crc_car in the image..."
find "$ROOT" -name __pycache__ -prune -exec rm -rf {} +
rsync -a --delete --exclude __pycache__ --exclude 'docs/*.pdf' --chown=1000:1000 \
  --chmod=D755,F644 "$ROOT/" "$HOME_IMG/crc_car/"
chmod 755 "$HOME_IMG"/crc_car/scripts/*.sh "$HOME_IMG"/crc_car/scripts/*.py
date '+%Y-%m-%d %H:%M golden patch' > "$HOME_IMG/crc_car/GOLDEN_VERSION"
chown 1000:1000 "$HOME_IMG/crc_car/GOLDEN_VERSION"

if ! grep -q "alias car=" "$HOME_IMG/.bashrc"; then
  cat >> "$HOME_IMG/.bashrc" <<'ALIASES'

# UEH CRC 2026 team shortcuts
alias car='bash ~/crc_car/scripts/run_car.sh'
alias car-check='bash ~/crc_car/scripts/car_check.sh'
ALIASES
  echo "Added the car / car-check shortcuts to .bashrc"
fi

# Identity must stay empty in the master (regenerated on each car's first boot)
[ -s "$MNT/etc/machine-id" ] && die "machine-id is set in the image: run prepare_golden.sh again"
ls "$MNT"/etc/ssh/ssh_host_* >/dev/null 2>&1 && die "SSH host keys found in the image"

sync
umount "$MNT"
e2fsck -fn "$RAW" >/dev/null || die "file system check failed on $RAW"

echo "Rebuilding the sparse image (a few minutes)..."
( cd "$L4T/bootloader" && ./mksparse --fillpattern=0 "$RAW" "$IMG.new" ) >/dev/null
mv -f "$IMG.new" "$IMG"
chown "${SUDO_USER:-root}:" "$IMG" "$RAW"
echo "Done: $IMG ($(du -h "$IMG" | cut -f1))"
echo "Flash the cars as before: sudo IMAGE=$IMG bash flash_car.sh"
