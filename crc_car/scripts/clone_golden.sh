#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Read the APP (root filesystem) partition of the golden car back to the PC.
# Run prepare_golden.sh on the car first, then put it in recovery mode
# (jumper FC REC to GND, power on, Micro-USB to this PC) and run:
#
#   sudo bash clone_golden.sh            # writes ~/jetson/golden/golden_app.img
#
# Then flash every other car with:
#   sudo IMAGE=~/jetson/golden/golden_app.img bash flash_car.sh
#
# Overrides: L4T (BSP directory), OUT (output directory)
# ---------------------------------------------------------------------------
set -euo pipefail

HOST_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
L4T="${L4T:-$HOST_HOME/jetson/Linux_for_Tegra}"
OUT="${OUT:-$HOST_HOME/jetson/golden}"
IMG="$OUT/golden_app.img"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run this script with sudo"
[ -x "$L4T/flash.sh" ] || die "flash.sh not found in $L4T"
n=$(lsusb -d 0955:7f21 | wc -l)
[ "$n" -eq 1 ] || die "expected exactly one Jetson Nano in recovery mode (0955:7f21), found $n"
avail=$(df --output=avail -B1G "$(dirname "$OUT")" | tail -1)
[ "$avail" -ge 35 ] || die "need about 35 GB free for the raw and sparse images, have ${avail} GB"

mkdir -p "$OUT"
rm -f "$IMG" "$IMG.raw"
cd "$L4T"
echo "Reading the APP partition (14 GiB over USB at ~3.6 MB/s, about 70 minutes)..."
./flash.sh -r -k APP -G "$IMG" jetson-nano-emmc mmcblk0p1 2>&1 | tee "$OUT/readback.log" || true

[ -s "$IMG" ] || die "readback failed, see $OUT/readback.log"
chown -R "${SUDO_USER:-root}:" "$OUT"
echo
echo "Golden image ready:"
ls -lh "$IMG" "$IMG.raw" 2>/dev/null
echo "The .raw file is only needed for inspection; delete it to save 14 GB."
echo "Flash the other cars with: sudo IMAGE=$IMG bash flash_car.sh"
