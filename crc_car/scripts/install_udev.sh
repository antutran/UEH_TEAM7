#!/usr/bin/env bash
# Install the CRC car udev rules (/dev/myserial, /dev/rplidar) and apply them
# without a reboot. Usage: sudo bash install_udev.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "Run this script with sudo"; exit 1; }
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES="$SELF/../udev/99-crc-car.rules"
[ -f "$RULES" ] || { echo "Rules file not found: $RULES"; exit 1; }

# Replace the older single-device rule from the bench test, if present
rm -f /etc/udev/rules.d/99-yahboom.rules
install -m 0644 "$RULES" /etc/udev/rules.d/99-crc-car.rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty --action=add
udevadm settle

for link in /dev/myserial /dev/rplidar; do
  if [ -e "$link" ]; then
    echo "$link -> $(readlink -f "$link")"
  else
    echo "$link: not present (device unplugged?)"
  fi
done
