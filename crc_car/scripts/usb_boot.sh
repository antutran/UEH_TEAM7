#!/bin/bash
# Move the root filesystem from eMMC to a USB drive (Waveshare JETSON-NANO-DEV-KIT).
# Creates a GPT with one ext4 partition and boots with root=PARTUUID=...
# (supported by the L4T 32.7.x initrd).
# Usage: sudo bash ~/usb_boot.sh
set -euo pipefail
CONF=/boot/extlinux/extlinux.conf
LINUX_FS_GUID=0FC63DAF-8483-4772-8E79-3D69D8477DE4

[ "$(id -u)" = 0 ] || { echo "Run this script with sudo"; exit 1; }
[ "$(findmnt -n -o SOURCE /)" = /dev/mmcblk0p1 ] || { echo "System is not running from eMMC (mmcblk0p1) - aborting"; exit 1; }
grep -q 'root=/dev/mmcblk0p1' "$CONF" || { echo "$CONF has no root=/dev/mmcblk0p1 - aborting"; exit 1; }

# Exactly one USB disk must be attached
mapfile -t USBS < <(lsblk -dn -o NAME,TRAN | awk '$2=="usb"{print $1}')
[ "${#USBS[@]}" -eq 1 ] || { echo "Attach exactly one USB disk (found ${#USBS[@]}: ${USBS[*]:-none})"; exit 1; }
DEV=/dev/${USBS[0]}

echo "USB disk: $DEV  $(lsblk -dn -o SIZE,VENDOR,MODEL $DEV)"
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT "$DEV"
echo "ALL data on this disk will be ERASED. Type YES to continue:"
read -r ans; [ "$ans" = YES ] || { echo "Cancelled."; exit 1; }

# Hold kernel/dtb/bootloader packages so apt upgrade cannot replace the modified dtb
apt-mark hold nvidia-l4t-kernel nvidia-l4t-kernel-dtbs nvidia-l4t-bootloader

# Unmount, wipe old signatures, create a GPT with one Linux partition spanning the disk
for p in "$DEV"?*; do [ -b "$p" ] && umount "$p" 2>/dev/null || true; done
for p in "$DEV"?*; do [ -b "$p" ] && wipefs -a "$p" >/dev/null || true; done
wipefs -a "$DEV"
printf 'label: gpt\n,,%s\n' "$LINUX_FS_GUID" | sfdisk --no-reread "$DEV"
blockdev --rereadpt "$DEV" || partprobe "$DEV" || true
udevadm settle
PART=${DEV}1
for i in $(seq 1 20); do [ -b "$PART" ] && break; sleep 0.5; done
[ -b "$PART" ] || { echo "$PART did not appear after partitioning"; exit 1; }
mkfs.ext4 -F -L jetson-usb "$PART"
udevadm settle
PARTUUID=$(blkid -s PARTUUID -o value "$PART")
[[ "$PARTUUID" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || { echo "Invalid PARTUUID: '$PARTUUID'"; exit 1; }
echo "PARTUUID = $PARTUUID"

cp "$CONF" "$CONF.bak"
mount "$PART" /mnt
echo "Copying the system to the USB disk (a few minutes, no output)..."
cp -ax / /mnt
sync

# Point root at the USB disk only AFTER the copy succeeded (on failure the board still boots from eMMC)
sed -i "s#root=/dev/mmcblk0p1#root=PARTUUID=$PARTUUID#" "$CONF" "/mnt$CONF"
grep -q "root=PARTUUID=$PARTUUID " "$CONF" || { cp "$CONF.bak" "$CONF"; echo "Failed to update $CONF, restored the backup"; exit 1; }
umount /mnt
grep APPEND "$CONF" | grep -v '^ *#'
echo "DONE. Reboot with: sudo reboot"
