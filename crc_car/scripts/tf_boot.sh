#!/bin/bash
# Move the root filesystem from eMMC to the TF card
# (Waveshare JETSON-NANO-DEV-KIT, method 1 of the Waveshare wiki).
# Usage: sudo bash ~/tf_boot.sh
set -euo pipefail
DEV=/dev/mmcblk1
CONF=/boot/extlinux/extlinux.conf

[ "$(id -u)" = 0 ] || { echo "Run this script with sudo"; exit 1; }
[ -b "$DEV" ] || { echo "$DEV (TF card) not found"; exit 1; }
[ "$(findmnt -n -o SOURCE /)" = /dev/mmcblk0p1 ] || { echo "System is not running from eMMC (mmcblk0p1) - aborting"; exit 1; }
grep -q 'root=/dev/mmcblk0p1' "$CONF" || { echo "$CONF has no root=/dev/mmcblk0p1 - aborting"; exit 1; }

echo "TF card: $DEV $(lsblk -dno SIZE $DEV)"
echo "ALL data on this card will be ERASED. Type YES to continue:"
read -r ans; [ "$ans" = YES ] || { echo "Cancelled."; exit 1; }

# Hold kernel/dtb/bootloader packages so apt upgrade cannot replace the TF-enabled dtb
apt-mark hold nvidia-l4t-kernel nvidia-l4t-kernel-dtbs nvidia-l4t-bootloader

# Unmount, wipe the old partition table (including the backup GPT; the kernel
# runs with the `gpt` option and would otherwise resurrect old partitions),
# then format the whole card as ext4
for p in "$DEV"p*; do umount "$p" 2>/dev/null || true; done
wipefs -a "$DEV"
blockdev --rereadpt "$DEV" || true
mkfs.ext4 -F -L jetson-tf "$DEV"

cp "$CONF" "$CONF.bak"
mount "$DEV" /mnt
echo "Copying the system to the card (5-15 minutes, no output)..."
cp -ax / /mnt
sync

# Point root at the card only AFTER the copy succeeded (on failure the board still boots from eMMC)
sed -i 's#root=/dev/mmcblk0p1#root=/dev/mmcblk1#' "$CONF" /mnt$CONF
grep -q 'root=/dev/mmcblk1 ' "$CONF" || { cp "$CONF.bak" "$CONF"; echo "Failed to update $CONF, restored the backup"; exit 1; }
umount /mnt
grep APPEND "$CONF" | grep -v '^#'
echo "DONE. Reboot with: sudo reboot"
