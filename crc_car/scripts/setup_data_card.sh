#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Format the TF card as the car's data disk, mounted at /data, so datasets,
# models, logs and rosbags do not fill the 14 GB eMMC. The OS, Docker images and
# TensorRT engines stay on the eMMC, which reads ~10x faster than the TF slot.
#
#   sudo bash ~/crc_car/scripts/setup_data_card.sh
#
# Layout and options:
#   - GPT with one ext4 partition labelled crc-data, no reserved blocks (-m 0),
#     one inode per 64 KiB (enough for image datasets, small inode tables),
#     inode tables written now so the card does no background init later.
#   - /etc/fstab by UUID: noatime (reads cause no writes), nofail +
#     x-systemd.device-timeout=5s (the car still boots without the card).
#     The journal is kept: cars lose power abruptly.
#   - /data/{datasets,models,logs,bags} owned by the login user.
#   - Bounded log growth on the eMMC: Docker json logs 3 x 10 MB per container,
#     systemd journal 100 MB.
# ---------------------------------------------------------------------------
set -euo pipefail

DEV=/dev/mmcblk1
PART=${DEV}p1
MNT=/data
LABEL=crc-data
CAR_USER="${SUDO_USER:-jetson}"

die() { echo "ERROR: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die "run this script with sudo"
[ -b "$DEV" ] || die "$DEV (TF card) not found"
case "$(findmnt -n -o SOURCE /)" in "$DEV"*) die "the system runs from the TF card; refusing to format it";; esac

echo "TF card: $DEV $(lsblk -dno SIZE "$DEV")"
lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT "$DEV"
# Show what is on the card before it is erased (read-only mount of ext4/vfat partitions)
PEEK=$(mktemp -d)
for p in "$DEV"p*; do
  [ -b "$p" ] || continue
  case "$(blkid -o value -s TYPE "$p" 2>/dev/null)" in
    ext4|ext3|ext2|vfat|exfat)
      if mount -o ro "$p" "$PEEK" 2>/dev/null; then
        echo "--- $p: $(df -h --output=used "$PEEK" | tail -1 | tr -d ' ') used; top level:"
        ls -A "$PEEK" | head -20 | sed 's/^/    /'
        [ -d "$PEEK/home" ] && echo "    home: $(ls -A "$PEEK/home" | tr '\n' ' ')"
        umount "$PEEK"
      fi
      ;;
  esac
done
rmdir "$PEEK"
echo "ALL data on this card will be ERASED. Type YES to continue:"
read -r ans; [ "$ans" = YES ] || { echo "Cancelled."; exit 1; }

echo "Unmounting and wiping old partitions..."
grep -q "^$DEV" /proc/swaps && swapoff "$(awk -v d="$DEV" '$1 ~ "^"d {print $1}' /proc/swaps)" || true
for p in "$DEV"p*; do [ -b "$p" ] && umount "$p" 2>/dev/null || true; done
umount "$MNT" 2>/dev/null || true
for p in "$DEV"p*; do [ -b "$p" ] && wipefs -a "$p" >/dev/null || true; done
wipefs -a "$DEV"
printf 'label: gpt\n,,0FC63DAF-8483-4772-8E79-3D69D8477DE4\n' | sfdisk --no-reread "$DEV"
blockdev --rereadpt "$DEV" || partprobe "$DEV" || true
udevadm settle
for i in $(seq 1 20); do [ -b "$PART" ] && break; sleep 0.5; done
[ -b "$PART" ] || die "$PART did not appear after partitioning"

echo "Formatting $PART as ext4 (writes the inode tables now, about a minute)..."
mkfs.ext4 -F -L "$LABEL" -m 0 -i 65536 -E lazy_itable_init=0,lazy_journal_init=0 "$PART"
UUID=$(blkid -s UUID -o value "$PART")
[ -n "$UUID" ] || die "could not read the UUID of $PART"

echo "Configuring /etc/fstab and mounting $MNT..."
sed -i "\#[[:space:]]$MNT[[:space:]]#d" /etc/fstab
echo "UUID=$UUID $MNT ext4 defaults,noatime,nofail,x-systemd.device-timeout=5s 0 2" >> /etc/fstab
mkdir -p "$MNT"
mount "$MNT"
mkdir -p "$MNT"/{datasets,models,logs,bags}
chown -R "$CAR_USER:$CAR_USER" "$MNT"

echo "Bounding log growth on the eMMC..."
python3 - <<'PY'
import json
path = '/etc/docker/daemon.json'
try:
    cfg = json.load(open(path))
except (FileNotFoundError, ValueError):
    cfg = {}
cfg['log-driver'] = 'json-file'
cfg['log-opts'] = {'max-size': '10m', 'max-file': '3'}
json.dump(cfg, open(path, 'w'), indent=4)
PY
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=100M\n' > /etc/systemd/journald.conf.d/99-crc-size.conf
systemctl restart systemd-journald
# The log limits apply to containers created after the daemon restart: recreate the drivers
systemctl restart docker
if docker image inspect crc_car:humble >/dev/null 2>&1; then
  sudo -u "$CAR_USER" bash "$(dirname "$0")/run_car.sh" bringup
fi

echo
echo "DONE: $(df -h --output=source,size,avail,target "$MNT" | tail -1)"
ls -la "$MNT"
