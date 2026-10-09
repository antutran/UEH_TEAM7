#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Flash a Waveshare JETSON-NANO-DEV-KIT (Jetson Nano eMMC module, p3448-0002)
# with L4T R32.7.5 + the PCN211181 overlay from ~/jetson/Linux_for_Tegra.
# A default user is baked into the image, so the board boots straight to the
# desktop with autologin and no first-boot setup wizard.
#
# Put the board in recovery mode first (jumper FC REC to GND, power on,
# Micro-USB cable to this PC), then:
#
#   sudo bash flash_car.sh             # first board: builds system.img (~12 min) and flashes
#   sudo REUSE=1 bash flash_car.sh     # further boards: reuse system.img (~4 min)
#   sudo WIFI_SSID="My Net" bash flash_car.sh                 # bake in an open Wi-Fi network
#   sudo WIFI_SSID="My Net" WIFI_PSK="secret" bash flash_car.sh  # ... or a WPA2-PSK one
#   sudo IMAGE=~/jetson/golden/golden_app.img bash flash_car.sh  # clone a golden car
#                                    # (image from clone_golden.sh; rootfs is not touched)
#
# Without a monitor the board is also reachable over the Micro-USB cable
# (L4T USB device mode): ssh jetson@192.168.55.1
#
# Overrides: L4T (BSP directory), CAR_USER, CAR_PASS, CAR_HOST, WIFI_SSID, WIFI_PSK, IMAGE
# ---------------------------------------------------------------------------
set -euo pipefail

HOST_HOME="$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)"
L4T="${L4T:-$HOST_HOME/jetson/Linux_for_Tegra}"
CAR_USER="${CAR_USER:-jetson}"
CAR_PASS="${CAR_PASS:-jetson}"
CAR_HOST="${CAR_HOST:-jetson-desktop}"
REUSE="${REUSE:-0}"
WIFI_SSID="${WIFI_SSID:-}"
WIFI_PSK="${WIFI_PSK:-}"
IMAGE="${IMAGE:-}"
LOG="$L4T/../flash_$(date +%Y%m%d_%H%M%S).log"

die() { echo "ERROR: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "run this script with sudo"
[ -x "$L4T/flash.sh" ] || die "flash.sh not found in $L4T"
[ -z "$IMAGE" ] || [ -s "$IMAGE" ] || die "IMAGE file not found: $IMAGE"

# --- 1. Exactly one Jetson in recovery mode ----------------------------------
n=$(lsusb -d 0955:7f21 | wc -l)
[ "$n" -eq 1 ] || die "expected exactly one Jetson Nano in recovery mode (0955:7f21), found $n"

# --- 2. Read the module EEPROM: must be a Nano eMMC module (3448 / SKU 0002) --
echo "Reading module EEPROM..."
(
  cd "$L4T/bootloader"
  rm -f cvm.bin
  ./tegraflash.py --chip 0x21 --applet nvtboot_recovery.bin --skipuid \
    --cmd "dump eeprom boardinfo cvm.bin" > "$LOG" 2>&1
) || die "could not read the EEPROM (see $LOG)"
board_id=$(cd "$L4T/bootloader" && ./chkbdinfo -i cvm.bin | tr -d ' ')
board_sku=$(cd "$L4T/bootloader" && ./chkbdinfo -k cvm.bin | tr -d ' ')
board_ver=$(cd "$L4T/bootloader" && ./chkbdinfo -f cvm.bin | tr -d ' ')
echo "Module: id=$board_id sku=$board_sku version=$board_ver"
[ "$board_id" = 3448 ] || die "not a Jetson Nano module (id $board_id)"
[ "$board_sku" = 0002 ] || die "SKU $board_sku is not the eMMC module (0002); an SD-card module is flashed by writing an SD image instead"
[[ "$board_ver" < 300 ]] && echo "Note: module version < 300 uses the a02 dtb (stock, TF slot on the baseboard stays disabled)"

# The EEPROM read reboots the board back into recovery; wait for it
for i in $(seq 1 30); do lsusb -d 0955:7f21 >/dev/null && break; sleep 1; done
lsusb -d 0955:7f21 >/dev/null || die "board did not come back in recovery mode"

# --- 3a. Clone mode: install the golden image as system.img -------------------
cd "$L4T"
if [ -n "$IMAGE" ]; then
  # Copy only when the golden image changed since the last copy (it is ~9 GB)
  stamp="$(stat -c '%s %Y' "$IMAGE") $(readlink -f "$IMAGE")"
  if [ "$(cat bootloader/system.img.source 2>/dev/null)" != "$stamp" ]; then
    echo "Installing golden image $IMAGE as system.img..."
    cp "$IMAGE" bootloader/system.img
    echo "$stamp" > bootloader/system.img.source
  else
    echo "system.img already holds $IMAGE"
  fi
  REUSE=1
else
  rm -f bootloader/system.img.source   # system.img will be rebuilt from rootfs/
fi

# --- 3b. Default user in the rootfs (skips oem-config) ------------------------
if [ -z "$IMAGE" ]; then
# Run from the BSP directory: the NVIDIA script ends with popd, which fails if
# the caller's directory is not readable by root (e.g. a FUSE-mounted cloud drive).
cd "$L4T"
if grep -q "^${CAR_USER}:" rootfs/etc/passwd; then
  echo "User '$CAR_USER' already exists in the rootfs"
else
  echo "Creating user '$CAR_USER' (hostname $CAR_HOST, autologin) in the rootfs"
  tools/l4t_create_default_user.sh -u "$CAR_USER" -p "$CAR_PASS" -n "$CAR_HOST" -a --accept-license
  REUSE=0   # rootfs changed, system.img must be rebuilt
fi
# Make sure the hostname is set (an interrupted user-creation run can leave it empty)
if [ ! -s rootfs/etc/hostname ]; then
  echo "$CAR_HOST" > rootfs/etc/hostname
  grep -qFx "127.0.1.1 $CAR_HOST" rootfs/etc/hosts || echo "127.0.1.1 $CAR_HOST" >> rootfs/etc/hosts
  REUSE=0
fi
echo "Rootfs hostname: $(cat rootfs/etc/hostname)"

# Per-board identity, generated on first boot so every flashed car is unique.
# Normally oem-config does this; the default-user path skips oem-config.
#  - SSH host keys: the sample rootfs ships none, so sshd would refuse all logins.
#  - machine-id: the sample rootfs ships a fixed one; an empty file makes systemd
#    create a fresh id on first boot (shared ids can give duplicate DHCP leases).
UNIT=rootfs/etc/systemd/system/crc-ssh-hostkeys.service
if [ ! -f "$UNIT" ]; then
  cat > "$UNIT" <<'UNIT_EOF'
[Unit]
Description=Generate missing SSH host keys (UEH CRC first boot)
Before=ssh.service
ConditionPathExists=!/etc/ssh/ssh_host_ed25519_key

[Service]
Type=oneshot
ExecStart=/usr/bin/ssh-keygen -A

[Install]
WantedBy=multi-user.target
UNIT_EOF
  ln -sf /etc/systemd/system/crc-ssh-hostkeys.service \
    rootfs/etc/systemd/system/multi-user.target.wants/crc-ssh-hostkeys.service
  REUSE=0
fi
if [ -s rootfs/etc/machine-id ]; then
  : > rootfs/etc/machine-id
  REUSE=0
fi
rm -f rootfs/etc/ssh/ssh_host_*   # never ship host keys inside the image

# Wi-Fi power save makes the Intel 8265 sleep through ARP requests, so the car is
# unreachable until it transmits; ship it disabled. NetworkManager reads conf.d in
# alphabetical order and the last value wins, so the name must sort after Ubuntu's
# default-wifi-powersave-on.conf (a "99-" prefix does not: digits sort before letters).
rm -f rootfs/etc/NetworkManager/conf.d/99-crc-wifi-powersave-off.conf
PS_FILE=rootfs/etc/NetworkManager/conf.d/zz-crc-wifi-powersave-off.conf
if [ ! -f "$PS_FILE" ]; then
  printf '[connection]\nwifi.powersave = 2\n' > "$PS_FILE"
  REUSE=0
fi

# Optional Wi-Fi profile baked into the image, so the board joins the network on
# first boot without a monitor. Open network: WIFI_SSID only; WPA2-PSK: add WIFI_PSK.
if [ -n "$WIFI_SSID" ]; then
  NM_DIR=rootfs/etc/NetworkManager/system-connections
  NM_FILE="$NM_DIR/$WIFI_SSID"
  NM_TMP=$(mktemp)
  {
    echo "[connection]"
    echo "id=$WIFI_SSID"
    echo "uuid=$(cat /proc/sys/kernel/random/uuid)"
    echo "type=wifi"
    echo "autoconnect=true"
    echo "permissions="
    echo
    echo "[wifi]"
    echo "mode=infrastructure"
    echo "ssid=$WIFI_SSID"
    if [ -n "$WIFI_PSK" ]; then
      echo
      echo "[wifi-security]"
      echo "key-mgmt=wpa-psk"
      echo "psk=$WIFI_PSK"
    fi
    echo
    echo "[ipv4]"
    echo "method=auto"
    echo
    echo "[ipv6]"
    echo "method=auto"
  } > "$NM_TMP"
  # Rebuild system.img only if the profile content (ignoring the uuid) changed
  if ! diff -q <(grep -v '^uuid=' "$NM_TMP") <(grep -v '^uuid=' "$NM_FILE" 2>/dev/null) >/dev/null; then
    mkdir -p "$NM_DIR"
    install -m 0600 -o root -g root "$NM_TMP" "$NM_FILE"
    REUSE=0
  fi
  rm -f "$NM_TMP"
  echo "Wi-Fi profile in image: $WIFI_SSID ($([ -n "$WIFI_PSK" ] && echo WPA2-PSK || echo open))"
fi

# The overlay tarball once changed these owners; the target root must be root-owned
chown root:root "$L4T/rootfs" "$L4T/rootfs/boot"

fi  # end of rootfs preparation (skipped in clone mode)

# --- 4. Flash ----------------------------------------------------------------
cd "$L4T"
if [ "$REUSE" = 1 ] && [ -f bootloader/system.img ]; then
  echo "Flashing with the existing system.img (REUSE=1)..."
  ./flash.sh -r jetson-nano-emmc mmcblk0p1 2>&1 | tee -a "$LOG" || true
else
  echo "Building system.img and flashing (about 15 minutes)..."
  ./flash.sh jetson-nano-emmc mmcblk0p1 2>&1 | tee -a "$LOG" || true
fi

if grep -q "flashed successfully" "$LOG"; then
  dtb=$(grep -o "copying dtbfile([^)]*)" "$LOG" | tail -1 | sed 's#.*/##; s#)##')
  echo
  echo "SUCCESS. Module version $board_ver, dtb $dtb"
  echo "Remove the FC REC jumper and the USB cable, then power-cycle the board."
  echo "It boots to the desktop as '$CAR_USER' (password '$CAR_PASS'). Log: $LOG"
else
  die "flashing failed, see $LOG"
fi
