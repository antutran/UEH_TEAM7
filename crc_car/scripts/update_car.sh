#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bring a car's tools up to date from this PC, without re-flashing. Run on the PC:
#
#   bash update_car.sh [car-address]          # default 192.168.55.1 (USB cable)
#
# Copies this jetson_ros2 checkout to ~/crc_car on the car (scripts, examples, docs;
# the per-car settings in ~/crc_car.env are not touched) and adds the team shortcuts
# `car` and `car-check` to ~/.bashrc. Docker images are not rebuilt: if the driver code
# changed, run `bash ~/crc_car/scripts/run_car.sh build` on the car afterwards.
# ---------------------------------------------------------------------------
set -euo pipefail

CAR="${1:-192.168.55.1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

find "$ROOT" -name __pycache__ -prune -exec rm -rf {} +
rsync -a --delete --exclude __pycache__ --exclude '*.docx' --exclude '*.pdf' \
  --chmod=F644,Fu+x,D755 -e "${SSH[*]}" "$ROOT/" "jetson@$CAR:crc_car/"

"${SSH[@]}" "jetson@$CAR" 'bash -s' <<'REMOTE'
set -e
chmod +x ~/crc_car/scripts/*.sh ~/crc_car/scripts/*.py
grep -q "alias car=" ~/.bashrc || cat >> ~/.bashrc <<'ALIASES'

# UEH CRC 2026 team shortcuts
alias car='bash ~/crc_car/scripts/run_car.sh'
alias car-check='bash ~/crc_car/scripts/car_check.sh'
ALIASES
echo "$(hostname): ~/crc_car updated, shortcuts: $(grep -c 'alias car' ~/.bashrc) lines"
REMOTE
