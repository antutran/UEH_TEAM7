#!/usr/bin/env bash
# Script tự động đẩy code lên Jetson
set -e

JETSON_IP="${JETSON_IP:-172.20.10.2}"
JETSON_USER="${JETSON_USER:-jetson}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAR_FILE="/tmp/ueh_team7_src.tar.gz"

echo "🚀 [1/3] Đóng gói mã nguồn src/..."
COPYFILE_DISABLE=1 tar -czf "$TAR_FILE" -C "$DIR" src

echo "📡 [2/3] Chép file sang Jetson ($JETSON_USER@$JETSON_IP)..."
scp -o StrictHostKeyChecking=no "$TAR_FILE" "$JETSON_USER@$JETSON_IP:$TAR_FILE"

echo "⚙️  [3/3] Giải nén và biên dịch trên Jetson..."
ssh -tt -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" << EOF
tar -xzf $TAR_FILE -C /home/$JETSON_USER/UEH_Team7/
rm -f $TAR_FILE
echo "=== GẮN CONTAINER VÀO UEH_Team7 (TEAM=tuan) ==="
SRC=/home/$JETSON_USER/UEH_Team7/src TEAM=tuan car up
echo "=== ĐANG BIÊN DỊCH (car compile) ==="
car compile
exit
EOF

echo ""
echo "✅ HOÀN TẤT ĐỒNG BỘ LÊN JETSON!"
echo "👉 Trên Jetson, bạn chỉ cần gõ:"
echo "   car start \"PYTHONUNBUFFERED=1 ros2 launch crc_sim run_with_viewer.launch.py\""
