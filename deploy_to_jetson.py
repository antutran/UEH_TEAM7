#!/usr/bin/env python3
"""
Script tự động đẩy toàn bộ code từ máy tính lên Jetson Orin (172.20.10.2)
Tự động nhập mật khẩu, đồng bộ tất cả workspace trên Jetson và chạy car compile.
"""

import os
import pty
import select
import sys
import time

import os
import pty
import select
import subprocess
import sys
import time

JETSON_IPS = ["172.20.10.2", "192.168.55.1"]
JETSON_USER = os.environ.get("JETSON_USER", "jetson")
JETSON_PASS = os.environ.get("JETSON_PASS", "jetson")
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

def check_ip(ip):
    try:
        res = subprocess.run(["ping", "-c", "1", "-W", "1000", ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False

def get_reachable_ip():
    if "JETSON_IP" in os.environ:
        return os.environ["JETSON_IP"]
    for ip in JETSON_IPS:
        if check_ip(ip):
            return ip
    return JETSON_IPS[0]

def run_cmd_pty(args, password=JETSON_PASS):
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(args[0], args)
    else:
        output = b''
        while True:
            r, _, _ = select.select([fd], [], [], 20)
            if not r:
                break
            try:
                data = os.read(fd, 2048)
                if not data:
                    break
                output += data
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                if b'password:' in output.lower():
                    os.write(fd, (password + '\n').encode())
                    output = b''
            except OSError:
                break
        _, status = os.waitpid(pid, 0)
        return os.WEXITSTATUS(status) == 0

def main():
    target_ip = get_reachable_ip()
    print(f"🔍 Đang kiểm tra kết nối tới Jetson ({target_ip})...")
    if not check_ip(target_ip):
        print(f"❌ Không thể ping thấy Jetson tại {target_ip}!")
        print("💡 Hãy kiểm tra:")
        print("   1. Bạn đã kết nối vào đúng mạng Wi-Fi của xe (Hotspot 172.20.10.x) chưa?")
        print("   2. Hoặc cắm cáp Micro-USB nối từ máy tính vào Jetson (IP 192.168.55.1).")
        sys.exit(1)

    print(f"🚀 [1/3] Đóng gói mã nguồn src/ tại: {WORKSPACE_DIR}")
    tar_path = "/tmp/ueh_team7_src.tar.gz"
    res = os.system(f"cd {WORKSPACE_DIR} && COPYFILE_DISABLE=1 tar -czf {tar_path} src/")
    if res != 0:
        print("❌ Lỗi đóng gói thư mục src/")
        sys.exit(1)

    print(f"\n📡 [2/3] Đẩy tệp tin sang Jetson ({JETSON_USER}@{target_ip})...")
    ok = run_cmd_pty(['scp', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no', tar_path, f'{JETSON_USER}@{target_ip}:{tar_path}'])
    if not ok:
        print("\n❌ LỖI: Không thể gửi file sang Jetson! Vui lòng kiểm tra kết nối.")
        sys.exit(1)

    print(f"\n⚙️  [3/3] Giải nén vào /home/{JETSON_USER}/UEH_Team7/ và biên dịch 'car compile'...")
    remote_script = f"""
tar -xzf {tar_path} -C /home/{JETSON_USER}/UEH_Team7/
rm -f {tar_path}

echo "=== GẮN CONTAINER VÀO /home/{JETSON_USER}/UEH_Team7/src (TEAM=tuan) ==="
SRC=/home/{JETSON_USER}/UEH_Team7/src TEAM=tuan car up

echo "=== ĐANG BIÊN DỊCH WORKSPACE (car compile) ==="
car compile
"""
    ok = run_cmd_pty(['ssh', '-tt', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=no', f'{JETSON_USER}@{target_ip}', remote_script])
    if not ok:
        print("\n❌ LỖI: Biên dịch trên Jetson thất bại!")
        sys.exit(1)

    print(f"\n✅ HOÀN TẤT ĐỒNG BỘ LÊN JETSON ({target_ip})!")
    print("👉 Bây giờ trên terminal của Jetson bạn chỉ cần chạy:")
    print('   car start "PYTHONUNBUFFERED=1 ros2 launch crc_sim run_with_viewer.launch.py"')

if __name__ == '__main__':
    main()

