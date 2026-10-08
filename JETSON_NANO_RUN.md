# UEH CRC 2026 - Hướng dẫn chạy trên NVIDIA Jetson Nano

Tài liệu này ghi lại toàn bộ quy trình cấu hình, build và vận hành code chính robot UEH Team 7 trên xe thực tế (NVIDIA Jetson Nano).

---

## 1. Thông tin phần cứng & môi trường hệ thống thực tế

- **Phần cứng:** NVIDIA Jetson Nano Developer Kit (4GB RAM, ARM Cortex-A57 aarch64, 128-core Maxwell GPU).
- **Hệ điều hành Host:** Ubuntu 18.04.6 LTS (Bionic Beaver), Linux Kernel 4.9.337-tegra.
- **NVIDIA L4T / JetPack:** L4T R32.7.5 (JetPack 4.6.5), CUDA 10.2, TensorRT 8.2.1.
- **Môi trường ROS:** ROS 2 Humble Hawksbill (chạy trong Docker container chuyên dụng `crc_car:humble` với NVIDIA Container Runtime, chia sẻ mạng Host và IPC).
- **Python:** Python 3.10.12 (bên trong Docker ROS 2 Humble).
- **Thư mục project trên Jetson:** `/home/jetson/UEH_Team7`.

### Thiết bị phần cứng đã kết nối và cấu hình:
- **CSI Camera (IMX219):** `/dev/video0` (CSI port CAM0), quản lý bởi service `crc-camera`, node ROS `/camera_node` publish topic `/camera/image_raw` (640x480 rgb8 @ 30 FPS).
- **Board điều khiển động cơ:** Yahboom ROS Board V3.0 tại `/dev/myserial` -> `/dev/ttyUSB0` (Chip CH340), node `/base_driver` điều khiển động cơ JGB37-520 qua PID encoder và nhận lệnh `/cmd_vel`.
- **LiDAR:** RPLIDAR C1 tại `/dev/rplidar` -> `/dev/ttyUSB1` (Chip CP210x), node `/sllidar_node` + `/scan_adapter` publish topic `/scan` (360 tia @ 10 Hz).

---

## 2. Các topic ROS 2 quan trọng

| Topic | Type | Chiều | Chức năng |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Sub | Ảnh thô từ camera robot (640x480 @ 30fps) |
| `/scan` | `sensor_msgs/msg/LaserScan` | Sub | Dữ liệu quét 360 độ từ LiDAR |
| `/odom` | `nav_msgs/msg/Odometry` | Sub | Vị trí và vận tốc tích hợp từ encoder bánh xe |
| `/imu` | `sensor_msgs/msg/Imu` | Sub | Góc nghiêng và gia tốc từ cảm biến IMU |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Pub | Lệnh vận tốc điều khiển robot (v, w) |
| `/lane_debug/image` | `sensor_msgs/msg/Image` | Pub | Ảnh trực quan hóa thuật toán nhận diện làn đường |
| `/detection_debug/image` | `sensor_msgs/msg/Image` | Pub | Ảnh trực quan hóa nhận diện biển báo & đèn giao thông |
| `/voltage` | `std_msgs/msg/Float32` | Monitor | Điện áp pin (bình thường 11.0V - 12.6V) |

---

## 3. Lệnh vận hành chính xác

Nhờ các alias và biến môi trường đã được cấu hình trong `~/.bashrc`, bạn có thể thực hiện mọi thao tác nhanh bằng lệnh `car`.

### Bước 1: SSH vào Jetson Nano
```bash
ssh jetson@172.20.10.9
# Hoặc nếu cắm cáp Micro-USB: ssh jetson@192.168.55.1
```

### Bước 2: Kiểm tra trạng thái phần cứng và cảm biến
```bash
car-check
```
*(Lệnh này tự động kiểm tra L4T, nhiệt độ, camera, LiDAR, Yahboom board, pin và các topic sensor. Cần đạt 26 PASS).*

### Bước 3: Khởi động container môi trường của Team 7
```bash
car up
```
*(Nếu cần chỉ định rõ: `TEAM=team7 SRC=/home/jetson/UEH_Team7/src bash ~/crc_car/scripts/run_car.sh up`)*

### Bước 4: Build project
```bash
car compile
```
*(Lệnh này chạy `colcon build` cho package `crc_sim` trong container)*

### Bước 5: Chạy chương trình chính của Robot & Mở quan sát Camera trực tiếp

**Cách 1 (Khuyên dùng - Chạy xe + Bật Web Dashboard quan sát Camera & Pin cùng lúc):**
```bash
car start "PYTHONUNBUFFERED=1 ros2 launch crc_sim run_with_viewer.launch.py"
```
*(Chương trình tự động chạy ngầm, xe hoạt động và đồng thời phát stream video Web Dashboard).*

**Cách 2 (Chỉ chạy node điều khiển xe, không bật Web Viewer để tiết kiệm tài nguyên tối đa):**
```bash
car start "PYTHONUNBUFFERED=1 ros2 run crc_sim starter"
```

### Bước 6: Quan sát trực tiếp trên màn hình máy tính / điện thoại
Mở trình duyệt web (Chrome / Safari) trên máy tính hoặc điện thoại cùng mạng:
👉 **`http://172.20.10.9:8080/`**  *(hoặc `http://192.168.55.1:8080/` nếu cắm cáp Micro-USB)*

Trên giao diện Web bạn có thể:
- 📺 **Song song (Dual View):** Xem cả 2 màn hình cùng lúc (Mắt camera xe thật + Phân tích vạch làn xe).
- 🛣️ **Lane Debug:** Phóng to hình ảnh thuật toán bám làn (Target point, Steering angle, Polyline).
- 📷 **Mắt Xe (Raw Camera):** Phóng to góc nhìn thực tế 640x480 từ camera IMX219 trên xe.
- ⚡ **Telemetry thời gian thực:** Hiển thị liên tục phần trăm Pin, điện áp (V), và vận tốc xe ($v, w$).

### Bước 7: Theo dõi log thực thi trên terminal
```bash
car logs
```
*(Nhấn Ctrl+C để thoát xem log, chương trình chính vẫn tiếp tục chạy trên xe).*

### Bước 8: Kiểm tra trạng thái container và topic
```bash
car status
```

### Bước 9: Dừng chương trình an toàn
```bash
car stop
```
*(Lệnh này sẽ gửi tín hiệu SIGINT -> SIGTERM tới nhóm tiến trình, dừng node và motor an toàn).*

---

## 4. Chạy trực tiếp qua Interactive Terminal (Nếu muốn quan sát stdout trực tiếp)

Nếu muốn chạy trực tiếp gắn với terminal hiện tại thay vì chạy background:
```bash
docker exec -it crc_car bash -c "source /opt/ros/humble/setup.bash && source /opt/crc_ws/install/setup.bash && source /ws_build/install/setup.bash && export ROS_DOMAIN_ID=5 && ros2 run crc_sim starter"
```
Nhấn `Ctrl + C` để dừng.

---

## 5. Lưu ý an toàn & Xử lý sự cố (Troubleshooting)

1. **Tại sao khi đặt robot trên bàn, bánh xe chưa quay?**
   - **Cơ chế An toàn 1 - Initial Acquire:** Code chính (`starter_node.py`) áp dụng cơ chế khóa làn ban đầu. Khi xe ở trên bàn (không có vạch kẻ làn đường màu trắng), node ở trạng thái `ACQUIRE 0/3` và chủ động giữ `v=0, w=0` để đảm bảo an toàn. Khi đặt vào sa bàn có vạch line trắng, node sẽ chuyển sang `RIGHT POLYLINE LOCKED` và bắt đầu điều khiển xe chạy.
   - **Cơ chế An toàn 2 - Obstacle Safety:** Nếu LiDAR phát hiện vật thể phía trước trong khoảng cách < 0.30 m (`stop_distance`), node sẽ kích hoạt `OBSTACLE STOP`.
2. **Điện áp pin:**
   - Kiểm tra điện áp pin qua topic `/voltage`: `docker exec crc_car bash -c "source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=5 && ros2 topic echo /voltage --once"`
   - Nếu điện áp xuống dưới 11.0V, cần sạc pin để tránh sụt áp gây reset Jetson Nano khi motor khởi động.
3. **Khởi động lại driver xe khi cần thiết:**
   - Nếu rút cắm lại camera hoặc USB LiDAR/motor:
   ```bash
   car bringup
   ```
