# UEH CRC 2026 - TÀI LIỆU VÀ MÃ NGUỒN CHÍNH (UEH TEAM 7)

Thư mục `main/` này chứa toàn bộ các file code cốt lõi đang trực tiếp điều khiển xe robot tự hành của **UEH Team 7** trên phần cứng **NVIDIA Jetson Orin Nano / Jetson Nano**.

---

## 📂 Danh mục các file trong thư mục

| Tên File | Ngôn ngữ | Vai trò chính |
|---|---|---|
| [`starter_node.py`](starter_node.py) | Python (ROS 2) | **Node não bộ điều khiển chính của xe**: Xử lý logic bám lane, điều khiển vận tốc thích ứng, ổn định góc lái bằng IMU, phanh an toàn LiDAR, nhận diện biển RAMP vượt dốc, vượt xe dừng cao tốc. |
| [`lane_detector.py`](lane_detector.py) | Python (OpenCV) | **Module thị giác máy tính nhận diện vạch đường**: Lọc Top-hat, HSV nhị phân, hành lang quét vạch phải (Corridor), khử nhiễu chói sáng bạt phản quang, thuật toán cửa sổ trượt (Sliding Windows), bộ nhớ đệm chống nhảy vạch. |
| [`web_viewer.py`](web_viewer.py) | Python (HTTP / MJPEG) | **Web Dashboard giám sát thời gian thực**: Phát stream video MJPEG độ trễ siêu thấp (Dual View: Camera thô + Phân tích Lane Debug), hiển thị Telemetry (Điện áp pin, Vận tốc $v, w$, FPS). |
| [`run_with_viewer.launch.py`](run_with_viewer.launch.py) | Python (ROS 2 Launch) | File khởi chạy đồng thời cả `starter_node` và `web_viewer` chỉ bằng 1 câu lệnh duy nhất. |
| [`crc_car.env`](crc_car.env) | Shell Config | File cấu hình phần cứng của xe: Thông số động cơ (PID, Feedforward PWM, đảo chiều motor/encoder), Camera CSI (lật hình, độ phân giải, HFOV). |
| [`images/`](images/) | PNG Images | Thư mục chứa các mẫu biển báo giao thông (RAMP dốc, STOP, TUNNEL hầm, Hướng rẽ,...) dùng cho so khớp mẫu template matching. |

---

## 🧠 Kiến trúc & Thuật toán điều khiển

### 1. Thuật toán bám duy nhất vạch phải (Pure Right-Lane Following)
* **Ý tưởng cốt lõi:** Thay vì tìm cả 2 vạch và tính tâm đường (dễ bị nhiễu bởi vạch đứt, bóng xe, ngã ba, đường hắt sáng), xe **chỉ bám duy nhất vào mép vạch trắng bên phải**.
* **Hành lang quét (Right Corridor):** Chỉ tìm kiếm vạch trong khoảng $x \in [380, 638]$ px. Bỏ qua hoàn toàn nửa bên trái khung hình để triệt tiêu việc nhận nhầm vạch tim đường hay ánh sáng chói giữa làn.
* **Căn vị trí xe:** Giữ vạch phải nằm ở tọa độ chuẩn `target_right_x = 460.0 px` (trên ảnh 640x480).
  * Nếu vạch lệch $> 460\text{px}$: Xe đang lệch sang trái $\rightarrow$ Đánh lái sang phải để ôm lại vạch.
  * Nếu vạch lệch $< 460\text{px}$: Xe đang quá sát mép phải $\rightarrow$ Đánh lái sang trái để giữ khoảng cách an toàn.

### 2. Bộ lọc chống chói sáng (Anti-Glare) & Chống nhảy vạch (Spike Rejection)
* **Top-hat Morphological Filter:** Khử ánh sáng chói loang lổ của đèn sân bạt.
* **Temporal Memory Window:** Khi đã khóa được vạch phải, thuật toán tạo một cửa sổ $\pm 55\text{px}$ quanh vị trí frame trước. Mọi vệt sáng xuất hiện bất ngờ ở giữa làn sẽ bị loại bỏ 100%.
* **Spike Rejection kết hợp IMU:** Nếu camera nhảy vọt $> 40\text{px}$ trong 1 frame nhưng con quay hồi chuyển IMU Gyro báo xe đang thẳng ($|\text{gyro\_z}| < 0.08\text{ rad/s}$), giá trị nhảy vọt sẽ bị từ chối như một lỗi nhiễu camera.

### 3. Bộ điều khiển bẻ lái chống lắc lư (Anti-Wobble Steering)
* **Vùng chết Deadzone (`deadzone = 12.0px`):** Khi sai số nằm trong $\pm 12\text{px}$, xe giữ thẳng bánh lái tuyệt đối ($w = 0$), giúp 2 bánh đồng tốc và đi thẳng mượt, triệt tiêu hiện tượng "rắn bò".
* **Bộ điều khiển PD + Lọc vi phân thông thấp:** Bù góc lái êm ái, kèm bộ giới hạn gia tốc lái (`max_steer_step = 0.07 rad/s`).
* **Con quay hồi chuyển IMU Gyro Heading-Hold:** Khi xe chạy thẳng, vận tốc góc quay $\text{gyro\_z}$ từ IMU được dùng để tự động bù góc lái giữ cho xe luôn hướng thẳng trục đường.

### 4. Điều khiển vận tốc thích ứng & Chống khựng hộp số (Stiction Prevention)
* **Chạy đường thẳng:** Tự động tăng tốc nhanh dần lên `straight_speed = 0.35 m/s`.
* **Vào cua:** Hãm tốc độ về `corner_speed = 0.18 m/s` để ôm cua chắc chắn.
* **Khắc phục hiện tượng giật dừng trên Jetson Orin:**
  * Động cơ JGB37-520 có tỉ số truyền 1:30 với tải trọng xe nặng đòi hỏi tối thiểu **18-20% PWM** để thắng ma sát tĩnh.
  * Tốc độ sàn được khống chế $\ge 0.18\text{ m/s}$ và `min_inner_forward = 0.045 m/s` trong hàm `drive()`, đảm bảo bánh phía trong không bao giờ bị đứng bánh.
  * Trong `crc_car.env`: Đặt `CRC_KI = 0.0` và `CRC_FF_OFFSET = 16.5%` để loại bỏ hoàn toàn dao động khâu tích phân giật nhịp.

---

## 🛠️ Hướng dẫn tinh chỉnh các thông số quan trọng (Tuning Guide)

Tất cả các tham số điều khiển đều nằm ở phần đầu hàm `__init__` của file [`starter_node.py`](starter_node.py):

| Tham số | Giá trị hiện tại | Ý nghĩa & Hướng dẫn điều chỉnh |
|---|---|---|
| `target_right_x` | `460.0` | **Vị trí xe trong làn.** Muốn xe đi sát sang bên PHẢI hơn $\rightarrow$ GIẢM giá trị này (vd: 450.0). Muốn xe dịch sang TRÁI hơn $\rightarrow$ TĂNG giá trị này (vd: 480.0 - 500.0). |
| `straight_speed` | `0.35` | Tốc độ tối đa trên đoạn đường thẳng (m/s). |
| `corner_speed` | `0.18` | Tốc độ khi ôm cua gắt (m/s). Khuyến nghị không để $< 0.16$ để tránh đứng bánh do ma sát hộp số. |
| `lost_line_speed` | `0.18` | Tốc độ chạy thẳng khi tạm thời mất vạch (m/s). |
| `kp` | `0.0030` | Hệ số P bám vạch. Tăng nếu xe phản ứng lái chậm; giảm nếu xe bị đảo lái lắc lư. |
| `kd` | `0.0008` | Hệ số D dập tắt dao động lắc lái. |
| `deadzone` | `12.0` | Khoảng chết pixel (|error| <= 12px thì đi thẳng). Giúp xe chạy thẳng tắp. |
| `stop_distance` | `0.28` | Khoảng cách phanh an toàn trước vật cản đo bởi LiDAR (m). |

---

## 🚀 Cách chạy trên xe Jetson

### 1. Khởi động chương trình chính + Web Dashboard
Trên terminal SSH của Jetson:
```bash
car start "PYTHONUNBUFFERED=1 ros2 launch crc_sim run_with_viewer.launch.py"
```

### 2. Mở Web Dashboard quan sát trực tiếp
Trên máy tính hoặc điện thoại kết nối cùng mạng Wi-Fi:
👉 **`http://172.20.10.2:8080/`** *(hoặc `http://192.168.55.1:8080/` nếu cắm dây cáp USB)*

### 3. Theo dõi log và Dừng xe
* Xem log thời gian thực:
  ```bash
  car logs
  ```
* Dừng xe:
  ```bash
  car stop
  ```
