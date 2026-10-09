# 🚗 Video Vision Lab (UEH Team 7 - CRC 2026)

Thư mục độc lập chuyên biệt dành riêng cho việc nghiên cứu, tối ưu và kiểm thử các thuật toán thị giác máy tính trên video sa bàn thi đấu:
`images/Screen Recording 2026-10-09 at 20.20.59.mov`

> ⚠️ **Tính độc lập 100%:** Toàn bộ mã nguồn, cấu hình và công cụ kiểm thử được đóng gói hoàn toàn trong thư mục `video_vision_lab/`, không làm ảnh hưởng hay thay đổi bất kỳ file nào khác trong dự án chính.

---

## 🎬 Phân Tích Video Sa Bàn Thực Tế

- **Định dạng & Thông số:** Video độ phân giải gốc **2268 x 1702** (tỉ lệ 4:3), tốc độ **15.86 FPS**, tổng cộng **2042 frames** (thời lượng **128.7 giây**).
- **Góc nhìn camera:** Camera góc rộng gắn trước mũi xe robot tự hành quay toàn cảnh sa bàn thi đấu.
- **Thách thức thị giác chính:**
  1. **Ánh đèn trần LED chói lóa (Glare):** Mặt sàn bạt đen bóng tạo các vệt phản xạ ánh sáng trắng rất mạnh dọc thân đường. Nếu chỉ dùng nhị phân hóa hoặc HSV thông thường sẽ bị nhầm lẫn vệt chói với vạch sơn trắng.
  2. **Đèn tín hiệu giao thông (Traffic Light):** Cột đèn 3 bóng dạng hộp đứng: bóng **Xanh ở trên cùng**, bóng **Vàng ở giữa**, bóng **Đỏ ở dưới cùng**. Khi đèn bật, ánh sáng LED vừa gây hiện tượng bão hòa màu (blooming), vừa tạo bóng phản chiếu cùng màu trên nền sàn gạch/bạt.
  3. **Biển báo giao thông ven đường:** Các biển báo cắm trên cọc ven đường (`hw_entry`, `turn_right`, `forward`, `stop`,...). Cần phân biệt rõ biển báo với trang phục của người tham gia di chuyển ở hậu cảnh.
  4. **Vạch dừng ngã tư & Vạch đi bộ (Zebra Crossing):** Các khối vạch trắng ngang nằm chắn ngang làn xe trước đèn tín hiệu.
  5. **Đoạn cua gắt (Sharp Curve):** Bán kính cong lớn đòi hỏi thuật toán khớp đa thức bậc 2 ($x = ay^2 + by + c$) mượt mà, không bị gãy góc.

---

## 🏛️ Cấu Trúc Thư Mục

```text
video_vision_lab/
├── __init__.py                 # Khởi tạo package
├── config.py                   # Tham số cấu hình tập trung (HSV, Top-Hat, ROI, Key Scenes)
├── lane_detector.py            # Thuật toán chống chói Top-Hat & bám đa thức vạch phải
├── traffic_light_detector.py   # Nhận diện cột đèn giao thông & phân loại Xanh/Vàng/Đỏ
├── sign_detector.py            # Nhận diện biển báo giao thông bằng Template Matching
├── visualizer.py               # Bộ dựng Dashboard AR Telemetry cao cấp (Tesla/Comma.ai style)
├── processor.py                # Đường ống thị giác tích hợp (VisionPipeline)
├── run_video.py                # Script chạy chính (Hỗ trợ GUI tương tác & Xuất video MP4)
├── test_samples.py             # Bộ kiểm thử tự động trên 6 kịch bản trọng yếu
├── snapshots/                  # Ảnh chụp màn hình kết quả kiểm thử các kịch bản
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chạy tương tác với giao diện GUI (Khuyên dùng)
```bash
python3 video_vision_lab/run_video.py
```

### 2. Các phím điều khiển khi đang mở cửa sổ GUI
| Phím | Chức năng |
| :--- | :--- |
| **`SPACE`** | **Tạm dừng / Tiếp tục chạy** (Pause / Play) |
| **`D`** hoặc **Mũi tên phải `→`** | Tiến 1 frame (khi tạm dừng) hoặc nhảy nhanh +40 frames |
| **`A`** hoặc **Mũi tên trái `←`** | Lùi 1 frame (khi tạm dừng) hoặc nhảy lùi -40 frames |
| **`S`** | **Chụp và lưu ảnh Dashboard hiện tại** vào thư mục `snapshots/` |
| **`R`** | Quay về đầu video (frame 0) |
| **`1`** -> **`6`** | **Nhảy ngay lập tức đến 6 kịch bản mốc:**<br>• `1`: Vạch xuất phát & Đèn xanh<br>• `2`: Đoạn thẳng chói đèn trần<br>• `3`: Khúc cua gắt sang phải<br>• `4`: Chân dốc cầu & vòng xuyến<br>• `5`: Đèn đỏ ngã tư & vạch dừng<br>• `6`: Cận cảnh đèn đỏ |
| **`Q`** hoặc **`ESC`** | Thoát chương trình |

---

### 3. Nhảy trực tiếp đến một cảnh cụ thể từ dòng lệnh
```bash
# Nhảy tới cảnh cua gắt:
python3 video_vision_lab/run_video.py --scene 3

# Nhảy tới cảnh đèn đỏ ngã tư:
python3 video_vision_lab/run_video.py --scene 5

# Bắt đầu chạy từ frame 1540:
python3 video_vision_lab/run_video.py --start-frame 1540
```

---

### 4. Xuất video đã xử lý ra file MP4
Bạn có thể render toàn bộ Dashboard kèm HUD và lưu thành file video MP4 với tốc độ xử lý siêu nhanh (~75 FPS):
```bash
# Xuất 300 frame đầu tiên ra video:
python3 video_vision_lab/run_video.py --save-video demo_run.mp4 --headless --max-frames 300

# Xuất toàn bộ video:
python3 video_vision_lab/run_video.py --save-video full_run.mp4 --headless
```

---

### 5. Chạy bộ kiểm thử tự động 6 kịch bản (Test Suite)
Chạy script kiểm thử để đánh giá độ chính xác và xuất toàn bộ ảnh phân tích vào `snapshots/`:
```bash
python3 video_vision_lab/test_samples.py
```

Kết quả mẫu:
```text
======================================================================
📊 BẢNG TỔNG HỢP KẾT QUẢ KIỂM THỬ:
Kịch bản                 | Frame  | Bám làn      | Đèn      | Vạch dừng  | Độ trễ  
----------------------------------------------------------------------
1_START_GREEN_LIGHT      | 0      | TRACKING     | GREEN    | CÓ         | 42.3ms  
2_GLARE_STRAIGHT         | 360    | TRACKING     | NONE     | CÓ         | 9.6ms   
3_SHARP_RIGHT_CURVE      | 840    | TRACKING     | NONE     | CÓ         | 10.9ms  
4_BRIDGE_RAMP            | 1080   | HOLD_PREV    | RED      | CÓ         | 9.6ms   
5_RED_LIGHT_STOP         | 1540   | TRACKING     | RED      | CÓ         | 14.7ms  
6_RED_LIGHT_CLOSE        | 1600   | TRACKING     | RED      | CÓ         | 12.1ms  
======================================================================
```

---

## 🧠 Giải Thích Nguyên Lý Kỹ Thuật

### 1. Bộ Lọc Quang Học Chống Chói (Anti-Glare Top-Hat Filter)
- Bóng đèn trần LED rọi xuống mặt bạt đen tạo ra các vệt chói loang rộng ($> 50\text{px}$). Trong khi đó, vạch băng keo trắng thi đấu có độ dày hẹp đồng nhất ($15 - 30\text{px}$).
- Áp dụng phép biến đổi hình thái học **Top-Hat** với kernel chữ nhật ngang $25 \times 5$:
  $$\text{TopHat}(I) = I - (I \circ K)$$
  Triệt tiêu hoàn toàn các vùng sáng loang rộng, chỉ giữ lại các vệt sáng có biên độ thay đổi cục bộ hẹp đúng chuẩn kích thước vạch kẻ.
- Kết hợp lọc điều kiện giá trị màu HSV ($V \ge 125, S \le 90$) và hàm `minAreaRect` để loại bỏ mép sàn bạt mỏng, thu được mask vạch trắng hoàn hảo.

### 2. Thuật Toán Bám Vạch Đa Thức Bậc 2 (2nd-Order Polynomial Lane Fitting)
- Dò quét từ chân xe lên đường chân trời qua 10 cửa sổ trượt (Sliding Windows), ưu tiên đỉnh xung có bề ngang dày nhất (Peak Width).
- Khớp phương trình đa thức bậc 2:
  $$x(y) = a \cdot y^2 + b \cdot y + c$$
- Đo đạc chính xác:
  - **Độ lệch tâm (Steering Error):** Khoảng cách pixel giữa điểm nhìn xa $y_{\text{lookahead}}$ và mục tiêu chuẩn $x = 510\text{px}$.
  - **Góc hướng (Heading Angle):** Đạo hàm tiếp tuyến $\frac{dx}{dy} = 2ay + b$.
  - **Độ cong mặt đường (Signed Curvature):** $k = \frac{x''}{(1 + (x')^2)^{1.5}}$.

### 3. Bộ Nhận Diện Đèn Tín Hiệu Giao Thông Chống Phản Chiếu
- Phát hiện ánh sáng LED cực đại (Prominence):
  - Đèn Xanh: $G > 135$ và $G > \frac{R + B}{2} + 14$
  - Đèn Đỏ: $R > 135$ và $R > \max(G, B) + 24$
- **Khử phản chiếu sàn gạch/bạt:** Khi đèn sáng, bóng phản chiếu luôn nằm ở phía dưới mặt sàn. Thuật toán sắp xếp các ứng viên theo trục $y$ tăng dần để khóa chặt vị trí bóng đèn thực trên không, đồng thời kiểm tra thân vỏ hộp đèn màu đen bao quanh.
- Ước lượng khoảng cách tới đèn dựa trên kích thước thấu kính (Distance Estimation).

### 4. Giao Diện Trực Quan Hóa (Telemetry HUD)
- Khung hình Camera chính được tăng cường AR: vẽ hành lang di chuyển an toàn dạng bán trong suốt (Safe Corridor), đường cong neon green bám vạch, vector độ lệch lái, hộp bounding box đèn và biển báo phát sáng theo trạng thái.
- Cột thông số bên phải hiển thị Mask quang học trực tiếp, đèn tín hiệu LED, đồng hồ đo độ cong, góc lái và quyết định tự hành thời gian thực: `CRUISE`, `BRAKE - STOP LINE`, `STOP - RED LIGHT`, `CORNER - SHARP RIGHT`.
