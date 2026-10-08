#!/usr/bin/env python3
"""
Ultra Low-Latency Web Viewer & Telemetry Dashboard for UEH Team 7 Robot
Optimized for NVIDIA Jetson Nano:
- Direct pass-through of hardware-compressed JPEG (/camera/image_raw/compressed)
- Zero-latency Condition notification for MJPEG delivery
- High frame rate (up to 20-25 FPS) with adaptive JPEG encoding
- Unified Single-Stream Dual View (/stream/dual) preventing multi-socket Wi-Fi congestion
- Real-time telemetry: Battery (V, %), Speed (v, w), Stream FPS
"""

import io
import json
import os
import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

PORT = int(os.environ.get("VIEWER_PORT", 8080))
TARGET_FPS = float(os.environ.get("VIEWER_FPS", 20.0))
MIN_INTERVAL = 1.0 / max(1.0, TARGET_FPS)

def make_placeholder_frame(title="Dang ket noi...", subtitle="Tin hieu se xuat hien tai day"):
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = (20, 24, 33)
    cv2.putText(img, "UEH Team 7", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2, cv2.LINE_AA)
    cv2.putText(img, title, (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    return buf.tobytes() if ret else None


# Shared State & Condition Variables for zero-latency notification
state_lock = threading.Lock()
frame_conditions = {
    'lane': threading.Condition(state_lock),
    'camera': threading.Condition(state_lock),
    'dual': threading.Condition(state_lock),
}

stream_data = {
    'lane': {
        'jpeg': make_placeholder_frame("Cho tin hieu /lane_debug/image", "Node starter dang phan tich"),
        'raw_bgr': None,
        'seq': 0,
        'ts': time.monotonic(),
        'fps': 0.0,
        'frame_count': 0,
        'fps_timer': time.monotonic(),
        'clients': 0
    },
    'camera': {
        'jpeg': make_placeholder_frame("Cho tin hieu camera...", "Kiem tra IMX219 CSI Driver"),
        'raw_bgr': None,
        'seq': 0,
        'ts': time.monotonic(),
        'fps': 0.0,
        'frame_count': 0,
        'fps_timer': time.monotonic(),
        'clients': 0
    },
    'dual': {
        'jpeg': None,
        'seq': 0,
        'ts': time.monotonic(),
        'clients': 0
    },
    'telemetry': {
        'voltage': 0.0,
        'battery_pct': 0.0,
        'linear_vel': 0.0,
        'angular_vel': 0.0,
        'lane_fps': 0.0,
        'cam_fps': 0.0,
        'target_fps': TARGET_FPS,
        'last_voltage_update': 0.0,
        'last_vel_update': 0.0,
    }
}


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Suppress request spam

    def do_GET(self):
        # 1. Main Dashboard HTML
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UEH Team 7 - Telemetry & Camera sieu muot</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #090d16;
      color: #f1f5f9;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 16px;
    }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      max-width: 1140px;
      margin-bottom: 14px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .title-group {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .logo-badge {{
      background: linear-gradient(135deg, #2563eb, #06b6d4);
      color: white;
      font-weight: 800;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 15px;
      letter-spacing: 0.5px;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }}
    h1 {{
      font-size: 20px;
      font-weight: 700;
      color: #f8fafc;
    }}
    .status-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      background: #141b2d;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .pill-battery {{
      background: rgba(16, 185, 129, 0.15);
      border-color: rgba(16, 185, 129, 0.35);
      color: #34d399;
    }}
    .pill-speed {{
      background: rgba(59, 130, 246, 0.15);
      border-color: rgba(59, 130, 246, 0.35);
      color: #60a5fa;
    }}
    .pill-fps {{
      background: rgba(245, 158, 11, 0.15);
      border-color: rgba(245, 158, 11, 0.35);
      color: #fbbf24;
    }}
    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 1.8s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.4; transform: scale(0.9); }}
    }}

    /* View Mode Tabs */
    .view-controls {{
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
      width: 100%;
      max-width: 1140px;
      flex-wrap: wrap;
    }}
    .btn {{
      padding: 9px 16px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: #141b2d;
      color: #94a3b8;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease-in-out;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .btn:hover {{
      background: #1e293b;
      color: #f1f5f9;
      border-color: rgba(255, 255, 255, 0.25);
    }}
    .btn.active {{
      background: #2563eb;
      color: white;
      border-color: #3b82f6;
      box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4);
    }}

    /* Stream Containers */
    .stream-container {{
      width: 100%;
      max-width: 1140px;
    }}
    .video-card {{
      background: #111827;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.7);
      display: flex;
      flex-direction: column;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.02);
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }}
    .card-title {{
      font-size: 13px;
      font-weight: 700;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .tag {{
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.08);
      color: #94a3b8;
    }}
    .tag-blue {{
      background: rgba(37, 99, 235, 0.2);
      color: #60a5fa;
    }}
    .tag-emerald {{
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
    }}
    .tag-purple {{
      background: rgba(147, 51, 234, 0.2);
      color: #c084fc;
    }}
    .video-wrapper {{
      width: 100%;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }}
    .video-wrapper img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .card-footer {{
      padding: 8px 14px;
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: #64748b;
      background: rgba(0, 0, 0, 0.25);
    }}
    .card-footer code {{
      color: #94a3b8;
      font-family: monospace;
    }}

    /* Grid for Separate Split */
    .grid-split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}

    @media (max-width: 768px) {{
      .grid-split {{
        grid-template-columns: 1fr;
      }}
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
    }}
  </style>
</head>
<body>

  <div class="header">
    <div class="title-group">
      <div class="logo-badge">TEAM 7</div>
      <h1>Ultra-Smooth Robot Telemetry & Camera</h1>
    </div>
    <div class="status-group">
      <div class="pill pill-battery" id="battery-pill">
        <span>⚡ PIN: <strong id="val-battery">--%</strong> (<span id="val-volts">--V</span>)</span>
      </div>
      <div class="pill pill-speed" id="speed-pill">
        <span>🏎️ v: <strong id="val-v">0.00</strong> m/s | w: <strong id="val-w">0.00</strong></span>
      </div>
      <div class="pill pill-fps" id="fps-pill">
        <span>📡 FPS: <strong id="val-fps">--</strong></span>
      </div>
      <div class="pill">
        <div class="status-dot"></div>
        <span>ROS 2 DOMAIN 7</span>
      </div>
    </div>
  </div>

  <div class="view-controls">
    <button class="btn active" id="btn-dual" onclick="setView('dual')">🚀 Song Song Ghép 1 Stream (Khuyên Dùng)</button>
    <button class="btn" id="btn-lane" onclick="setView('lane')">🛣️ Chỉ Xem Làn (Lane Debug)</button>
    <button class="btn" id="btn-cam" onclick="setView('cam')">📷 Chỉ Xem Mắt Xe (Hardware JPEG)</button>
    <button class="btn" id="btn-split" onclick="setView('split')">👥 2 Khung Riêng Biệt (Split Cards)</button>
  </div>

  <div class="stream-container">
    <!-- View 1: Unified Dual Single Stream (Most efficient, zero lag) -->
    <div class="video-card" id="view-dual" style="display: flex;">
      <div class="card-header">
        <div class="card-title">
          <span>📺 Kênh Ghép Đồng Bộ: Trái (Phân Tích Làn) | Phải (Camera Thực)</span>
        </div>
        <span class="tag tag-purple">Single HTTP Stream ~20 FPS</span>
      </div>
      <div class="video-wrapper" style="aspect-ratio: 8 / 3;">
        <img id="img-dual" src="/stream/dual" alt="Dual Stream" />
      </div>
      <div class="card-footer">
        <div>Truyền dẫn: <code>/stream/dual (1 kết nối socket duy nhất - triệt tiêu lag Wi-Fi)</code></div>
        <div>Tốc độ đo: <code id="footer-dual-fps">~{TARGET_FPS:.0f} FPS</code></div>
      </div>
    </div>

    <!-- View 2: Single Lane -->
    <div class="video-card" id="view-lane" style="display: none;">
      <div class="card-header">
        <div class="card-title">
          <span>🛣️ Thuật toán bám làn (Lane Debug Overlay)</span>
        </div>
        <span class="tag tag-blue">Topic /lane_debug/image</span>
      </div>
      <div class="video-wrapper" style="aspect-ratio: 4 / 3;">
        <img id="img-lane" src="" alt="Lane Stream" />
      </div>
      <div class="card-footer">
        <div>Nguồn: <code>starter_node.py (ROI &amp; Lookahead Target)</code></div>
        <div>Độ phân giải: <code>480 &times; 360</code></div>
      </div>
    </div>

    <!-- View 3: Single Cam -->
    <div class="video-card" id="view-cam" style="display: none;">
      <div class="card-header">
        <div class="card-title">
          <span>📷 Mắt xe thật (Sony IMX219 CSI Camera)</span>
        </div>
        <span class="tag tag-emerald">Hardware Pre-Compressed JPEG</span>
      </div>
      <div class="video-wrapper" style="aspect-ratio: 4 / 3;">
        <img id="img-cam" src="" alt="Camera Stream" />
      </div>
      <div class="card-footer">
        <div>Nguồn: <code>/camera/image_raw/compressed (Không tốn CPU Jetson)</code></div>
        <div>Phần cứng: <code>IMX219 (CSI-2 CAM0)</code></div>
      </div>
    </div>

    <!-- View 4: Split Cards -->
    <div class="grid-split" id="view-split" style="display: none;">
      <div class="video-card">
        <div class="card-header">
          <div class="card-title">🛣️ Bám làn</div>
          <span class="tag tag-blue">Overlay</span>
        </div>
        <div class="video-wrapper" style="aspect-ratio: 4 / 3;">
          <img id="img-split-lane" src="" alt="Lane" />
        </div>
      </div>
      <div class="video-card">
        <div class="card-header">
          <div class="card-title">📷 Mắt xe</div>
          <span class="tag tag-emerald">Hardware JPEG</span>
        </div>
        <div class="video-wrapper" style="aspect-ratio: 4 / 3;">
          <img id="img-split-cam" src="" alt="Cam" />
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentMode = 'dual';

    function setView(mode) {{
      currentMode = mode;
      const vDual = document.getElementById('view-dual');
      const vLane = document.getElementById('view-lane');
      const vCam = document.getElementById('view-cam');
      const vSplit = document.getElementById('view-split');

      const imgDual = document.getElementById('img-dual');
      const imgLane = document.getElementById('img-lane');
      const imgCam = document.getElementById('img-cam');
      const imgSplitLane = document.getElementById('img-split-lane');
      const imgSplitCam = document.getElementById('img-split-cam');

      const bDual = document.getElementById('btn-dual');
      const bLane = document.getElementById('btn-lane');
      const bCam = document.getElementById('btn-cam');
      const bSplit = document.getElementById('btn-split');

      [bDual, bLane, bCam, bSplit].forEach(b => b.classList.remove('active'));

      // Tắt tất cả kết nối stream không dùng để giải phóng 100% băng thông
      imgDual.src = '';
      imgLane.src = '';
      imgCam.src = '';
      imgSplitLane.src = '';
      imgSplitCam.src = '';

      vDual.style.display = 'none';
      vLane.style.display = 'none';
      vCam.style.display = 'none';
      vSplit.style.display = 'none';

      if (mode === 'dual') {{
        vDual.style.display = 'flex';
        bDual.classList.add('active');
        imgDual.src = '/stream/dual?t=' + Date.now();
      }} else if (mode === 'lane') {{
        vLane.style.display = 'flex';
        bLane.classList.add('active');
        imgLane.src = '/stream/lane?t=' + Date.now();
      }} else if (mode === 'cam') {{
        vCam.style.display = 'flex';
        bCam.classList.add('active');
        imgCam.src = '/stream/camera?t=' + Date.now();
      }} else if (mode === 'split') {{
        vSplit.style.display = 'grid';
        bSplit.classList.add('active');
        imgSplitLane.src = '/stream/lane?t=' + Date.now();
        imgSplitCam.src = '/stream/camera?t=' + Date.now();
      }}
    }}

    async function updateTelemetry() {{
      try {{
        const res = await fetch('/api/status');
        if (res.ok) {{
          const data = await res.json();
          document.getElementById('val-battery').innerText = data.battery_pct.toFixed(1) + '%';
          document.getElementById('val-volts').innerText = data.voltage.toFixed(2) + 'V';
          document.getElementById('val-v').innerText = data.linear_vel.toFixed(2);
          document.getElementById('val-w').innerText = data.angular_vel.toFixed(2);
          
          const fps = Math.max(data.lane_fps, data.cam_fps);
          document.getElementById('val-fps').innerText = fps > 0 ? fps.toFixed(1) + ' FPS' : '--';

          const batPill = document.getElementById('battery-pill');
          if (data.voltage < 11.0 && data.voltage > 0) {{
            batPill.style.background = 'rgba(239, 68, 68, 0.2)';
            batPill.style.borderColor = 'rgba(239, 68, 68, 0.5)';
            batPill.style.color = '#f87171';
          }} else {{
            batPill.style.background = 'rgba(16, 185, 129, 0.15)';
            batPill.style.borderColor = 'rgba(16, 185, 129, 0.35)';
            batPill.style.color = '#34d399';
          }}
        }}
      }} catch (e) {{}}
    }}

    setInterval(updateTelemetry, 800);
    updateTelemetry();
  </script>
</body>
</html>
"""
            self.wfile.write(html.encode('utf-8'))

        # 2. Telemetry JSON API
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            with state_lock:
                tel = dict(stream_data['telemetry'])
            self.wfile.write(json.dumps(tel).encode('utf-8'))

        # 3. Stream MJPEG (/stream/lane, /stream/camera, /stream/dual)
        elif self.path.startswith('/stream'):
            clean_path = self.path.split('?')[0]
            if 'dual' in clean_path:
                channel = 'dual'
            elif 'camera' in clean_path:
                channel = 'camera'
            else:
                channel = 'lane'

            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            cond = frame_conditions[channel]
            with state_lock:
                stream_data[channel]['clients'] += 1

            last_seq = -1
            try:
                while True:
                    with cond:
                        # Zero-polling wait: wakes up instantly when notify_all() is called
                        start_wait = time.monotonic()
                        while stream_data[channel]['seq'] == last_seq:
                            cond.wait(timeout=0.15)
                            if time.monotonic() - start_wait >= 0.5:
                                break
                        jpeg_bytes = stream_data[channel]['jpeg']
                        last_seq = stream_data[channel]['seq']

                    if jpeg_bytes is not None:
                        header = (
                            b'--FRAME\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(jpeg_bytes)).encode('ascii') + b'\r\n\r\n'
                        )
                        self.wfile.write(header)
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError, Exception):
                pass
            finally:
                with state_lock:
                    stream_data[channel]['clients'] = max(0, stream_data[channel]['clients'] - 1)

        else:
            self.send_response(404)
            self.end_headers()


class DashboardSubscriber(Node):
    def __init__(self):
        super().__init__('robot_web_dashboard')
        self.bridge = CvBridge()
        self.last_lane_time = 0.0
        self.last_cam_time = 0.0

        # Subscriptions
        self.sub_lane = self.create_subscription(
            Image, '/lane_debug/image', self.on_lane_image, 2)
        
        # Subscribe direct to hardware-compressed image (zero CPU overhead)
        self.sub_cam_comp = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.on_cam_compressed, 2)
        
        # Fallback to raw image if compressed not present
        self.sub_cam_raw = self.create_subscription(
            Image, '/camera/image_raw', self.on_cam_raw, 2)

        self.sub_voltage = self.create_subscription(
            Float32, '/voltage', self.on_voltage, 10)
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.on_cmd_vel, 10)

        self.has_compressed_cam = False
        self.get_logger().info(f'Dashboard ultra-smooth streamer ready on port {PORT} at {TARGET_FPS} FPS')

    def _update_fps(self, channel):
        info = stream_data[channel]
        info['frame_count'] += 1
        now = time.monotonic()
        elapsed = now - info['fps_timer']
        if elapsed >= 1.0:
            info['fps'] = info['frame_count'] / elapsed
            info['frame_count'] = 0
            info['fps_timer'] = now
            if channel == 'lane':
                stream_data['telemetry']['lane_fps'] = info['fps']
            elif channel == 'camera':
                stream_data['telemetry']['cam_fps'] = info['fps']

    def _maybe_update_dual(self):
        """Tạo khung hình ghép Side-by-Side khi có client đang xem /stream/dual."""
        if stream_data['dual']['clients'] <= 0:
            return

        lane_bgr = stream_data['lane']['raw_bgr']
        cam_bgr = stream_data['camera']['raw_bgr']

        if lane_bgr is None and cam_bgr is None:
            return

        target_h, target_w = 300, 400
        
        if lane_bgr is not None:
            l_crop = cv2.resize(lane_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            l_crop = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(l_crop, "Cho Lane Debug...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if cam_bgr is not None:
            c_crop = cv2.resize(cam_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            c_crop = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(c_crop, "Cho Camera...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Ghep ngang 2 anh (Total: 800x300)
        dual_canvas = np.hstack([l_crop, c_crop])
        ret, buf = cv2.imencode('.jpg', dual_canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 48])
        if ret:
            with frame_conditions['dual']:
                stream_data['dual']['jpeg'] = buf.tobytes()
                stream_data['dual']['seq'] += 1
                stream_data['dual']['ts'] = time.monotonic()
                frame_conditions['dual'].notify_all()

    def on_lane_image(self, msg):
        now = time.monotonic()
        if now - self.last_lane_time < MIN_INTERVAL:
            return
        self.last_lane_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Lưu raw cho dual view
            stream_data['lane']['raw_bgr'] = cv_img

            # Chi encode rieng neu co client xem lane hoac split
            if stream_data['lane']['clients'] > 0:
                out_img = cv_img
                if cv_img.shape[1] > 480:
                    out_img = cv2.resize(cv_img, (480, 360), interpolation=cv2.INTER_LINEAR)
                ret, buf = cv2.imencode('.jpg', out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 48])
                if ret:
                    with frame_conditions['lane']:
                        stream_data['lane']['jpeg'] = buf.tobytes()
                        stream_data['lane']['seq'] += 1
                        stream_data['lane']['ts'] = now
                        self._update_fps('lane')
                        frame_conditions['lane'].notify_all()

            self._maybe_update_dual()
        except Exception:
            pass

    def on_cam_compressed(self, msg):
        """Pass-through truc tiep frame JPEG tu phan cung, khong ton CPU encode!"""
        self.has_compressed_cam = True
        now = time.monotonic()
        if now - self.last_cam_time < MIN_INTERVAL:
            return
        self.last_cam_time = now

        jpeg_data = bytes(msg.data)
        
        # Luu anh bgr de ghep dual chi khi can
        if stream_data['dual']['clients'] > 0:
            np_arr = np.frombuffer(jpeg_data, np.uint8)
            stream_data['camera']['raw_bgr'] = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        with frame_conditions['camera']:
            stream_data['camera']['jpeg'] = jpeg_data
            stream_data['camera']['seq'] += 1
            stream_data['camera']['ts'] = now
            self._update_fps('camera')
            frame_conditions['camera'].notify_all()

        self._maybe_update_dual()

    def on_cam_raw(self, msg):
        """Fallback neu topic compressed khong ton tai."""
        if self.has_compressed_cam:
            return  # Da dung hardware compressed, bo qua raw

        now = time.monotonic()
        if now - self.last_cam_time < MIN_INTERVAL:
            return
        self.last_cam_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stream_data['camera']['raw_bgr'] = cv_img

            if stream_data['camera']['clients'] > 0:
                out_img = cv_img
                if cv_img.shape[1] > 480:
                    out_img = cv2.resize(cv_img, (480, 360), interpolation=cv2.INTER_LINEAR)
                ret, buf = cv2.imencode('.jpg', out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 48])
                if ret:
                    with frame_conditions['camera']:
                        stream_data['camera']['jpeg'] = buf.tobytes()
                        stream_data['camera']['seq'] += 1
                        stream_data['camera']['ts'] = now
                        self._update_fps('camera')
                        frame_conditions['camera'].notify_all()

            self._maybe_update_dual()
        except Exception:
            pass

    def on_voltage(self, msg):
        now = time.monotonic()
        volts = float(msg.data)
        pct = max(0.0, min(100.0, (volts - 9.6) / (12.6 - 9.6) * 100.0))
        with state_lock:
            stream_data['telemetry']['voltage'] = volts
            stream_data['telemetry']['battery_pct'] = pct
            stream_data['telemetry']['last_voltage_update'] = now

    def on_cmd_vel(self, msg):
        now = time.monotonic()
        with state_lock:
            stream_data['telemetry']['linear_vel'] = float(msg.linear.x)
            stream_data['telemetry']['angular_vel'] = float(msg.angular.z)
            stream_data['telemetry']['last_vel_update'] = now


def run_http_server(server):
    try:
        server.serve_forever()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = DashboardSubscriber()

    httpd = ThreadedHTTPServer(('0.0.0.0', PORT), ViewerHandler)
    server_thread = threading.Thread(target=run_http_server, args=(httpd,), daemon=True)
    server_thread.start()

    print(f'[INFO] High-FPS Web Dashboard running at http://0.0.0.0:{PORT}')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
