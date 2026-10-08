#!/usr/bin/env python3
"""
Lightweight Web Viewer & Telemetry Dashboard for UEH Team 7 Robot
Streams /camera/image_raw and /lane_debug/image via low-latency MJPEG.
Also reports real-time battery voltage (%) and robot velocity (/cmd_vel).
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
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

PORT = int(os.environ.get("VIEWER_PORT", 8080))
TARGET_FPS = float(os.environ.get("VIEWER_FPS", 5.0))
MIN_INTERVAL = 1.0 / max(0.5, TARGET_FPS)

def make_placeholder_frame(title="Waiting for stream...", subtitle="Robot stream will appear here"):
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (20, 24, 33)
    cv2.putText(img, "UEH Team 7", (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2, cv2.LINE_AA)
    cv2.putText(img, title, (40, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (241, 245, 249), 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (40, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return buf.tobytes() if ret else None


# Shared State
state_lock = threading.Lock()
stream_data = {
    'lane': {
        'jpeg': make_placeholder_frame("Waiting for /lane_debug/image...", "Node starter is acquiring lane"),
        'ts': time.monotonic(),
        'fps': 0.0,
        'clients': 0
    },
    'camera': {
        'jpeg': make_placeholder_frame("Waiting for /camera/image_raw...", "Checking camera driver"),
        'ts': time.monotonic(),
        'fps': 0.0,
        'clients': 0
    },
    'telemetry': {
        'voltage': 0.0,
        'battery_pct': 0.0,
        'linear_vel': 0.0,
        'angular_vel': 0.0,
        'last_voltage_update': 0.0,
        'last_vel_update': 0.0,
    }
}


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

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
  <title>UEH Team 7 - Robot Live Telemetry & Camera</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0b0f19;
      color: #e2e8f0;
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
      max-width: 1100px;
      margin-bottom: 16px;
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
      font-weight: 700;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 15px;
      letter-spacing: 0.5px;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
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
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.08);
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
    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.35; }}
    }}

    /* Control Tabs */
    .view-controls {{
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
      width: 100%;
      max-width: 1100px;
    }}
    .btn {{
      padding: 8px 16px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      background: #1e2433;
      color: #94a3b8;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .btn:hover {{
      background: #283046;
      color: #f1f5f9;
    }}
    .btn.active {{
      background: #2563eb;
      color: white;
      border-color: #3b82f6;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }}

    /* Video Cards Grid */
    .stream-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      width: 100%;
      max-width: 1100px;
    }}
    .stream-grid.single {{
      grid-template-columns: 1fr;
      max-width: 800px;
    }}
    .video-card {{
      background: #141926;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
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
      letter-spacing: 0.3px;
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
    .video-wrapper {{
      width: 100%;
      aspect-ratio: 4 / 3;
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
      background: rgba(0, 0, 0, 0.2);
    }}
    .card-footer code {{
      color: #94a3b8;
      font-family: monospace;
    }}

    @media (max-width: 768px) {{
      .stream-grid {{
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
      <h1>Live Telemetry & Camera</h1>
    </div>
    <div class="status-group">
      <div class="pill pill-battery" id="battery-pill">
        <span>⚡ PIN: <strong id="val-battery">--%</strong> (<span id="val-volts">--V</span>)</span>
      </div>
      <div class="pill pill-speed" id="speed-pill">
        <span>🏎️ v: <strong id="val-v">0.00</strong> m/s | w: <strong id="val-w">0.00</strong> rad/s</span>
      </div>
      <div class="pill">
        <div class="status-dot"></div>
        <span>ROS2 DOMAIN 7</span>
      </div>
    </div>
  </div>

  <div class="view-controls">
    <button class="btn active" id="btn-dual" onclick="setView('dual')">📺 Song Song (Dual)</button>
    <button class="btn" id="btn-lane" onclick="setView('lane')">🛣️ Phân tích Làn (Lane Debug)</button>
    <button class="btn" id="btn-cam" onclick="setView('cam')">📷 Mắt Xe (Raw Camera)</button>
  </div>

  <div class="stream-grid" id="stream-container">
    <!-- Card 1: Lane Debug -->
    <div class="video-card" id="card-lane">
      <div class="card-header">
        <div class="card-title">
          <span>🛣️ Thuật toán bám làn (Lane Debug)</span>
        </div>
        <span class="tag tag-blue">Overlay &amp; Target</span>
      </div>
      <div class="video-wrapper">
        <img id="img-lane" src="/stream/lane" alt="Lane Debug Stream" />
      </div>
      <div class="card-footer">
        <div>Topic: <code>/lane_debug/image</code></div>
        <div>Tốc độ: <code>~{TARGET_FPS:.0f} FPS</code></div>
      </div>
    </div>

    <!-- Card 2: Raw Camera -->
    <div class="video-card" id="card-cam">
      <div class="card-header">
        <div class="card-title">
          <span>📷 Mắt xe thật (IMX219 CSI Camera)</span>
        </div>
        <span class="tag tag-emerald">640 &times; 480 @ 30fps</span>
      </div>
      <div class="video-wrapper">
        <img id="img-cam" src="/stream/camera" alt="Raw Camera Stream" />
      </div>
      <div class="card-footer">
        <div>Topic: <code>/camera/image_raw</code></div>
        <div>Sensor: <code>IMX219 (CAM0)</code></div>
      </div>
    </div>
  </div>

  <script>
    function setView(mode) {{
      const grid = document.getElementById('stream-container');
      const cardLane = document.getElementById('card-lane');
      const cardCam = document.getElementById('card-cam');
      const bDual = document.getElementById('btn-dual');
      const bLane = document.getElementById('btn-lane');
      const bCam = document.getElementById('btn-cam');

      [bDual, bLane, bCam].forEach(b => b.classList.remove('active'));

      if (mode === 'dual') {{
        grid.className = 'stream-grid';
        cardLane.style.display = 'flex';
        cardCam.style.display = 'flex';
        bDual.classList.add('active');
      }} else if (mode === 'lane') {{
        grid.className = 'stream-grid single';
        cardLane.style.display = 'flex';
        cardCam.style.display = 'none';
        bLane.classList.add('active');
      }} else if (mode === 'cam') {{
        grid.className = 'stream-grid single';
        cardLane.style.display = 'none';
        cardCam.style.display = 'flex';
        bCam.classList.add('active');
      }}
    }}

    // Auto reconnect stream if interrupted
    function setupReconnect(imgId, streamUrl) {{
      const el = document.getElementById(imgId);
      el.onerror = () => {{
        setTimeout(() => {{
          el.src = streamUrl + '?t=' + Date.now();
        }}, 1000);
      }};
    }}
    setupReconnect('img-lane', '/stream/lane');
    setupReconnect('img-cam', '/stream/camera');

    // Poll Telemetry API every 1 second
    async function updateTelemetry() {{
      try {{
        const res = await fetch('/api/status');
        if (res.ok) {{
          const data = await res.json();
          document.getElementById('val-battery').innerText = data.battery_pct.toFixed(1) + '%';
          document.getElementById('val-volts').innerText = data.voltage.toFixed(2) + 'V';
          document.getElementById('val-v').innerText = data.linear_vel.toFixed(2);
          document.getElementById('val-w').innerText = data.angular_vel.toFixed(2);

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
    setInterval(updateTelemetry, 1000);
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
            self.wfile.write(json.encode if False else json.dumps(tel).encode('utf-8'))

        # 3. Stream MJPEG
        elif self.path.startswith('/stream/'):
            topic_key = 'camera' if 'camera' in self.path else 'lane'
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            with state_lock:
                stream_data[topic_key]['clients'] += 1

            last_sent_time = 0.0
            try:
                while True:
                    with state_lock:
                        jpeg_bytes = stream_data[topic_key]['jpeg']
                        ts = stream_data[topic_key]['ts']

                    if jpeg_bytes is not None and ts != last_sent_time:
                        last_sent_time = ts
                        header = (
                            b'--FRAME\r\n'
                            b'Content-Type: image/jpeg\r\n'
                            b'Content-Length: ' + str(len(jpeg_bytes)).encode('ascii') + b'\r\n\r\n'
                        )
                        self.wfile.write(header)
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                        self.wfile.flush()

                    time.sleep(0.04)
            except Exception:
                pass
            finally:
                with state_lock:
                    stream_data[topic_key]['clients'] = max(0, stream_data[topic_key]['clients'] - 1)

        # 4. Fallback legacy /stream
        elif self.path in ('/stream', '/stream?'):
            self.path = '/stream/lane'
            self.do_GET()

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
            Image, '/lane_debug/image', self.on_lane_image, 10)
        self.sub_cam = self.create_subscription(
            Image, '/camera/image_raw', self.on_cam_image, 10)
        self.sub_voltage = self.create_subscription(
            Float32, '/voltage', self.on_voltage, 10)
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.on_cmd_vel, 10)

        self.get_logger().info('Dashboard initialized for /camera/image_raw & /lane_debug/image')

    def on_lane_image(self, msg):
        now = time.monotonic()
        if now - self.last_lane_time < MIN_INTERVAL:
            return
        self.last_lane_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 480:
                cv_img = cv2.resize(cv_img, (480, 360), interpolation=cv2.INTER_AREA)
            ret, buf = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if ret:
                with state_lock:
                    stream_data['lane']['jpeg'] = buf.tobytes()
                    stream_data['lane']['ts'] = now
        except Exception:
            pass

    def on_cam_image(self, msg):
        now = time.monotonic()
        if now - self.last_cam_time < MIN_INTERVAL:
            return
        self.last_cam_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 480:
                cv_img = cv2.resize(cv_img, (480, 360), interpolation=cv2.INTER_AREA)
            ret, buf = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if ret:
                with state_lock:
                    stream_data['camera']['jpeg'] = buf.tobytes()
                    stream_data['camera']['ts'] = now
        except Exception:
            pass

    def on_voltage(self, msg):
        now = time.monotonic()
        volts = float(msg.data)
        # 3S LiPo: 12.6V = 100%, 9.6V = 0%
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

    print(f'[INFO] Web Dashboard running at http://0.0.0.0:{PORT}')

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
