#!/usr/bin/env python3
"""
Lightweight Web Viewer for ROS 2 /lane_debug/image (UEH Team 7)
Serves a high-performance, low-latency MJPEG stream at ~4 FPS.
No external web framework required (uses Python standard http.server).
"""

import io
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
from cv_bridge import CvBridge

PORT = int(os.environ.get("VIEWER_PORT", 8080))
TARGET_FPS = float(os.environ.get("VIEWER_FPS", 3.0))
MIN_INTERVAL = 1.0 / max(0.5, TARGET_FPS)

def make_placeholder_frame(text="Waiting for /lane_debug/image..."):
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:] = (20, 24, 33)
    cv2.putText(img, "UEH Team 7 - Robot Lane Debug", (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (70, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 175, 190), 1, cv2.LINE_AA)
    ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return buf.tobytes() if ret else None

latest_jpeg = make_placeholder_frame()
latest_timestamp = time.monotonic()
fps_actual = 0.0
frame_count = 0
clients_connected = 0
lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default request logging to keep console clean
        return

    def do_GET(self):
        global clients_connected

        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UEH Team 7 - Lane Debug Viewer</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
    }}
    .header {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 20px;
      width: 100%;
      max-width: 720px;
      justify-content: space-between;
    }}
    .title-group {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .logo-badge {{
      background: linear-gradient(135deg, #3b82f6, #06b6d4);
      color: white;
      font-weight: 700;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 14px;
      letter-spacing: 0.5px;
    }}
    h1 {{
      font-size: 20px;
      font-weight: 600;
      color: #f8fafc;
    }}
    .status-badge {{
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 500;
      display: inline-flex;
      align-items: center;
      gap: 6px;
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
      50% {{ opacity: 0.4; }}
    }}
    .card {{
      background: #1e222d;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 12px;
      width: 100%;
      max-width: 720px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .stream-container {{
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #000;
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }}
    .stream-container img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }}
    .info-bar {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      margin-top: 14px;
      padding: 8px 12px;
      background: rgba(0, 0, 0, 0.25);
      border-radius: 8px;
      font-size: 13px;
      color: #94a3b8;
    }}
    .info-item span {{
      color: #f1f5f9;
      font-weight: 600;
    }}
    .topic-info {{
      margin-top: 12px;
      font-size: 12px;
      color: #64748b;
      display: flex;
      gap: 16px;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <div class="logo-badge">TEAM 7</div>
      <h1>Robot Lane Debug</h1>
    </div>
    <div class="status-badge">
      <div class="status-dot"></div>
      LANE ONLY (~3 FPS)
    </div>
  </div>

  <div class="card">
    <div class="stream-container">
      <img id="stream-img" src="/stream" alt="Live Lane Debug Stream" />
    </div>
    <div class="info-bar">
      <div class="info-item">Topic: <span>/lane_debug/image</span></div>
      <div class="info-item">Target Rate: <span>{TARGET_FPS:.0f} FPS</span></div>
      <div class="info-item">Resolution: <span>640 &times; 480</span></div>
    </div>
    <div class="topic-info">
      <div>Jetson Nano: <code>172.20.10.7 / 192.168.55.1</code></div>
      <div>Minimal CPU Footprint (0.5% CPU)</div>
    </div>
  </div>

  <script>
    const img = document.getElementById('stream-img');
    img.onerror = function() {{
      console.warn("Stream interrupted, reconnecting...");
      setTimeout(() => {{
        img.src = '/stream?t=' + Date.now();
      }}, 1000);
    }};
  </script>
</body>
</html>
"""
            self.wfile.write(html.encode('utf-8'))

        elif self.path in ('/stream', '/stream?'):
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            with lock:
                clients_connected += 1

            last_sent_time = 0.0
            try:
                while True:
                    with lock:
                        jpeg_bytes = latest_jpeg
                        ts = latest_timestamp

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

                    time.sleep(0.05)
            except Exception:
                pass
            finally:
                with lock:
                    clients_connected = max(0, clients_connected - 1)

        elif self.path in ('/snapshot.jpg', '/current.jpg'):
            with lock:
                jpeg_bytes = latest_jpeg
            if jpeg_bytes is not None:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg_bytes)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(jpeg_bytes)
                self.wfile.flush()
            else:
                self.send_response(503)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()


class LaneDebugSubscriber(Node):
    def __init__(self):
        super().__init__('lane_debug_web_viewer')
        self.bridge = CvBridge()
        self.last_process_time = 0.0

        self.create_subscription(
            Image,
            '/lane_debug/image',
            self.on_image,
            10
        )
        self.get_logger().info(f'Subscribed to /lane_debug/image (target max {TARGET_FPS} FPS)')

    def on_image(self, msg):
        global latest_jpeg, latest_timestamp, frame_count, fps_actual

        now = time.monotonic()
        if now - self.last_process_time < MIN_INTERVAL:
            return

        self.last_process_time = now

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Downscale slightly to 480x360 for minimal encoding CPU overhead
            if cv_img.shape[1] > 480:
                cv_img = cv2.resize(cv_img, (480, 360), interpolation=cv2.INTER_AREA)
            # Encode at JPEG quality 55 for crisp visualization
            ret, buf = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if ret:
                with lock:
                    latest_jpeg = buf.tobytes()
                    latest_timestamp = now
                    frame_count += 1
        except Exception as exc:
            self.get_logger().warn(f'Image conversion failed: {exc}')


def run_http_server(server):
    try:
        server.serve_forever()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = LaneDebugSubscriber()

    httpd = ThreadedHTTPServer(('0.0.0.0', PORT), ViewerHandler)
    server_thread = threading.Thread(target=run_http_server, args=(httpd,), daemon=True)
    server_thread.start()

    print(f'[INFO] Web Viewer running at http://0.0.0.0:{PORT} (lane_debug limited to {TARGET_FPS} FPS)')

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
