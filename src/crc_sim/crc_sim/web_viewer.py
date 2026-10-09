#!/usr/bin/env python3
"""
Ultra Low-Latency Web Viewer & Interactive Mission Control for UEH Team 7 Robot
- Real-time Hardware JPEG streaming (25-30 FPS)
- Master Vehicle Arming & Safety: RUN AUTO / EMERGENCY STOP
- Manual Keyboard Teleoperation (Arrow keys / WASD / Touch D-Pad)
- Dual & Single Camera Views with live telemetry
"""

import io
import json
import os
import sys
import time
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

PORT = int(os.environ.get("VIEWER_PORT", 8080))
TARGET_FPS = float(os.environ.get("VIEWER_FPS", 25.0))
MIN_INTERVAL = 1.0 / max(1.0, TARGET_FPS)

# Global node reference for HTTP request handler
global_dashboard_node = None

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
        'robot_mode': 'STOPPED',   # 'STOPPED', 'AUTO', 'MANUAL'
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
        global TARGET_FPS, MIN_INTERVAL
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        params = urllib.parse.parse_qs(parsed_url.query)

        # 1. Main Dashboard HTML
        if path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UEH Team 7 - Robot Mission Control & Camera</title>
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
      padding: 14px;
      user-select: none;
    }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      max-width: 1140px;
      margin-bottom: 12px;
      flex-wrap: wrap;
      gap: 10px;
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
      font-size: 19px;
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
    .pill-mode {{
      font-weight: 700;
      letter-spacing: 0.5px;
    }}
    .mode-stopped {{
      background: rgba(239, 68, 68, 0.2);
      border-color: rgba(239, 68, 68, 0.5);
      color: #f87171;
    }}
    .mode-auto {{
      background: rgba(16, 185, 129, 0.2);
      border-color: rgba(16, 185, 129, 0.5);
      color: #34d399;
      box-shadow: 0 0 12px rgba(16, 185, 129, 0.3);
    }}
    .mode-manual {{
      background: rgba(168, 85, 247, 0.2);
      border-color: rgba(168, 85, 247, 0.5);
      color: #c084fc;
      box-shadow: 0 0 12px rgba(168, 85, 247, 0.3);
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

    /* MASTER VEHICLE CONTROL BAR */
    .master-controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      max-width: 1140px;
      margin-bottom: 12px;
      padding: 10px 14px;
      background: #111827;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .btn-action {{
      padding: 10px 20px;
      font-size: 14px;
      font-weight: 700;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease-in-out;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }}
    .btn-run {{
      background: linear-gradient(135deg, #10b981, #059669);
      color: white;
    }}
    .btn-run:hover {{
      background: linear-gradient(135deg, #34d399, #10b981);
      box-shadow: 0 0 16px rgba(16, 185, 129, 0.5);
      transform: translateY(-1px);
    }}
    .btn-stop {{
      background: linear-gradient(135deg, #ef4444, #dc2626);
      color: white;
    }}
    .btn-stop:hover {{
      background: linear-gradient(135deg, #f87171, #ef4444);
      box-shadow: 0 0 16px rgba(239, 68, 68, 0.5);
      transform: translateY(-1px);
    }}
    .btn-manual {{
      background: linear-gradient(135deg, #8b5cf6, #7c3aed);
      color: white;
    }}
    .btn-manual:hover {{
      background: linear-gradient(135deg, #a78bfa, #8b5cf6);
      box-shadow: 0 0 16px rgba(139, 92, 246, 0.5);
      transform: translateY(-1px);
    }}
    .btn-manual.active {{
      outline: 2px solid #c084fc;
      box-shadow: 0 0 18px rgba(192, 132, 252, 0.6);
    }}

    /* View Controls */
    .view-controls {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      width: 100%;
      max-width: 1140px;
      flex-wrap: wrap;
    }}
    .btn {{
      padding: 8px 14px;
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
    }}
    .btn.active {{
      background: #2563eb;
      color: white;
      border-color: #3b82f6;
    }}

    /* Stream Containers */
    .stream-container {{
      width: 100%;
      max-width: 1140px;
      margin-bottom: 14px;
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
    .tag-blue {{ background: rgba(37, 99, 235, 0.2); color: #60a5fa; }}
    .tag-emerald {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .tag-purple {{ background: rgba(147, 51, 234, 0.2); color: #c084fc; }}
    .video-wrapper {{
      width: 100%;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
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
    .grid-split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}

    /* MANUAL TELEOP WIDGET */
    .teleop-panel {{
      width: 100%;
      max-width: 1140px;
      background: #111827;
      border: 1px solid rgba(168, 85, 247, 0.3);
      border-radius: 12px;
      padding: 14px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .teleop-info {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .teleop-title {{
      font-size: 14px;
      font-weight: 700;
      color: #c084fc;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .kbd-guide {{
      display: flex;
      gap: 8px;
      font-size: 12px;
      color: #94a3b8;
      flex-wrap: wrap;
    }}
    kbd {{
      background: #1f293d;
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 4px;
      padding: 2px 6px;
      color: #e2e8f0;
      font-family: monospace;
      font-weight: 700;
    }}
    .dpad-container {{
      display: grid;
      grid-template-columns: 50px 50px 50px;
      grid-template-rows: 45px 45px;
      gap: 6px;
      justify-content: center;
    }}
    .dpad-btn {{
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 8px;
      color: #f8fafc;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.1s;
    }}
    .dpad-btn:active, .dpad-btn.pressed {{
      background: #8b5cf6;
      color: white;
      transform: scale(0.95);
      box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    }}
    .dpad-stop {{
      background: rgba(239, 68, 68, 0.2);
      border-color: rgba(239, 68, 68, 0.4);
      color: #f87171;
    }}

    @media (max-width: 768px) {{
      .grid-split {{ grid-template-columns: 1fr; }}
      .header {{ flex-direction: column; align-items: flex-start; }}
      .master-controls {{ flex-direction: column; align-items: stretch; }}
    }}
  </style>
</head>
<body>

  <!-- HEADER -->
  <div class="header">
    <div class="title-group">
      <div class="logo-badge">TEAM 7</div>
      <h1>Robot Mission Control & Live Feed</h1>
    </div>
    <div class="status-group">
      <div class="pill pill-mode mode-stopped" id="mode-pill">
        <span>🛑 TRẠNG THÁI: <strong id="val-mode">ĐỨNG YÊN (CHỜ LỆNH)</strong></span>
      </div>
      <div class="pill pill-battery" id="battery-pill">
        <span>⚡ PIN: <strong id="val-battery">--%</strong> (<span id="val-volts">--V</span>)</span>
      </div>
      <div class="pill pill-speed" id="speed-pill">
        <span>🏎️ v: <strong id="val-v">0.00</strong> m/s | w: <strong id="val-w">0.00</strong></span>
      </div>
      <div class="pill pill-fps" id="fps-pill">
        <span>📡 <strong id="val-fps">--</strong></span>
      </div>
    </div>
  </div>

  <!-- MASTER VEHICLE CONTROL BAR -->
  <div class="master-controls">
    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
      <button class="btn-action btn-run" id="btn-run" onclick="sendControl('run')">
        <span>▶️ RUN AUTO (Bắt Đầu Tự Hành)</span>
      </button>
      <button class="btn-action btn-stop" id="btn-stop" onclick="sendControl('stop')">
        <span>⏹️ EMERGENCY STOP (Dừng Xe)</span>
      </button>
      <button class="btn-action btn-manual" id="btn-manual" onclick="sendControl('manual')">
        <span>🎮 LÁI BÀN PHÍM (Manual)</span>
      </button>
    </div>
    <div style="font-size: 12px; color: #94a3b8; font-weight: 600;">
      <span>Bánh xe CHỈ quay khi bạn nhấn RUN hoặc bấm phím lái</span>
    </div>
  </div>

  <!-- MANUAL TELEOPERATION WIDGET -->
  <div class="teleop-panel" id="teleop-panel" style="display: none;">
    <div class="teleop-info">
      <div class="teleop-title">
        <span>🎮 ĐIỀU KHIỂN THỦ CÔNG (MANUAL ACTIVE)</span>
      </div>
      <div class="kbd-guide">
        <span>Tiến: <kbd>↑</kbd> hoặc <kbd>W</kbd></span>
        <span>Lùi: <kbd>↓</kbd> hoặc <kbd>S</kbd></span>
        <span>Rẽ Trái: <kbd>←</kbd> hoặc <kbd>A</kbd></span>
        <span>Rẽ Phải: <kbd>→</kbd> hoặc <kbd>D</kbd></span>
        <span>Dừng: <kbd>Space</kbd> / Nhả phím</span>
      </div>
    </div>
    <!-- D-PAD VIRTUAL BUTTONS -->
    <div class="dpad-container">
      <div></div>
      <button class="dpad-btn" id="dpad-up" onmousedown="startTeleop(0.22, 0.0)" onmouseup="stopTeleop()" ontouchstart="startTeleop(0.22, 0.0)" ontouchend="stopTeleop()">⬆️</button>
      <div></div>
      <button class="dpad-btn" id="dpad-left" onmousedown="startTeleop(0.12, 0.65)" onmouseup="stopTeleop()" ontouchstart="startTeleop(0.12, 0.65)" ontouchend="stopTeleop()">⬅️</button>
      <button class="dpad-btn dpad-stop" id="dpad-center" onclick="sendControl('stop')">⏹️</button>
      <button class="dpad-btn" id="dpad-right" onmousedown="startTeleop(0.12, -0.65)" onmouseup="stopTeleop()" ontouchstart="startTeleop(0.12, -0.65)" ontouchend="stopTeleop()">➡️</button>
    </div>
  </div>

  <!-- VIEW & FPS CONTROLS -->
  <div class="view-controls">
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
      <button class="btn active" id="btn-dual" onclick="setView('dual')">🚀 Song Song Ghép 1 Stream</button>
      <button class="btn" id="btn-lane" onclick="setView('lane')">🛣️ Chỉ Xem Làn</button>
      <button class="btn" id="btn-cam" onclick="setView('cam')">📷 Chỉ Xem Mắt Xe</button>
      <button class="btn" id="btn-split" onclick="setView('split')">👥 2 Khung Riêng Biệt</button>
    </div>
    <div style="display: flex; gap: 6px; align-items: center; margin-left: auto; flex-wrap: wrap;">
      <span style="font-size: 12px; color: #94a3b8; font-weight: 600;">FPS STREAM:</span>
      <button class="btn btn-fps" id="fps-15" onclick="changeFps(15)">15</button>
      <button class="btn btn-fps" id="fps-20" onclick="changeFps(20)">20</button>
      <button class="btn btn-fps active" id="fps-25" onclick="changeFps(25)">25 (Chuẩn)</button>
      <button class="btn btn-fps" id="fps-30" onclick="changeFps(30)">30 (Max)</button>
    </div>
  </div>

  <!-- STREAMS -->
  <div class="stream-container">
    <!-- View 1: Unified Dual Stream -->
    <div class="video-card" id="view-dual" style="display: flex;">
      <div class="card-header">
        <div class="card-title">
          <span>📺 Kênh Ghép Đồng Bộ: Trái (Phân Tích Làn) | Phải (Camera Thực)</span>
        </div>
        <span class="tag tag-purple">Single HTTP Stream ~25 FPS</span>
      </div>
      <div class="video-wrapper" style="aspect-ratio: 8 / 3;">
        <img id="img-dual" src="/stream/dual" alt="Dual Stream" />
      </div>
      <div class="card-footer">
        <div>Truyền dẫn: <code>/stream/dual (1 socket duy nhất)</code></div>
        <div>Mạng: <code>Wi-Fi Hotspot Kev (172.20.10.2) / USB (192.168.55.1)</code></div>
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
        <div>Nguồn: <code>starter_node.py (Lookahead &amp; Curvature)</code></div>
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
        <div>Nguồn: <code>/camera/image_raw/compressed (Pass-through)</code></div>
        <div>Cảm biến: <code>IMX219 (CSI CAM0)</code></div>
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
    let currentMode = 'STOPPED';
    let teleopTimer = null;
    let activeKeys = {{}};

    async function sendControl(action) {{
      try {{
        const res = await fetch('/api/control?action=' + action);
        if (res.ok) {{
          const data = await res.json();
          updateModeUI(data.mode);
        }}
      }} catch (e) {{}}
    }}

    function updateModeUI(mode) {{
      currentMode = mode;
      const pill = document.getElementById('mode-pill');
      const val = document.getElementById('val-mode');
      const panel = document.getElementById('teleop-panel');
      const btnMan = document.getElementById('btn-manual');

      pill.className = 'pill pill-mode';
      if (mode === 'AUTO') {{
        pill.classList.add('mode-auto');
        val.innerText = 'TỰ HÀNH (AUTO RUNNING)';
        panel.style.display = 'none';
        btnMan.classList.remove('active');
      }} else if (mode === 'MANUAL') {{
        pill.classList.add('mode-manual');
        val.innerText = 'LÁI THỦ CÔNG (MANUAL KEYBOARD)';
        panel.style.display = 'flex';
        btnMan.classList.add('active');
      }} else {{
        pill.classList.add('mode-stopped');
        val.innerText = 'ĐỨNG YÊN (CHỜ LỆNH)';
        panel.style.display = 'none';
        btnMan.classList.remove('active');
      }}
    }}

    function startTeleop(v, w) {{
      if (currentMode !== 'MANUAL') {{
        sendControl('manual').then(() => doSendTeleop(v, w));
      }} else {{
        doSendTeleop(v, w);
      }}
      if (teleopTimer) clearInterval(teleopTimer);
      teleopTimer = setInterval(() => doSendTeleop(v, w), 80);
    }}

    function stopTeleop() {{
      if (teleopTimer) {{
        clearInterval(teleopTimer);
        teleopTimer = null;
      }}
      doSendTeleop(0.0, 0.0);
    }}

    async function doSendTeleop(v, w) {{
      try {{
        await fetch('/api/teleop?v=' + v + '&w=' + w);
      }} catch (e) {{}}
    }}

    // KEYBOARD TELEOP LISTENER
    window.addEventListener('keydown', (e) => {{
      const key = e.key.toLowerCase();
      if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 's', 'a', 'd', ' '].includes(key)) {{
        e.preventDefault();
      }} else {{
        return;
      }}

      if (key === ' ') {{
        sendControl('stop');
        return;
      }}

      if (!activeKeys[key]) {{
        activeKeys[key] = true;
        evaluateKeyboardTeleop();
      }}
    }});

    window.addEventListener('keyup', (e) => {{
      const key = e.key.toLowerCase();
      if (activeKeys[key]) {{
        delete activeKeys[key];
        evaluateKeyboardTeleop();
      }}
    }});

    function evaluateKeyboardTeleop() {{
      let v = 0.0;
      let w = 0.0;
      if (activeKeys['arrowup'] || activeKeys['w']) v += 0.22;
      if (activeKeys['arrowdown'] || activeKeys['s']) v -= 0.16;
      if (activeKeys['arrowleft'] || activeKeys['a']) w += 0.65;
      if (activeKeys['arrowright'] || activeKeys['d']) w -= 0.65;

      if (v !== 0.0 || w !== 0.0) {{
        startTeleop(v, w);
      }} else {{
        stopTeleop();
      }}
    }}

    function setView(mode) {{
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

    async function changeFps(target) {{
      try {{
        const res = await fetch('/api/set_fps?fps=' + target);
        if (res.ok) {{
          document.querySelectorAll('.btn-fps').forEach(b => b.classList.remove('active'));
          const btn = document.getElementById('fps-' + target);
          if (btn) btn.classList.add('active');
        }}
      }} catch (e) {{}}
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

          if (data.robot_mode && data.robot_mode !== currentMode) {{
            updateModeUI(data.robot_mode);
          }}

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

    setInterval(updateTelemetry, 700);
    updateTelemetry();
  </script>
</body>
</html>
"""
            self.wfile.write(html.encode('utf-8'))

        # 2. Control API (/api/control?action=run|stop|manual)
        elif path == '/api/control':
            action = params.get('action', ['stop'])[0].lower()
            target_mode = 'STOPPED'
            if action in ('run', 'start', 'auto'):
                target_mode = 'AUTO'
            elif action == 'manual':
                target_mode = 'MANUAL'
            else:
                target_mode = 'STOPPED'

            if global_dashboard_node is not None:
                global_dashboard_node.set_mode(target_mode)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'mode': target_mode}).encode('utf-8'))

        # 3. Teleop API (/api/teleop?v=...&w=...)
        elif path == '/api/teleop':
            v = float(params.get('v', [0.0])[0])
            w = float(params.get('w', [0.0])[0])
            if global_dashboard_node is not None:
                global_dashboard_node.teleop_cmd(v, w)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'v': v, 'w': w}).encode('utf-8'))

        # 4. Telemetry Set FPS API
        elif path == '/api/set_fps':
            try:
                new_fps = float(params.get('fps', [25.0])[0])
                TARGET_FPS = max(5.0, min(30.0, new_fps))
                MIN_INTERVAL = 1.0 / TARGET_FPS
                with state_lock:
                    stream_data['telemetry']['target_fps'] = TARGET_FPS
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok', 'fps': TARGET_FPS}).encode('utf-8'))
            except Exception:
                self.send_response(500)
                self.end_headers()

        # 5. Telemetry JSON API
        elif path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            with state_lock:
                tel = dict(stream_data['telemetry'])
            self.wfile.write(json.dumps(tel).encode('utf-8'))

        # 6. Stream MJPEG (/stream/lane, /stream/camera, /stream/dual)
        elif path.startswith('/stream'):
            if 'dual' in path:
                channel = 'dual'
            elif 'camera' in path:
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
        global global_dashboard_node
        global_dashboard_node = self

        self.bridge = CvBridge()
        self.last_lane_time = 0.0
        self.last_cam_time = 0.0

        # Control Publishers
        self.pub_mode = self.create_publisher(String, '/team_control/mode', 10)
        self.pub_teleop_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control Subscription (Sync with other nodes if mode is toggled)
        self.sub_mode = self.create_subscription(
            String, '/team_control/mode', self.on_mode_msg, 10)

        # Telemetry Subscriptions
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
        self.get_logger().info(f'Dashboard Mission Control ready on port {PORT} (Armed: STOPPED)')

    def on_mode_msg(self, msg):
        mode = msg.data.strip().upper()
        with state_lock:
            stream_data['telemetry']['robot_mode'] = mode

    def set_mode(self, mode):
        mode = mode.upper()
        with state_lock:
            stream_data['telemetry']['robot_mode'] = mode
        msg = String()
        msg.data = mode
        self.pub_mode.publish(msg)
        if mode in ('STOPPED', 'STOP', 'MANUAL'):
            self.pub_teleop_cmd.publish(Twist())
        self.get_logger().info(f'Vehicle Mode changed to: {mode}')

    def teleop_cmd(self, v, w):
        with state_lock:
            cur_mode = stream_data['telemetry']['robot_mode']
        if cur_mode == 'MANUAL':
            t = Twist()
            t.linear.x = float(max(-0.35, min(0.35, v)))
            t.angular.z = float(max(-1.0, min(1.0, w)))
            self.pub_teleop_cmd.publish(t)

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
            stream_data['lane']['raw_bgr'] = cv_img

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
        self.has_compressed_cam = True
        now = time.monotonic()
        if now - self.last_cam_time < MIN_INTERVAL:
            return
        self.last_cam_time = now

        jpeg_data = bytes(msg.data)
        
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
        if self.has_compressed_cam:
            return

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

    print(f'[INFO] Interactive Mission Control running at http://0.0.0.0:{PORT}')

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
