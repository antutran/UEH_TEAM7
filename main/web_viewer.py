#!/usr/bin/env python3
"""
Ultra Low-Latency Web Viewer & Telemetry Dashboard for UEH Team 7 Robot
- Direct pass-through of hardware-compressed JPEG (/camera/image_raw/compressed)
- Zero-latency Condition notification for MJPEG delivery
- High frame rate (up to 20-25 FPS) with adaptive JPEG encoding
- Unified Single-Stream Dual View (/stream/dual) preventing multi-socket Wi-Fi congestion
- Real-time telemetry: Battery (V, %), Speed (v, w), Gyro Z, Stream FPS
- Autonomous Mode (START / STOP) & Manual Calibration Mode (Keyboard Arrow keys / D-Pad)
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
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage, Imu
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

PORT = int(os.environ.get("VIEWER_PORT", 8080))
TARGET_FPS = float(os.environ.get("VIEWER_FPS", 25.0))
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
        'gyro_z': 0.0,
        'lane_fps': 0.0,
        'cam_fps': 0.0,
        'target_fps': TARGET_FPS,
        'car_state': 'STANDBY',
        'last_voltage_update': 0.0,
        'last_vel_update': 0.0,
    }
}

g_dashboard_node = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # Suppress request spam

    def do_GET(self):
        global TARGET_FPS, MIN_INTERVAL
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
  <title>UEH Team 7 - Live Robot Dashboard & Calibration</title>
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
    }}
    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      max-width: 1140px;
      margin-bottom: 12px;
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
    .pill-gyro {{
      background: rgba(168, 85, 247, 0.15);
      border-color: rgba(168, 85, 247, 0.35);
      color: #c084fc;
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

    /* Main Mode Tabs: Auto vs Manual Calib */
    .mode-tab-bar {{
      display: flex;
      width: 100%;
      max-width: 1140px;
      background: #111827;
      padding: 6px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      margin-bottom: 12px;
      gap: 6px;
    }}
    .tab-btn {{
      flex: 1;
      padding: 10px 18px;
      border-radius: 8px;
      border: none;
      background: transparent;
      color: #94a3b8;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s ease;
    }}
    .tab-btn:hover {{
      color: #f1f5f9;
      background: rgba(255, 255, 255, 0.05);
    }}
    .tab-btn.active {{
      background: #2563eb;
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }}
    .tab-btn.active.tab-manual {{
      background: linear-gradient(135deg, #9333ea, #7c3aed);
      box-shadow: 0 4px 14px rgba(147, 51, 234, 0.45);
    }}

    /* Panel: AUTO CONTROLS */
    .control-panel {{
      width: 100%;
      max-width: 1140px;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      padding: 14px 20px;
      margin-bottom: 12px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45);
    }}
    .auto-panel-inner {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .ctrl-left {{
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .ctrl-right {{
      display: flex;
      align-items: center;
      gap: 14px;
      margin-left: auto;
      flex-wrap: wrap;
    }}
    .ctrl-label {{
      font-size: 13px;
      font-weight: 800;
      color: #94a3b8;
      letter-spacing: 0.5px;
    }}
    .btn-ctrl {{
      padding: 12px 28px;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 800;
      cursor: pointer;
      border: none;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .btn-start {{
      background: linear-gradient(135deg, #10b981, #059669);
      color: #ffffff;
      box-shadow: 0 4px 16px rgba(16, 185, 129, 0.45);
    }}
    .btn-start:hover {{
      background: linear-gradient(135deg, #34d399, #10b981);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.65);
    }}
    .btn-start:active {{
      transform: translateY(1px);
    }}
    .btn-stop {{
      background: linear-gradient(135deg, #ef4444, #dc2626);
      color: #ffffff;
      box-shadow: 0 4px 16px rgba(239, 68, 68, 0.45);
    }}
    .btn-stop:hover {{
      background: linear-gradient(135deg, #f87171, #ef4444);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(239, 68, 68, 0.65);
    }}
    .btn-stop:active {{
      transform: translateY(1px);
    }}

    .car-status-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 18px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.5px;
      border: 1px solid;
      transition: all 0.3s ease;
    }}
    .badge-standby {{
      background: rgba(245, 158, 11, 0.15);
      border-color: rgba(245, 158, 11, 0.45);
      color: #fbbf24;
    }}
    .badge-standby .badge-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #f59e0b;
      box-shadow: 0 0 10px #f59e0b;
    }}
    .badge-running {{
      background: rgba(16, 185, 129, 0.22);
      border-color: rgba(16, 185, 129, 0.6);
      color: #34d399;
      animation: pulse-glow 2s infinite;
    }}
    .badge-running .badge-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 12px #10b981;
    }}
    .badge-manual {{
      background: rgba(168, 85, 247, 0.22);
      border-color: rgba(168, 85, 247, 0.6);
      color: #c084fc;
      animation: pulse-glow-purple 2s infinite;
    }}
    .badge-manual .badge-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #a855f7;
      box-shadow: 0 0 12px #a855f7;
    }}

    @keyframes pulse-glow {{
      0%, 100% {{ box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }}
      50% {{ box-shadow: 0 0 25px rgba(16, 185, 129, 0.65); }}
    }}
    @keyframes pulse-glow-purple {{
      0%, 100% {{ box-shadow: 0 0 15px rgba(168, 85, 247, 0.3); }}
      50% {{ box-shadow: 0 0 25px rgba(168, 85, 247, 0.65); }}
    }}
    .ctrl-hint {{
      font-size: 12px;
      color: #64748b;
    }}

    /* Panel: MANUAL CALIBRATION CONTROLS */
    .manual-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: start;
    }}
    @media (max-width: 900px) {{
      .manual-grid {{ grid-template-columns: 1fr; }}
    }}

    .dpad-container {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      background: #0f172a;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .dpad-row {{
      display: flex;
      gap: 10px;
      justify-content: center;
      align-items: center;
    }}
    .btn-dpad {{
      width: 74px;
      height: 60px;
      border-radius: 10px;
      border: 1px solid rgba(255, 255, 255, 0.15);
      background: #1e293b;
      color: #f1f5f9;
      font-size: 18px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 3px;
      transition: all 0.15s ease;
      user-select: none;
    }}
    .btn-dpad span {{
      font-size: 10px;
      font-weight: 600;
      color: #94a3b8;
    }}
    .btn-dpad:hover {{
      background: #334155;
      border-color: rgba(255, 255, 255, 0.3);
    }}
    .btn-dpad:active, .btn-dpad.active {{
      background: #2563eb;
      color: white;
      transform: scale(0.96);
      box-shadow: 0 0 16px rgba(37, 99, 235, 0.6);
    }}
    .btn-dpad-fwd {{
      width: 158px;
      background: linear-gradient(135deg, #1d4ed8, #2563eb);
      border-color: #3b82f6;
    }}
    .btn-dpad-fwd:hover {{
      background: linear-gradient(135deg, #2563eb, #3b82f6);
    }}
    .btn-dpad-fwd.active {{
      background: #10b981 !important;
      box-shadow: 0 0 20px rgba(16, 185, 129, 0.7) !important;
    }}
    .btn-dpad-stop {{
      background: #ef4444;
      border-color: #f87171;
    }}
    .btn-dpad-stop:hover {{
      background: #dc2626;
    }}

    .calib-settings {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: #0f172a;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .setting-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .preset-group {{
      display: flex;
      gap: 6px;
    }}
    .btn-preset {{
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: #1e293b;
      color: #94a3b8;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .btn-preset:hover {{
      color: #fff;
      background: #334155;
    }}
    .btn-preset.active {{
      background: #2563eb;
      color: #fff;
      border-color: #3b82f6;
    }}
    .slider-box {{
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
    }}
    .slider-box input[type="range"] {{
      flex: 1;
      accent-color: #3b82f6;
      cursor: pointer;
    }}
    .slider-val {{
      font-family: monospace;
      font-size: 14px;
      font-weight: 700;
      color: #60a5fa;
      min-width: 65px;
      text-align: right;
    }}
    .calib-guide-box {{
      margin-top: 8px;
      background: rgba(147, 51, 234, 0.1);
      border: 1px solid rgba(147, 51, 234, 0.3);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 12px;
      line-height: 1.6;
      color: #e2e8f0;
    }}
    .calib-guide-box strong {{
      color: #c084fc;
    }}

    /* Live Direction Evaluation Pill */
    .eval-box {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      background: #141b2d;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      font-size: 13px;
    }}
    .eval-badge {{
      font-weight: 800;
      padding: 3px 10px;
      border-radius: 6px;
    }}
    .eval-straight {{
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
    }}
    .eval-left {{
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
    }}
    .eval-right {{
      background: rgba(245, 158, 11, 0.2);
      color: #fbbf24;
    }}

    /* View Mode Tabs */
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
    .tag-blue {{ background: rgba(37, 99, 235, 0.2); color: #60a5fa; }}
    .tag-emerald {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
    .tag-purple {{ background: rgba(147, 51, 234, 0.2); color: #c084fc; }}
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
    .grid-split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
  </style>
</head>
<body>

  <!-- HEADER -->
  <div class="header">
    <div class="title-group">
      <div class="logo-badge">TEAM 7</div>
      <h1>Ultra-Smooth Robot Telemetry & Calibration</h1>
    </div>
    <div class="status-group">
      <div class="pill pill-battery" id="battery-pill">
        <span>⚡ PIN: <strong id="val-battery">--%</strong> (<span id="val-volts">--V</span>)</span>
      </div>
      <div class="pill pill-speed" id="speed-pill">
        <span>🏎️ v: <strong id="val-v">0.00</strong> m/s | w: <strong id="val-w">0.00</strong></span>
      </div>
      <div class="pill pill-gyro" id="gyro-pill">
        <span>🧭 Gyro Z: <strong id="val-gyro">0.000</strong> rad/s</span>
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

  <!-- MODE SELECTOR TABS -->
  <div class="mode-tab-bar">
    <button class="tab-btn active" id="tab-auto-btn" onclick="switchControlMode('auto')">
      🤖 CHẾ ĐỘ TỰ ĐỘNG (AUTO RACE)
    </button>
    <button class="tab-btn tab-manual" id="tab-manual-btn" onclick="switchControlMode('manual')">
      🎮 HIỆU CHUẨN THỦ CÔNG (MANUAL CALIB)
    </button>
  </div>

  <!-- PANEL 1: AUTO MODE (START / STOP) -->
  <div class="control-panel" id="panel-auto" style="display: block;">
    <div class="auto-panel-inner">
      <div class="ctrl-left">
        <span class="ctrl-label">LỆNH XUẤT PHÁT:</span>
        <button class="btn-ctrl btn-start" id="btn-start" onclick="sendCarCmd('start')">
          ▶️ XUẤT PHÁT (START)
        </button>
        <button class="btn-ctrl btn-stop" id="btn-stop" onclick="sendCarCmd('stop')">
          ⏹️ DỪNG XE (STOP)
        </button>
      </div>
      <div class="ctrl-right">
        <div class="car-status-badge badge-standby" id="car-status-badge">
          <span class="badge-dot"></span>
          <span id="car-status-text">CHỜ LỆNH (STANDBY)</span>
        </div>
        <span class="ctrl-hint">⌨️ Phím tắt: [Space] Chạy/Dừng | [R] Chạy | [S] Dừng</span>
      </div>
    </div>
  </div>

  <!-- PANEL 2: MANUAL CALIBRATION MODE (ARROW KEYS & D-PAD) -->
  <div class="control-panel" id="panel-manual" style="display: none;">
    <div class="manual-grid">
      <!-- Left: D-Pad -->
      <div class="dpad-container">
        <div style="font-size: 13px; font-weight: 800; color: #c084fc; margin-bottom: 4px;">
          🎮 BÀN PHÍM ĐIỀU KHIỂN &amp; HIỆU CHUẨN
        </div>
        
        <!-- Forward button -->
        <div class="dpad-row">
          <button class="btn-dpad btn-dpad-fwd" id="btn-dpad-up"
                  onmousedown="startDrive('up')" onmouseup="stopDrive()" onmouseleave="stopDrive()"
                  ontouchstart="startDrive('up'); event.preventDefault();" ontouchend="stopDrive()">
            ⬆️ TIẾN THẲNG
            <span>[Phím Mũi Tên Lên / W] (w = 0.0)</span>
          </button>
        </div>

        <!-- Middle row: Left, Stop, Right -->
        <div class="dpad-row">
          <button class="btn-dpad" id="btn-dpad-left"
                  onmousedown="startDrive('left')" onmouseup="stopDrive()" onmouseleave="stopDrive()"
                  ontouchstart="startDrive('left'); event.preventDefault();" ontouchend="stopDrive()">
            ⬅️
            <span>Trái [A]</span>
          </button>
          <button class="btn-dpad btn-dpad-stop" id="btn-dpad-stop" onclick="stopDrive()">
            ⏹️
            <span>Phanh</span>
          </button>
          <button class="btn-dpad" id="btn-dpad-right"
                  onmousedown="startDrive('right')" onmouseup="stopDrive()" onmouseleave="stopDrive()"
                  ontouchstart="startDrive('right'); event.preventDefault();" ontouchend="stopDrive()">
            ➡️
            <span>Phải [D]</span>
          </button>
        </div>

        <!-- Down button -->
        <div class="dpad-row">
          <button class="btn-dpad" id="btn-dpad-down" style="width: 158px;"
                  onmousedown="startDrive('down')" onmouseup="stopDrive()" onmouseleave="stopDrive()"
                  ontouchstart="startDrive('down'); event.preventDefault();" ontouchend="stopDrive()">
            ⬇️ LÙI THẲNG
            <span>[Phím Mũi Tên Xuống / S]</span>
          </button>
        </div>

        <div style="font-size: 11px; color: #64748b; margin-top: 4px;">
          💡 Nhấn giữ phím để chạy, thả phím ra xe lập tức phanh dừng!
        </div>
      </div>

      <!-- Right: Settings & Calibration Guidance -->
      <div class="calib-settings">
        <div class="setting-row">
          <span style="font-size: 13px; font-weight: 700; color: #94a3b8;">TỐC ĐỘ TEST (v):</span>
          <div class="preset-group">
            <button class="btn-preset" id="p-015" onclick="setSpeedPreset(0.15)">0.15 m/s (Chậm)</button>
            <button class="btn-preset active" id="p-022" onclick="setSpeedPreset(0.22)">0.22 m/s (Chuẩn)</button>
            <button class="btn-preset" id="p-030" onclick="setSpeedPreset(0.30)">0.30 m/s (Nhanh)</button>
          </div>
        </div>

        <div class="slider-box">
          <input type="range" id="speed-slider" min="0.05" max="0.45" step="0.01" value="0.22" oninput="onSpeedSliderChange(this.value)">
          <div class="slider-val" id="speed-slider-val">0.22 m/s</div>
        </div>

        <div class="setting-row" style="margin-top: 4px;">
          <span style="font-size: 13px; font-weight: 700; color: #94a3b8;">TỐC ĐỘ BẺ LÁI (w):</span>
          <div class="slider-box" style="flex: 1; max-width: 260px;">
            <input type="range" id="turn-slider" min="0.2" max="1.2" step="0.05" value="0.50" oninput="onTurnSliderChange(this.value)">
            <div class="slider-val" id="turn-slider-val">0.50 rad/s</div>
          </div>
        </div>

        <div class="eval-box">
          <span style="color: #94a3b8;">Trạng thái hướng đi xe:</span>
          <span class="eval-badge eval-straight" id="eval-status-badge">ĐANG ĐỨNG YÊN</span>
        </div>

        <div class="calib-guide-box">
          <strong>📌 HƯỚNG DẪN TEST HIỆU CHUẨN 2 BÁNH XE:</strong><br>
          1. Đặt xe trên sàn phẳng có vạch hoặc mép gạch làm mốc tham chiếu.<br>
          2. Giữ phím <strong>[Mũi tên Lên (↑)]</strong> hoặc nút <strong>[⬆️ TIẾN THẲNG]</strong> để xe chạy thẳng 2 - 3 mét.<br>
          3. Quan sát quỹ đạo:
             - Nếu xe bị <strong>nhao sang TRÁI</strong>: Bánh Phải mạnh hơn bánh Trái.<br>
             - Nếu xe bị <strong>nhao sang PHẢI</strong>: Bánh Trái mạnh hơn bánh Phải.<br>
          4. Nhả phím để xe dừng lại và báo kết quả để điều chỉnh tỉ số feedforward <code>CRC_FF_GAIN_*</code>!
        </div>
      </div>
    </div>
  </div>

  <!-- VIEW MODE BUTTONS -->
  <div class="view-controls">
    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
      <button class="btn active" id="btn-dual" onclick="setView('dual')">🚀 Song Song Ghép 1 Stream (Khuyên Dùng)</button>
      <button class="btn" id="btn-lane" onclick="setView('lane')">🛣️ Chỉ Xem Làn (Lane Debug)</button>
      <button class="btn" id="btn-cam" onclick="setView('cam')">📷 Chỉ Xem Mắt Xe (Hardware JPEG)</button>
      <button class="btn" id="btn-split" onclick="setView('split')">👥 2 Khung Riêng Biệt (Split Cards)</button>
    </div>
    <div style="display: flex; gap: 6px; align-items: center; margin-left: auto; flex-wrap: wrap;">
      <span style="font-size: 12px; color: #94a3b8; font-weight: 600;">FPS STREAM:</span>
      <button class="btn btn-fps" id="fps-15" onclick="changeFps(15)">15</button>
      <button class="btn btn-fps" id="fps-20" onclick="changeFps(20)">20</button>
      <button class="btn btn-fps active" id="fps-25" onclick="changeFps(25)">25 (Khuyên Dùng)</button>
      <button class="btn btn-fps" id="fps-30" onclick="changeFps(30)">30 (Max)</button>
    </div>
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
    let currentViewMode = 'dual';
    let currentControlMode = 'auto'; // 'auto' or 'manual'
    let manualSpeed = 0.22;
    let manualTurnRate = 0.50;
    let activeDriveDirection = null;
    let driveIntervalTimer = null;

    function switchControlMode(mode) {{
      currentControlMode = mode;
      const btnAuto = document.getElementById('tab-auto-btn');
      const btnManual = document.getElementById('tab-manual-btn');
      const panelAuto = document.getElementById('panel-auto');
      const panelManual = document.getElementById('panel-manual');

      if (mode === 'manual') {{
        btnAuto.classList.remove('active');
        btnManual.classList.add('active');
        panelAuto.style.display = 'none';
        panelManual.style.display = 'block';
        sendCarCmd('manual');
      }} else {{
        btnManual.classList.remove('active');
        btnAuto.classList.add('active');
        panelManual.style.display = 'none';
        panelAuto.style.display = 'block';
        sendCarCmd('stop');
      }}
    }}

    function setSpeedPreset(val) {{
      manualSpeed = val;
      document.getElementById('speed-slider').value = val;
      document.getElementById('speed-slider-val').innerText = val.toFixed(2) + ' m/s';
      ['015', '022', '030'].forEach(id => {{
        const btn = document.getElementById('p-' + id);
        if (btn) btn.classList.remove('active');
      }});
      if (val === 0.15) document.getElementById('p-015').classList.add('active');
      else if (val === 0.22) document.getElementById('p-022').classList.add('active');
      else if (val === 0.30) document.getElementById('p-030').classList.add('active');
    }}

    function onSpeedSliderChange(val) {{
      manualSpeed = parseFloat(val);
      document.getElementById('speed-slider-val').innerText = manualSpeed.toFixed(2) + ' m/s';
      ['015', '022', '030'].forEach(id => {{
        const btn = document.getElementById('p-' + id);
        if (btn) btn.classList.remove('active');
      }});
    }}

    function onTurnSliderChange(val) {{
      manualTurnRate = parseFloat(val);
      document.getElementById('turn-slider-val').innerText = manualTurnRate.toFixed(2) + ' rad/s';
    }}

    async function sendManualVelocity(v, w) {{
      try {{
        await fetch(`/api/manual?v=${{v.toFixed(3)}}&w=${{w.toFixed(3)}}`);
      }} catch (e) {{}}
    }}

    function startDrive(direction) {{
      if (activeDriveDirection === direction) return;
      activeDriveDirection = direction;
      updateDpadVisuals(direction);

      let v = 0.0, w = 0.0;
      if (direction === 'up') {{
        v = manualSpeed;
        w = 0.0;
      }} else if (direction === 'down') {{
        v = -manualSpeed;
        w = 0.0;
      }} else if (direction === 'left') {{
        v = manualSpeed * 0.4;
        w = manualTurnRate;
      }} else if (direction === 'right') {{
        v = manualSpeed * 0.4;
        w = -manualTurnRate;
      }}

      sendManualVelocity(v, w);
      if (driveIntervalTimer) clearInterval(driveIntervalTimer);
      driveIntervalTimer = setInterval(() => {{
        sendManualVelocity(v, w);
      }}, 80);
    }}

    function stopDrive() {{
      if (driveIntervalTimer) {{
        clearInterval(driveIntervalTimer);
        driveIntervalTimer = null;
      }}
      activeDriveDirection = null;
      updateDpadVisuals(null);
      sendManualVelocity(0.0, 0.0);
      setTimeout(() => sendManualVelocity(0.0, 0.0), 50);
    }}

    function updateDpadVisuals(activeDir) {{
      ['up', 'down', 'left', 'right'].forEach(dir => {{
        const btn = document.getElementById('btn-dpad-' + dir);
        if (btn) {{
          if (dir === activeDir) btn.classList.add('active');
          else btn.classList.remove('active');
        }}
      }});
    }}

    // KEYBOARD EVENT LISTENERS
    const pressedKeys = new Set();
    window.addEventListener('keydown', (e) => {{
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;

      // Global hotkeys for AUTO mode
      if (e.code === 'Space') {{
        e.preventDefault();
        if (currentControlMode === 'manual') {{
          stopDrive();
        }} else {{
          const cur = document.getElementById('car-status-text').innerText;
          if (cur.includes('RUNNING')) sendCarCmd('stop');
          else sendCarCmd('start');
        }}
        return;
      }}
      if (e.key === 'r' || e.key === 'R') {{
        if (currentControlMode === 'auto') sendCarCmd('start');
        return;
      }}
      if (e.key === 's' || e.key === 'S') {{
        if (currentControlMode === 'auto') sendCarCmd('stop');
        else stopDrive();
        return;
      }}

      // Manual Navigation Keys (Arrows & WASD)
      if (['ArrowUp', 'KeyW', 'ArrowDown', 'KeyS', 'ArrowLeft', 'KeyA', 'ArrowRight', 'KeyD'].includes(e.code)) {{
        e.preventDefault();
        if (currentControlMode !== 'manual') {{
          switchControlMode('manual');
        }}

        pressedKeys.add(e.code);
        if (e.code === 'ArrowUp' || e.code === 'KeyW') startDrive('up');
        else if (e.code === 'ArrowDown' || e.code === 'KeyS') startDrive('down');
        else if (e.code === 'ArrowLeft' || e.code === 'KeyA') startDrive('left');
        else if (e.code === 'ArrowRight' || e.code === 'KeyD') startDrive('right');
      }}
    }});

    window.addEventListener('keyup', (e) => {{
      if (pressedKeys.has(e.code)) {{
        pressedKeys.delete(e.code);
        if (pressedKeys.size === 0) {{
          stopDrive();
        }} else {{
          // Revert to remaining held key
          if (pressedKeys.has('ArrowUp') || pressedKeys.has('KeyW')) startDrive('up');
          else if (pressedKeys.has('ArrowDown') || pressedKeys.has('KeyS')) startDrive('down');
          else if (pressedKeys.has('ArrowLeft') || pressedKeys.has('KeyA')) startDrive('left');
          else if (pressedKeys.has('ArrowRight') || pressedKeys.has('KeyD')) startDrive('right');
        }}
      }}
    }});

    // Stream Views Management
    function setView(mode) {{
      currentViewMode = mode;
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

    async function updateTelemetry() {{
      try {{
        const res = await fetch('/api/status');
        if (res.ok) {{
          const data = await res.json();
          document.getElementById('val-battery').innerText = data.battery_pct.toFixed(1) + '%';
          document.getElementById('val-volts').innerText = data.voltage.toFixed(2) + 'V';
          document.getElementById('val-v').innerText = data.linear_vel.toFixed(2);
          document.getElementById('val-w').innerText = data.angular_vel.toFixed(2);
          
          if (data.gyro_z !== undefined) {{
            document.getElementById('val-gyro').innerText = (data.gyro_z >= 0 ? '+' : '') + data.gyro_z.toFixed(3);
            updateEvaluationBadge(data.linear_vel, data.gyro_z);
          }}

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

          if (data.car_state) {{
            updateCarStateUI(data.car_state);
          }}
        }}
      }} catch (e) {{}}
    }}

    function updateEvaluationBadge(v, gz) {{
      const b = document.getElementById('eval-status-badge');
      if (!b) return;
      if (Math.abs(v) < 0.03) {{
        b.className = 'eval-badge';
        b.style.background = 'rgba(255, 255, 255, 0.1)';
        b.style.color = '#94a3b8';
        b.innerText = 'ĐANG ĐỨNG YÊN';
      }} else if (Math.abs(gz) < 0.08) {{
        b.className = 'eval-badge eval-straight';
        b.innerText = '✅ ĐI THẲNG TỐT (Bánh đều)';
      }} else if (gz > 0.08) {{
        b.className = 'eval-badge eval-left';
        b.innerText = `⚠️ LỆCH TRÁI (+${{gz.toFixed(2)}} rad/s)`;
      }} else {{
        b.className = 'eval-badge eval-right';
        b.innerText = `⚠️ LỆCH PHẢI (${{gz.toFixed(2)}} rad/s)`;
      }}
    }}

    async function sendCarCmd(cmd) {{
      try {{
        const res = await fetch('/api/car_cmd?cmd=' + cmd);
        if (res.ok) {{
          const data = await res.json();
          updateCarStateUI(data.car_state || (cmd === 'start' ? 'RUNNING' : (cmd === 'manual' ? 'MANUAL' : 'STANDBY')));
        }}
      }} catch (e) {{}}
    }}

    function updateCarStateUI(state) {{
      const badge = document.getElementById('car-status-badge');
      const text = document.getElementById('car-status-text');
      const btnStart = document.getElementById('btn-start');
      const btnStop = document.getElementById('btn-stop');

      if (state === 'RUNNING') {{
        badge.className = 'car-status-badge badge-running';
        text.innerText = 'ĐANG CHẠY (RUNNING)';
        btnStart.style.opacity = '0.5';
        btnStop.style.opacity = '1.0';
      }} else if (state === 'MANUAL') {{
        badge.className = 'car-status-badge badge-manual';
        text.innerText = 'CHẾ ĐỘ THỦ CÔNG (MANUAL)';
        btnStart.style.opacity = '1.0';
        btnStop.style.opacity = '1.0';
      }} else {{
        badge.className = 'car-status-badge badge-standby';
        text.innerText = 'CHỜ LỆNH (STANDBY)';
        btnStart.style.opacity = '1.0';
        btnStop.style.opacity = '0.7';
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

    setInterval(updateTelemetry, 700);
    updateTelemetry();
  </script>
</body>
</html>
"""
            self.wfile.write(html.encode('utf-8'))

        # 2. Telemetry Set FPS API
        elif self.path.startswith('/api/set_fps'):
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
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

        # 3. Telemetry JSON API
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            with state_lock:
                tel = dict(stream_data['telemetry'])
            self.wfile.write(json.dumps(tel).encode('utf-8'))

        # 4. Car Command API (START / STOP / MANUAL)
        elif self.path.startswith('/api/car_cmd'):
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                cmd = params.get('cmd', [''])[0].strip().lower()
                if cmd and g_dashboard_node is not None:
                    g_dashboard_node.send_car_cmd(cmd)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    cur_st = stream_data['telemetry'].get('car_state', 'STANDBY')
                    self.wfile.write(json.dumps({'status': 'ok', 'cmd': cmd, 'car_state': cur_st}).encode('utf-8'))
                else:
                    self.send_response(400)
                    self.end_headers()
            except Exception:
                self.send_response(500)
                self.end_headers()

        # 5. Manual Drive Velocity API (/api/manual?v=...&w=...)
        elif self.path.startswith('/api/manual'):
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                v = float(params.get('v', [0.0])[0])
                w = float(params.get('w', [0.0])[0])
                if g_dashboard_node is not None:
                    if stream_data['telemetry'].get('car_state') != 'MANUAL':
                        g_dashboard_node.send_car_cmd('manual')
                    g_dashboard_node.send_manual_drive(v, w)
                    with state_lock:
                        stream_data['telemetry']['linear_vel'] = v
                        stream_data['telemetry']['angular_vel'] = w
                        stream_data['telemetry']['car_state'] = 'MANUAL'
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok', 'v': v, 'w': w}).encode('utf-8'))
            except Exception:
                self.send_response(500)
                self.end_headers()

        # 6. Stream MJPEG (/stream/lane, /stream/camera, /stream/dual)
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
        
        # Hardware-compressed image
        self.sub_cam_comp = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.on_cam_compressed, qos_profile_sensor_data)
        
        # Raw fallback
        self.sub_cam_raw = self.create_subscription(
            Image, '/camera/image_raw', self.on_cam_raw, qos_profile_sensor_data)

        self.sub_voltage = self.create_subscription(
            Float32, '/voltage', self.on_voltage, 10)
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.sub_imu = self.create_subscription(
            Imu, '/imu', self.on_imu, qos_profile_sensor_data)

        # Publisher & Subscriber for car commands and manual drive
        self.pub_car_cmd = self.create_publisher(String, '/car_cmd', 10)
        self.sub_car_status = self.create_subscription(
            String, '/car_status', self.on_car_status, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.has_compressed_cam = False
        self.get_logger().info(f'Dashboard ultra-smooth streamer ready on port {PORT} at {TARGET_FPS} FPS')

    def on_car_status(self, msg):
        st = msg.data.strip().upper()
        with state_lock:
            stream_data['telemetry']['car_state'] = st

    def on_imu(self, msg):
        with state_lock:
            stream_data['telemetry']['gyro_z'] = float(msg.angular_velocity.z)

    def send_car_cmd(self, cmd):
        msg = String()
        msg.data = cmd
        self.pub_car_cmd.publish(msg)
        with state_lock:
            if cmd in ('start', 'run', 'go'):
                stream_data['telemetry']['car_state'] = 'RUNNING'
            elif cmd in ('stop', 'pause', 'halt'):
                stream_data['telemetry']['car_state'] = 'STANDBY'
            elif cmd in ('manual', 'teleop', 'calib'):
                stream_data['telemetry']['car_state'] = 'MANUAL'
        self.get_logger().info(f'[WEB DASHBOARD] Gui lenh dieu khien xe: {cmd}')

    def send_manual_drive(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub_cmd_vel.publish(msg)

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
    global g_dashboard_node
    rclpy.init(args=args)
    node = DashboardSubscriber()
    g_dashboard_node = node

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
