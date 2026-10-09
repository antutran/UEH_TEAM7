#!/usr/bin/env python3
"""Save camera frames as JPEG files, e.g. to build a sign dataset on the real track.
Runs inside the team container (bash run_car.sh frames [period_s] [count]).
The camera node already publishes JPEG on /camera/image_raw/compressed, so each frame
is written as received: no decoding or re-encoding on the CPU.
Output: /data/datasets/frames_<date>_<time>/ (TF card), or ~/crc_ws/src/_frames_... without one."""
import os
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

period = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
count = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # 0 = until Ctrl+C
base = '/data/datasets' if os.access('/data', os.W_OK) else '/ws/src'
prefix = 'frames_' if base == '/data/datasets' else '_frames_'
out_dir = os.path.join(base, prefix + time.strftime('%Y%m%d_%H%M%S'))
os.makedirs(out_dir, exist_ok=True)
host_dir = out_dir.replace('/ws/src', '~/crc_ws/src', 1)

saved = 0
last = 0.0


def on_frame(msg):
    global saved, last
    now = time.monotonic()
    if now - last < period:
        return
    last = now
    path = os.path.join(out_dir, f'{saved:05d}.jpg')
    with open(path, 'wb') as f:
        f.write(msg.data)
    os.chown(path, 1000, 1000)   # the car user, so the files can be copied and deleted
    saved += 1
    if saved % 10 == 0:
        print(f'{saved} frames', flush=True)


rclpy.init()
node = rclpy.create_node('save_frames')
node.create_subscription(CompressedImage, '/camera/image_raw/compressed', on_frame,
                         qos_profile_sensor_data)
os.chown(out_dir, 1000, 1000)
print(f'Saving one frame every {period:g} s to {host_dir} '
      f'({"Ctrl+C to stop" if count == 0 else f"{count} frames"})', flush=True)
try:
    while rclpy.ok() and (count == 0 or saved < count):
        rclpy.spin_once(node, timeout_sec=0.5)
except KeyboardInterrupt:
    pass
print(f'Saved {saved} frames in {host_dir}')
node.destroy_node()
rclpy.try_shutdown()
