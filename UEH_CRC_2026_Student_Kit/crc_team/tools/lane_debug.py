#!/usr/bin/env python3
"""Run the lane detector on saved frames and write annotated images: the fastest way to
tune the lane settings, without driving. Works on the laptop and on the car.

  python3 lane_debug.py '/data/datasets/frames_xxx/*.jpg' --config ../config/race_car.yaml
  # on the car, inside the team container:
  bash run_car.sh run "python3 /ws/src/crc_team/tools/lane_debug.py '/data/datasets/frames_xxx/*.jpg' --config /ws/src/crc_team/config/race_car.yaml"

Writes <folder>/lane_debug/<name>.jpg: camera image (left) and bird's-eye mask with the
left (green), right (blue) and centre (red) fits. Prints one line per image.
"""
import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from crc_team.birdseye import BirdsEye, intrinsics_from_hfov  # noqa: E402
from crc_team.lane import LaneDetector  # noqa: E402
from crc_team.config import DEFAULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pattern', help="image files, e.g. 'frames/*.jpg' (quote it)")
    ap.add_argument('--config', help='race YAML (default: the built-in car defaults)')
    ap.add_argument('--hfov', type=float, default=62.0,
                    help='camera horizontal FOV (deg) when the config has no camera_matrix')
    args = ap.parse_args()

    c = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            c.update(yaml.safe_load(f) or {})
    files = sorted(glob.glob(args.pattern))
    if not files:
        sys.exit('no images match ' + args.pattern)
    first = cv2.imread(files[0])
    h, w = first.shape[:2]
    K = np.reshape(c['camera_matrix'], (3, 3)) if len(c['camera_matrix']) == 9 else \
        intrinsics_from_hfov(math.radians(args.hfov), w, h)
    be = BirdsEye(K, c['dist_coeffs'], c['fisheye'], c['be_x_min'], c['be_x_max'],
                  c['be_half_width'], c['be_ppm'], (w, h))
    if len(c['birdseye_points']) == 8 and len(c['birdseye_ground']) == 8:
        be.from_points(np.reshape(c['birdseye_points'], (4, 2)), np.reshape(c['birdseye_ground'], (4, 2)))
    else:
        be.from_geometry(c['cam_height'], c['cam_x'], c['cam_y'], math.radians(c['cam_pitch_deg']))
    det = LaneDetector(be, c['lane_width'], c['line_width'], c['min_contrast'])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(files[0])), 'lane_debug')
    os.makedirs(out_dir, exist_ok=True)
    found = 0
    for f in files:
        img = cv2.imread(f)
        t = time.perf_counter()
        res = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        ms = (time.perf_counter() - t) * 1000
        found += res.ok
        dbg = det.draw(res)
        scale = img.shape[0] / dbg.shape[0]
        dbg = cv2.resize(dbg, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), np.hstack([img, dbg]))
        lines = ('L' if res.left is not None else '.') + ('R' if res.right is not None else '.')
        off = f'{res.offset(c["lookahead"]):+.3f} m' if res.ok else '   -   '
        width = f'width {res.width:.3f} m' if res.width else ''
        print(f'{os.path.basename(f)}  lines={lines}  centre at {c["lookahead"]} m: {off}  '
              f'{"zebra " if res.zebra else ""}{width}  {ms:.1f} ms')
    print(f'\nlane found in {found}/{len(files)} images; annotated images in {out_dir}')


if __name__ == '__main__':
    main()
