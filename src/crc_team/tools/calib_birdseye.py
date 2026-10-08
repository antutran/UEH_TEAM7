#!/usr/bin/env python3
"""Bird's-eye calibration from one photo: four clicks, no measuring of angles.

On the track:
  1. Put the car on a STRAIGHT lane, centred and pointing along the lane.
  2. Lay two strips of tape across the lane, NEAR and FAR from the car, and measure the
     distance from the WHEEL AXLE to each strip (e.g. 0.35 m and 0.80 m).
  3. Save a frame:  bash run_car.sh frames 1 1   (one frame into /data/datasets/frames_...)
On your laptop (needs python3 with opencv-python and numpy):
  4. python3 calib_birdseye.py frame.jpg --near 0.35 --far 0.80
     Click, in this order, where the tape crosses the CENTRE of each lane line:
     near-left, near-right, far-left, far-right.  Press any key when done.
  5. Paste the two printed lines into config/race_car.yaml.
No screen? Read the four pixel positions in any image viewer and pass them:
     python3 calib_birdseye.py frame.jpg --near 0.35 --far 0.80 --points u1 v1 u2 v2 u3 v3 u4 v4
A preview (birdseye_preview.jpg) is written next to the image: the lane lines must come
out straight, vertical and parallel.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from crc_team.birdseye import BirdsEye, intrinsics_from_hfov  # noqa: E402

NAMES = ['near-left', 'near-right', 'far-left', 'far-right']


def click_points(img):
    pts = []
    view = img.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(view, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(view, NAMES[len(pts) - 1], (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

    cv2.namedWindow('click 4 points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('click 4 points', on_mouse)
    while True:
        cv2.imshow('click 4 points', view)
        title = f'next: {NAMES[len(pts)]}' if len(pts) < 4 else 'done: press any key'
        cv2.setWindowTitle('click 4 points', title)
        if cv2.waitKey(30) >= 0 and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image')
    ap.add_argument('--near', type=float, required=True, help='wheel axle to the near tape (m)')
    ap.add_argument('--far', type=float, required=True, help='wheel axle to the far tape (m)')
    ap.add_argument('--lane-width', type=float, default=0.425,
                    help='distance between the CENTRES of the two lines (m), default 0.425')
    ap.add_argument('--points', type=float, nargs=8, help='u v of the 4 points, if no screen')
    ap.add_argument('--hfov', type=float, default=62.0, help='camera horizontal FOV (deg)')
    ap.add_argument('--config', help='race YAML with camera_matrix/dist_coeffs from '
                    'calib_intrinsics.py (use the same lens model as the car)')
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f'cannot read {args.image}')
    pts = np.reshape(args.points, (4, 2)) if args.points else np.array(click_points(img), float)
    half = args.lane_width / 2
    ground = np.array([[args.near, half], [args.near, -half], [args.far, half], [args.far, -half]])

    h, w = img.shape[:2]
    K, dist, fisheye = intrinsics_from_hfov(np.radians(args.hfov), w, h), [], False
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        if len(cfg.get('camera_matrix', [])) == 9:
            K = np.reshape(cfg['camera_matrix'], (3, 3))
            dist, fisheye = cfg.get('dist_coeffs', []), cfg.get('fisheye', False)
    be = BirdsEye(K, dist, fisheye, x_min=max(0.05, args.near - 0.15), x_max=args.far + 0.4,
                  half_width=0.6, image_size=(w, h)).from_points(pts, ground)
    top = be.warp(img)
    for y in (half, -half):           # where the lines should be
        u, _ = be.to_pixel(0, y)
        cv2.line(top, (int(u), 0), (int(u), top.shape[0]), (0, 255, 255), 1)
    out = os.path.join(os.path.dirname(os.path.abspath(args.image)), 'birdseye_preview.jpg')
    cv2.imwrite(out, cv2.resize(top, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))

    print('\n# Paste into config/race_car.yaml')
    print('birdseye_points: [' + ', '.join(f'{v:.1f}' for v in pts.ravel()) + ']')
    print('birdseye_ground: [' + ', '.join(f'{v:.3f}' for v in ground.ravel()) + ']')
    print(f'\nPreview: {out}  (the white lines must sit on the yellow guides)')
    print('Note: the points are only valid for this camera mounting. Re-run after the camera '
          'moves, and if you also calibrate the lens (calib_intrinsics.py), click on an '
          'image from the same camera; the config must then also hold camera_matrix and dist_coeffs.')


if __name__ == '__main__':
    main()
