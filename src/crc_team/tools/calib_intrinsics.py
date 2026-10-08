#!/usr/bin/env python3
"""Lens calibration (camera matrix and distortion) from chessboard photos. Optional for a
normal lens; recommended for wide-angle lenses, where straight lines look bent.

  1. Print a chessboard (default 9 x 6 inner corners, 25 mm squares) and glue it flat.
  2. On the car:  bash run_car.sh frames 0.5 40
     and slowly move the board in front of the camera: near, far, every corner of the
     image, tilted. 20-40 sharp frames are enough.
  3. Copy the folder to your laptop and run:
     python3 calib_intrinsics.py /path/to/frames_xxx [--fisheye] [--cols 9 --rows 6 --square 0.025]
  4. Paste the printed lines into config/race_car.yaml, then redo calib_birdseye.py with
     --config config/race_car.yaml.
An undistorted example (undistorted_preview.jpg) is written into the folder: straight
edges in the room must look straight.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    ap.add_argument('--cols', type=int, default=9, help='inner corners per row')
    ap.add_argument('--rows', type=int, default=6, help='inner corners per column')
    ap.add_argument('--square', type=float, default=0.025, help='square size (m)')
    ap.add_argument('--fisheye', action='store_true', help='fisheye model (lenses over ~120 deg)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, '*.jpg')) + glob.glob(os.path.join(args.folder, '*.png')))
    files = [f for f in files if 'preview' not in f]
    if not files:
        sys.exit('no .jpg/.png images in ' + args.folder)
    board = np.zeros((args.rows * args.cols, 1, 3), np.float32)
    board[:, 0, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
    obj, img_pts, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows),
                                                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)
            obj.append(board)
            img_pts.append(corners)
    print(f'chessboard found in {len(obj)} of {len(files)} images')
    if len(obj) < 8:
        sys.exit('need at least 8 good images: move the board around more, keep it sharp')

    if args.fisheye:
        K, D = np.zeros((3, 3)), np.zeros((4, 1))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img_pts, size, K, D, flags=flags, criteria=crit)
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(obj, img_pts, size, None, None)
    print(f'reprojection error: {rms:.2f} px (good: below 0.5; above 1: retake the photos)')

    img = cv2.imread(files[0])
    if args.fisheye:
        und = cv2.fisheye.undistortImage(img, K, D, Knew=K)
    else:
        und = cv2.undistort(img, K, D)
    out = os.path.join(args.folder, 'undistorted_preview.jpg')
    cv2.imwrite(out, np.hstack([img, und]))

    print('\n# Paste into config/race_car.yaml')
    print('camera_matrix: [' + ', '.join(f'{v:.2f}' for v in K.ravel()) + ']')
    print('dist_coeffs: [' + ', '.join(f'{v:.5f}' for v in np.ravel(D)) + ']')
    print(f'fisheye: {"true" if args.fisheye else "false"}')
    print(f'\nPreview (left original, right undistorted): {out}')


if __name__ == '__main__':
    main()
