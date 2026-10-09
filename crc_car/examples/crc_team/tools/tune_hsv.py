#!/usr/bin/env python3
"""Find HSV colour ranges (traffic light lamps, coloured markers) with sliders.
Runs on a laptop with a screen:  python3 tune_hsv.py frame.jpg
Move the sliders until only the object stays white in the mask, then press q: the range
is printed in the format of light.py (OpenCV hue is 0..179; red needs two ranges, one
near 0 and one near 179).
"""
import sys

import cv2
import numpy as np


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    img = cv2.imread(sys.argv[1])
    if img is None:
        sys.exit('cannot read ' + sys.argv[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    win = 'tune_hsv (q to quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for name, value, top in (('H low', 0, 179), ('H high', 179, 179), ('S low', 100, 255),
                             ('S high', 255, 255), ('V low', 120, 255), ('V high', 255, 255)):
        cv2.createTrackbar(name, win, value, top, lambda v: None)
    while True:
        lo = tuple(cv2.getTrackbarPos(n, win) for n in ('H low', 'S low', 'V low'))
        hi = tuple(cv2.getTrackbarPos(n, win) for n in ('H high', 'S high', 'V high'))
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        cv2.imshow(win, np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print(f'range: ({lo}, {hi})')


if __name__ == '__main__':
    main()
