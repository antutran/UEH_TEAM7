"""Lane detection on the bird's-eye image. Pure OpenCV/NumPy, no ROS, so it can be tested
offline on saved frames (tools/lane_debug.py).

Pipeline (about 2-4 ms per frame on the car):
  1. gray image -> bird's-eye view (BirdsEye.warp)
  2. white-line mask: morphological top-hat (bright thin things on a darker road), which
     adapts to the light level, so it also works in the dark tunnel, then a threshold
  3. sliding windows from the bottom up, starting where the left and right lines are
     expected (+-lane_width/2), or where they were in the previous frame
  4. a 2nd-order fit y(x) in metres for each line found
  5. lane centre: middle of both lines, or half a lane width away from the one line
     seen, measured square to the line (so it stays right in tight bends)

The car drives in the right-hand lane: its "left" line is the centre line of the road
(dashed or solid), its "right" line the outer edge.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class LaneResult:
    ok: bool = False                  # a lane centre could be estimated
    zebra: bool = False               # a zebra crossing is in view
    left: object = None               # np.poly1d y(x) of the left line in metres, or None
    right: object = None              # same for the right line
    centre: object = None             # np.poly1d y(x) of the lane centre
    width: float = 0.0                # measured lane width at x_ref (0 when only one line)
    half: float = 0.2                 # half the lane width used for one-line estimates
    mask: object = None               # bird's-eye binary mask (for debugging)
    points: dict = field(default_factory=dict)

    def target(self, dist):
        """Point (x, y) on the lane centre about `dist` metres ahead."""
        if self.left is not None and self.right is not None:
            return dist, float(self.centre(dist))
        line, sign = (self.right, 1.0) if self.right is not None else (self.left, -1.0)
        slope_of = line.deriv()
        x0 = dist
        for _ in range(3):              # find the line point whose normal ends at x = dist
            slope = float(slope_of(x0))
            x0 = dist + sign * self.half * slope / math.hypot(1.0, slope)
        slope = float(slope_of(x0))
        norm = math.hypot(1.0, slope)
        return x0 - sign * self.half * slope / norm, float(line(x0)) + sign * self.half / norm

    def offset(self, x):
        """Lateral position of the lane centre about x ahead (m, + = left)."""
        return self.target(x)[1] if self.ok else 0.0


class LaneDetector:
    def __init__(self, birdseye, lane_width=0.40, line_width=0.025, min_contrast=25,
                 windows=8, window_half=0.06, min_pixels=8, prefer='both'):
        self.be = birdseye
        self.lane_width = lane_width
        self.min_contrast = min_contrast
        self.windows = windows
        self.window_half_px = max(3, int(window_half * birdseye.ppm))
        self.min_pixels = min_pixels
        # Widest a lane line may look in one image row (pixels): 2.5 line widths covers
        # lines seen at an angle in bends; zebra stripes and arrows are much wider
        self.max_row_px = 2.5 * line_width * birdseye.ppm + 2
        self.prefer = prefer            # 'both', 'left' or 'right' (e.g. at a junction)
        k = max(5, int(4 * line_width * birdseye.ppm) | 1)   # ~4 line widths, odd
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        # Erode the visibility mask so the edge of the camera view is not seen as a line
        self.valid = cv2.erode(birdseye.valid, np.ones((k, k), np.uint8))
        self.last = {'left': None, 'right': None}

    def line_mask(self, gray):
        top = self.be.warp(gray)
        hat = cv2.morphologyEx(top, cv2.MORPH_TOPHAT, self.kernel)
        hat[self.valid == 0] = 0
        # Threshold: half of the brightest line contrast in view, never below min_contrast
        peak = float(np.percentile(hat[self.valid > 0], 99.5)) if self.valid.any() else 0.0
        thr = max(self.min_contrast, 0.5 * peak)
        mask = (hat > thr).astype(np.uint8)
        # Zebra crossing: rows with 4 or more separate white segments are not lane lines
        starts = np.count_nonzero(np.diff(mask, axis=1) == 1, axis=1)
        self.zebra_rows = starts >= 4
        mask[self.zebra_rows] = 0
        return mask

    def detect(self, gray):
        mask = self.line_mask(gray)
        h, w = mask.shape
        ppm = self.be.ppm
        half = self.lane_width / 2.0
        res = LaneResult(mask=mask, zebra=int(self.zebra_rows.sum()) > 3, half=self.lane_width / 2)
        fits = {}
        ys, xs = np.nonzero(mask)
        for side, sign in (('left', 1.0), ('right', -1.0)):
            prev = self.last[side]
            # Start column: previous fit near the bottom, else the expected line position
            if prev is not None:
                y0 = float(prev(self.be.x_min + 0.05))
            else:
                y0 = sign * half
            u_start, _ = self.be.to_pixel(0.0, y0)
            # Refine the start with the column histogram of the lower third
            lower = mask[2 * h // 3:, :]
            lo = int(max(0, u_start - 0.35 * self.lane_width * ppm))
            hi = int(min(w, u_start + 0.35 * self.lane_width * ppm))
            if hi - lo > 2:
                hist = lower[:, lo:hi].sum(axis=0)
                if hist.max() >= 3:
                    u_start = lo + float(np.argmax(hist))
            pts_u, pts_v = [], []
            u = u_start
            band = h / self.windows
            found = 0
            misses = 0
            drift = 0.0                 # how far the line moved per window (bends)
            for i in range(self.windows):
                u += drift              # look where a curving line will be, not where it was
                v_hi = h - i * band
                v_lo = v_hi - band
                sel = ((ys >= v_lo) & (ys < v_hi) &
                       (xs >= u - self.window_half_px) & (xs <= u + self.window_half_px))
                n = int(sel.sum())
                # A line across the road (stop line, junction) fills the whole window
                # width: skip it instead of letting it pull the fit sideways
                across = n and np.ptp(xs[sel]) > 1.6 * self.window_half_px
                # A blob wider than a line (zebra stripe, arrow, ramp chevron): skip too
                too_wide = n > self.max_row_px * band
                if n >= self.min_pixels and not across and not too_wide:
                    centre = float(xs[sel].mean())
                    if found:
                        drift = float(np.clip(centre - (u - drift), -self.window_half_px,
                                              self.window_half_px))
                    u = centre
                    pts_u.append(xs[sel])
                    pts_v.append(ys[sel])
                    found += 1
                    misses = 0
                else:
                    misses += 1
                    if misses > 3:      # a dashed line skips some windows; a lost line many
                        break
            if found >= 2:
                uu = np.concatenate(pts_u)
                vv = np.concatenate(pts_v)
                gx, gy = self.be.to_ground(uu + 0.5, vv + 0.5)
                deg = 2 if found >= 4 and np.ptp(gx) > 0.25 else 1
                fits[side] = np.poly1d(np.polyfit(gx, gy, deg))
                res.points[side] = (uu, vv)

        x_ref = self.be.x_min + 0.10
        left, right = fits.get('left'), fits.get('right')
        # Sanity check: two lines must be roughly one lane apart, else trust the one
        # that moved less since the last frame
        if left is not None and right is not None:
            # Check near AND far: two lines that cross (an X) are not a lane
            width = float(left(x_ref) - right(x_ref))
            x_far = self.be.x_max - 0.10
            width_far = float(left(x_far) - right(x_far))
            ok_width = lambda wd: 0.6 * self.lane_width < wd < 1.5 * self.lane_width  # noqa: E731
            if not (ok_width(width) and ok_width(width_far)):
                left, right = self._keep_steadier(left, right, x_ref)
            else:
                res.width = width
        if self.prefer == 'left' and left is not None:
            right = None
        elif self.prefer == 'right' and right is not None:
            left = None

        res.left, res.right = left, right
        if left is not None and right is not None:
            res.centre = (left + right) / 2.0
        elif left is not None:
            res.centre = left - half
        elif right is not None:
            res.centre = right + half
        res.ok = res.centre is not None
        self.last = {'left': left, 'right': right}
        return res

    def _keep_steadier(self, left, right, x_ref):
        d_left = abs(left(x_ref) - self.last['left'](x_ref)) if self.last['left'] is not None else 1.0
        d_right = abs(right(x_ref) - self.last['right'](x_ref)) if self.last['right'] is not None else 1.0
        return (left, None) if d_left < d_right else (None, right)

    def draw(self, res):
        """Colour debug image of the bird's-eye mask with the fits (left green, right blue,
        centre red)."""
        img = cv2.cvtColor(res.mask * 255, cv2.COLOR_GRAY2BGR)
        xs = np.linspace(self.be.x_min, self.be.x_max, 20)
        centre = np.array([res.target(x) for x in xs]) if res.ok else None
        for poly, colour in ((res.left, (0, 200, 0)), (res.right, (255, 120, 0)),
                             (centre, (0, 0, 255))):
            if poly is None:
                continue
            if isinstance(poly, np.ndarray):
                u, v = self.be.to_pixel(poly[:, 0], poly[:, 1])
            else:
                u, v = self.be.to_pixel(xs, poly(xs))
            pts = np.stack([u, v], axis=1).astype(np.int32)
            cv2.polylines(img, [pts], False, colour, 2)
        cu, _ = self.be.to_pixel(0.0, 0.0)
        cv2.line(img, (int(cu), img.shape[0] - 8), (int(cu), img.shape[0] - 1), (0, 255, 255), 2)
        return img
