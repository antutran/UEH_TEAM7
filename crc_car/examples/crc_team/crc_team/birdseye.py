"""Bird's-eye view: turns the camera image into a top-down map of the road in metres.

Ground frame (as in ROS base_footprint): x forward from the wheel axle, y to the left.
The bird's-eye image covers x in [x_min, x_max] and y in [-half_width, +half_width]:
  column u = (half_width - y) * ppm      (left of the car is on the left of the image)
  row    v = (x_max - x) * ppm           (far is at the top, near at the bottom)

The mapping is built once (cv2.remap tables), so each frame costs a single remap of a
small image. Two ways to describe the camera:
  * geometry: camera height, forward offset and downward pitch, plus the intrinsics
    (from /camera/camera_info or a calibration). Good default, no clicking.
  * points:   four image points whose ground positions are known (tools/calib_birdseye.py).
    More accurate on a real car, because nobody measures the pitch to 0.5 degrees.
Lens distortion (tools/calib_intrinsics.py) is handled in both cases.
"""
import cv2
import numpy as np


class BirdsEye:
    def __init__(self, K, dist=None, fisheye=False, x_min=0.25, x_max=0.95,
                 half_width=0.60, ppm=150.0, image_size=(640, 480)):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.zeros(5) if dist is None or len(dist) == 0 else np.asarray(dist, np.float64)
        self.fisheye = fisheye
        self.x_min, self.x_max, self.half_width, self.ppm = x_min, x_max, half_width, ppm
        self.image_size = image_size
        self.width = int(round(2 * half_width * ppm))
        self.height = int(round((x_max - x_min) * ppm))
        # Ground coordinates of every output pixel (pixel centres)
        u, v = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        self.grid_x = x_max - v / ppm
        self.grid_y = half_width - u / ppm
        self.map_x = self.map_y = self.valid = None

    # --- pixel <-> metre helpers for the bird's-eye image ---------------------------
    def to_pixel(self, x, y):
        return (self.half_width - y) * self.ppm, (self.x_max - x) * self.ppm

    def to_ground(self, u, v):
        return self.x_max - v / self.ppm, self.half_width - u / self.ppm

    # --- building the remap tables ------------------------------------------------------
    def from_geometry(self, cam_height, cam_x=0.0, cam_y=0.0, pitch=0.0, roll=0.0, yaw=0.0):
        """Camera at (cam_x, cam_y, cam_height) in the ground frame, tilted down by pitch
        (radians, positive = looking down). Small roll/yaw corrections are optional."""
        dx = self.grid_x - cam_x
        dy = self.grid_y - cam_y
        dz = -cam_height * np.ones_like(dx)
        # Rotate the ground vectors into the camera body frame (x forward, y left, z up)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        cyw, syw = np.cos(yaw), np.sin(yaw)
        # inverse yaw, then inverse pitch, then inverse roll
        x1, y1 = cyw * dx + syw * dy, -syw * dx + cyw * dy
        x2, z2 = cp * x1 - sp * dz, sp * x1 + cp * dz
        y3, z3 = cr * y1 + sr * z2, -sr * y1 + cr * z2
        # Optical frame: right = -y, down = -z, forward = x
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = -y3 / x2
            yn = -z3 / x2
        behind = x2 <= 1e-3
        xn[behind] = yn[behind] = 1e6
        self._build(xn, yn)
        return self

    def from_points(self, image_points, ground_points):
        """image_points: four (u, v) pixels in the camera image.
        ground_points: the same four points as (x, y) metres in the ground frame."""
        img = np.asarray(image_points, np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            norm = cv2.fisheye.undistortPoints(img, self.K, self.dist[:4])
        else:
            norm = cv2.undistortPoints(img, self.K, self.dist)
        H = cv2.getPerspectiveTransform(np.asarray(ground_points, np.float32).reshape(4, 2),
                                        norm.reshape(4, 2).astype(np.float32))
        # OpenCV scales H so that the ground origin gets w = +1, but the origin (the wheel
        # axle) is usually behind the camera: flip the sign so the clicked points, which
        # are in front of the camera, have w > 0
        g0 = np.asarray(ground_points, np.float64).reshape(4, 2)[0]
        if (H @ np.array([g0[0], g0[1], 1.0]))[2] < 0:
            H = -H
        pts =np.stack([self.grid_x, self.grid_y, np.ones_like(self.grid_x)], axis=-1) @ H.T
        with np.errstate(divide='ignore', invalid='ignore'):
            xn = pts[..., 0] / pts[..., 2]
            yn = pts[..., 1] / pts[..., 2]
        behind = pts[..., 2] <= 0
        xn[behind] = yn[behind] = 1e6
        self._build(xn, yn)
        return self

    def _build(self, xn, yn):
        """Normalised undistorted camera coordinates -> distorted pixel coordinates."""
        obj = np.stack([xn.ravel(), yn.ravel(), np.ones(xn.size)], axis=-1).reshape(-1, 1, 3)
        far = np.abs(obj[:, 0, 0]) > 10      # outside any lens: skip the distortion model
        obj[far, 0, :2] = 0.0
        zero = np.zeros(3)
        if self.fisheye:
            px, _ = cv2.fisheye.projectPoints(obj, zero, zero, self.K, self.dist[:4])
        else:
            px, _ = cv2.projectPoints(obj, zero, zero, self.K, self.dist)
        px = px.reshape(-1, 2)
        px[far] = -1.0
        w, h = self.image_size
        self.map_x = px[:, 0].reshape(xn.shape).astype(np.float32)
        self.map_y = px[:, 1].reshape(xn.shape).astype(np.float32)
        self.valid = ((self.map_x >= 0) & (self.map_x <= w - 1) &
                      (self.map_y >= 0) & (self.map_y <= h - 1)).astype(np.uint8)

    def warp(self, image):
        """Camera image (gray or colour) -> bird's-eye image; pixels the camera cannot see are 0."""
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def intrinsics_from_hfov(hfov_rad, width=640, height=480):
    """Ideal pinhole camera matrix for a given horizontal field of view."""
    f = (width / 2.0) / np.tan(hfov_rad / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]])
