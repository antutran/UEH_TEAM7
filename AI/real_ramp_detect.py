
from pathlib import Path
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Tested on images around 1200x800 pixels
WIDTHS = (35, 42, 50, 58, 66, 75, 85, 98, 112, 130, 148)
RATIOS = (0.76, 0.87, 1.0, 1.13, 1.28)

MATCH_THRESHOLD = 0.63
HUMP_THRESHOLD = 30


def triangle_template(w, h):
    """Create an ideal red-warning-triangle shape."""
    template = np.zeros((h, w), dtype=np.uint8)

    points = np.array([
        [round(w * 0.50), round(h * 0.07)],
        [round(w * 0.07), round(h * 0.90)],
        [round(w * 0.93), round(h * 0.90)]
    ], dtype=np.int32)

    thickness = max(2, round(w * 0.065))

    cv2.polylines(
        template, [points], True,
        255, thickness, cv2.LINE_AA
    )

    return cv2.GaussianBlur(
        template.astype(np.float32),
        (0, 0), 1.0
    )


def hump_contrast(gray, x, y, w, h):
    """Check whether the bottom-center symbol is dark."""

    def patch_mean(x1, x2, y1, y2):
        xa = x + round(x1 * w)
        xb = x + round(x2 * w)
        ya = y + round(y1 * h)
        yb = y + round(y2 * h)

        return float(np.mean(gray[ya:yb, xa:xb]))

    # Bright interior of the warning triangle
    upper = patch_mean(0.38, 0.62, 0.38, 0.55)

    # Location of the black speed-bump symbol
    lower = patch_mean(0.32, 0.68, 0.65, 0.80)

    return upper - lower


def box_iou(a, b):
    """Intersection-over-union for duplicate removal."""
    xa, ya, wa, ha = a[:4]
    xb, yb, wb, hb = b[:4]

    intersection = (
        max(0, min(xa + wa, xb + wb) - max(xa, xb))
        * max(0, min(ya + ha, yb + hb) - max(ya, yb))
    )

    union = wa * ha + wb * hb - intersection
    return intersection / max(union, 1)


def detect_ramp(img):
    """Return detected ramp warning signs."""

    b, g, r = cv2.split(img)

    # Red-vs-green contrast works better on faded red signs
    redness = np.maximum(
        r.astype(np.float32) - g.astype(np.float32),
        0
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    candidates = []

    # Multi-scale triangle matching
    for w in WIDTHS:
        for ratio in RATIOS:
            h = round(w / ratio)

            if h >= img.shape[0] or w >= img.shape[1]:
                continue

            template = triangle_template(w, h)

            score_map = cv2.matchTemplate(
                redness,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            # Keep local maxima
            local_max = cv2.dilate(
                score_map,
                np.ones((9, 9), dtype=np.uint8)
            )

            ys, xs = np.where(
                (score_map >= MATCH_THRESHOLD)
                & (score_map >= local_max - 1e-6)
            )

            # Limit excessive candidates
            if len(xs) > 40:
                indices = np.argsort(
                    score_map[ys, xs]
                )[-40:]

                ys = ys[indices]
                xs = xs[indices]

            for y, x in zip(ys, xs):
                x, y = int(x), int(y)

                contrast = hump_contrast(
                    gray, x, y, w, h
                )

                # Reject triangles without a dark bump
                if contrast < HUMP_THRESHOLD:
                    continue

                score = float(score_map[y, x])

                candidates.append(
                    (x, y, w, h, score)
                )

    # Non-Maximum Suppression
    candidates.sort(
        key=lambda item: item[4],
        reverse=True
    )

    detections = []

    for candidate in candidates:
        if all(
            box_iou(candidate, previous) < 0.15
            for previous in detections
        ):
            detections.append(candidate)

    return detections


# =====================================
# RUN ON ramp_real FOLDER
# =====================================

folder = Path("ramp_real")

if not folder.is_dir():
    raise FileNotFoundError(
        f"Folder not found: {folder.resolve()}"
    )

images = sorted(
    p for p in folder.iterdir()
    if p.suffix.lower() in (".png", ".jpg", ".jpeg")
)

if not images:
    raise RuntimeError("No images found in ramp_real")

columns = 2
rows = math.ceil(len(images) / columns)

fig, axes = plt.subplots(
    rows, columns,
    figsize=(15, 4.5 * rows),
    squeeze=False
)

for ax in axes.flat:
    ax.axis("off")

for index, path in enumerate(images):
    img = cv2.imread(str(path))

    if img is None:
        print(f"Cannot read: {path.name}")
        continue

    detections = detect_ramp(img)
    output = img.copy()

    print(f"\n{path.name}: {len(detections)} ramp detected")

    for x, y, w, h, score in detections:

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            output,
            f"RAMP {score:.2f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(
            f"  bbox=({x}, {y}, {w}, {h}), "
            f"score={score:.3f}"
        )

    ax = axes.flat[index]

    ax.imshow(
        cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    )

    ax.set_title(
        f"{path.name} | Ramp: {len(detections)}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
