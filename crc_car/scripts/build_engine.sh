#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the sign detector's TensorRT engine from a YOLO ONNX model, on the car.
# Run on the Jetson host, NOT inside a container (TensorRT 8.2 kernel generation
# fails in the container). Engines are tied to the TensorRT version and GPU, so
# build on a Jetson Nano with this image; identical cars can share the result.
#
#   bash build_engine.sh model.onnx [labels.txt] [name]
#
#   model.onnx   YOLOv5 or YOLOv8/v11 export with a fixed input size (e.g. imgsz 416),
#                opset <= 17; FP32 or FP16 (--half) exports both work
#   labels.txt   class names, one per line, in training order (optional)
#   name         output name in ~/crc_engines (default: signs -> signs.engine, signs.labels)
#
# Smaller inputs are much faster on the Nano: 640x640 ~47 ms per frame, 416 and 320 less.
# Building takes several minutes. Then restart the drivers: bash run_car.sh bringup
# ---------------------------------------------------------------------------
set -euo pipefail

ONNX="${1:-}"
LABELS="${2:-}"
NAME="${3:-signs}"
OUT_DIR="${ENGINES:-$HOME/crc_engines}"
TRTEXEC=/usr/src/tensorrt/bin/trtexec

die() { echo "ERROR: $*" >&2; exit 1; }
[ -n "$ONNX" ] || die "usage: bash build_engine.sh model.onnx [labels.txt] [name]"
[ -f "$ONNX" ] || die "ONNX file not found: $ONNX"
[ -x "$TRTEXEC" ] || die "trtexec not found (install libnvinfer-bin, see setup_car.sh)"
[ -f /.dockerenv ] && die "run this on the Jetson host, not inside a container"

mkdir -p "$OUT_DIR"
ENGINE="$OUT_DIR/$NAME.engine"
TMP="$OUT_DIR/.$NAME.engine.tmp"
LOG="$OUT_DIR/$NAME.build.log"

echo "Building $ENGINE from $ONNX (FP16, several minutes; log: $LOG)..."
start=$(date +%s)
if ! "$TRTEXEC" --onnx="$ONNX" --fp16 --workspace=1024 --saveEngine="$TMP" > "$LOG" 2>&1 \
   || ! grep -q "PASSED" "$LOG"; then
  rm -f "$TMP"
  grep -E "\[E\]|ERROR" "$LOG" | tail -5 >&2 || true
  die "engine build failed, see $LOG"
fi
mv -f "$TMP" "$ENGINE"

if [ -n "$LABELS" ]; then
  [ -f "$LABELS" ] || die "labels file not found: $LABELS"
  # Normalise line endings and drop empty lines
  tr -d '\r' < "$LABELS" | sed '/^[[:space:]]*$/d' > "$OUT_DIR/$NAME.labels"
  echo "Labels: $(wc -l < "$OUT_DIR/$NAME.labels") classes -> $OUT_DIR/$NAME.labels"
elif [ ! -f "$OUT_DIR/$NAME.labels" ]; then
  echo "Note: no labels file; detections will be reported as class_<id>"
fi

echo "Done in $(( $(date +%s) - start )) s: $ENGINE"
grep -E "Throughput|GPU Compute Time: min" "$LOG" | sed 's/^.*\[I\] /  /' || true
echo "Use it: set CRC_YOLO_ENGINE=/engines/$NAME.engine in ~/crc_car.env, then bash run_car.sh bringup"
