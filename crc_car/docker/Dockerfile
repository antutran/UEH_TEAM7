# UEH CRC 2026 - on-car runtime (Jetson Nano, JetPack 4.6 / L4T R32.7)
# ROS 2 Humble on Ubuntu 22.04, matching the simulation environment (Student Pack)
# without Gazebo/RViz. The GPU stack (CUDA 10.2, cuDNN 8.2, TensorRT 8.2) is NOT
# baked into the image: the host libraries are bind-mounted at run time
# (see scripts/run_car.sh).
# Build context: the jetson_ros2 directory (docker build -f docker/Dockerfile .)

# --- Stage 1: build the car packages (compilers stay out of the final image) ---
FROM ros:humble-ros-base AS build
SHELL ["/bin/bash", "-c"]
# Captive-portal networks (e.g. campus Wi-Fi) rewrite plain HTTP to a login page,
# so every apt source is switched to HTTPS. packages.ros.org has no valid HTTPS
# certificate of its own; its host ftp.osuosl.org serves the same tree over HTTPS.
RUN sed -i -e 's#http://packages.ros.org/#https://ftp.osuosl.org/pub/ros/packages.ros.org/#g' \
           -e 's#http://#https://#g' \
      /etc/apt/sources.list \
      $(ls /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null) && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git curl python3-colcon-common-extensions \
      libopencv-dev ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*
# RPLIDAR C1 driver: the apt rplidar_ros (2.1.4) predates C1 support, so build
# Slamtec's sllidar_ros2 from source, pinned to a known commit.
ARG SLLIDAR_COMMIT=34300099fadfc772965962dec837bf436706188f
# Built in its own layer BEFORE the car code is copied, so editing crc_bringup does
# not recompile this C++ driver (2-3 minutes on the Nano); only the next layer reruns.
RUN mkdir -p /opt/crc_ws/src && cd /opt/crc_ws/src && \
    git clone https://github.com/Slamtec/sllidar_ros2.git && \
    git -C sllidar_ros2 checkout -q "$SLLIDAR_COMMIT" && rm -rf sllidar_ros2/.git && \
    source /opt/ros/humble/setup.bash && cd /opt/crc_ws && \
    MAKEFLAGS=-j2 colcon build --merge-install --packages-select sllidar_ros2 \
      --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    rm -rf build log
# TensorRT 8.2.1 public headers (Apache-2.0) for the YOLO node; the libraries come from the
# host at run time, so the image needs neither the CUDA toolkit nor TensorRT itself.
ARG TRT_TAG=8.2.1
RUN mkdir -p /opt/trt/include && cd /opt/trt/include && \
    for h in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferRuntime.h NvInferRuntimeCommon.h NvInferVersion.h; do \
      curl -sSfL -o "$h" "https://raw.githubusercontent.com/NVIDIA/TensorRT/$TRT_TAG/include/$h"; \
    done
COPY ros/crc_yolo /opt/crc_ws/src/crc_yolo
RUN source /opt/ros/humble/setup.bash && cd /opt/crc_ws && \
    MAKEFLAGS=-j2 colcon build --merge-install --packages-select crc_yolo \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DTRT_INCLUDE_DIR=/opt/trt/include && \
    rm -rf build log
COPY ros/crc_bringup /opt/crc_ws/src/crc_bringup
RUN source /opt/ros/humble/setup.bash && cd /opt/crc_ws && \
    colcon build --merge-install --packages-select crc_bringup && \
    rm -rf build log

# --- Stage 2: runtime image ----------------------------------------------------
FROM ros:humble-ros-base
SHELL ["/bin/bash", "-c"]

# libegl1: the Tegra driver libraries mounted from the host need libEGL.so.1,
# which the base image does not ship.
RUN sed -i -e 's#http://packages.ros.org/#https://ftp.osuosl.org/pub/ros/packages.ros.org/#g' \
           -e 's#http://#https://#g' \
      /etc/apt/sources.list \
      $(ls /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources 2>/dev/null) && \
    apt-get update && apt-get install -y --no-install-recommends \
      libegl1 \
      ros-humble-cv-bridge \
      ros-humble-image-transport \
      ros-humble-vision-msgs \
      python3-opencv \
      python3-numpy \
      python3-scipy \
      python3-serial \
      python3-pip \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Extra tools teams commonly use (separate layer so the base layer above stays cached):
# compressed image transport, keyboard teleop for practice, xacro for URDF launch files,
# tf_transformations (quaternion <-> Euler), Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-compressed-image-transport \
      ros-humble-teleop-twist-keyboard \
      ros-humble-xacro \
      ros-humble-tf-transformations \
      python3-transforms3d \
      python3-pil \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime (CPU) for small ONNX models, and simple-pid. --no-deps keeps Ubuntu's numpy 1.21.5,
# which cv2 and cv_bridge are built against (every onnxruntime wheel asks for >= 1.21.6, and an
# upgrade could pull numpy 2); sympy is skipped, it is only used by the quantisation tools.
RUN python3 -m pip install --no-cache-dir --no-deps \
      onnxruntime==1.18.1 simple-pid==2.0.1 flatbuffers coloredlogs humanfriendly && \
    python3 -c "import numpy, cv2, onnxruntime, simple_pid, transforms3d, PIL; \
assert numpy.__version__ == '1.21.5', numpy.__version__; \
print('numpy', numpy.__version__, '| cv2', cv2.__version__, '| onnxruntime', onnxruntime.__version__, \
onnxruntime.get_available_providers())"

COPY --from=build /opt/crc_ws/install /opt/crc_ws/install
COPY docker/bringup_entry.sh /opt/crc_ws/bringup_entry.sh
# Cloud-synced copies can lose the executable bit
RUN chmod +x /opt/crc_ws/bringup_entry.sh

# nvidia-container-runtime only exposes the GPU when NVIDIA_VISIBLE_DEVICES is set.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:/usr/lib/aarch64-linux-gnu/tegra \
    PYTHONDONTWRITEBYTECODE=1 \
    ROS_DOMAIN_ID=30

# Same workspace layout as the Student Pack: team sources in /ws/src (bind mount),
# build artefacts in the /ws_build volume so the source tree stays clean.
RUN mkdir -p /ws/src /ws_build /engines
WORKDIR /ws

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'source /opt/crc_ws/install/setup.bash' >> /root/.bashrc && \
    echo '[ -f /ws_build/install/setup.bash ] && source /ws_build/install/setup.bash' >> /root/.bashrc && \
    echo 'alias build="colcon build --symlink-install --build-base /ws_build/build --install-base /ws_build/install"' >> /root/.bashrc

CMD ["/bin/bash"]
