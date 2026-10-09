// UEH CRC 2026 sign detector: YOLO (v5 or v8/v11 export) on TensorRT 8.2, Jetson Nano.
//
// Subscribes /camera/image_raw (rgb8 or bgr8), publishes vision_msgs/Detection2DArray on /signs
// (class name, score, box in image pixels) and, only while someone subscribes, a JPEG with the
// detections drawn on /signs/debug/compressed.
//
// TensorRT and the CUDA runtime are loaded from the host libraries with dlopen, so this node
// builds against headers only. Kept light for the Nano: buffers allocated once, only the newest
// frame is processed (older ones are dropped), optional rate limit, NMS on the CPU.
//
// Parameters:
//   engine       TensorRT engine file (default /engines/signs.engine)
//   labels       class names, one per line (default: engine path with .labels instead of .engine)
//   conf         score threshold (default 0.4)
//   nms          IoU threshold for non-maximum suppression (default 0.45)
//   max_rate     maximum inference rate in Hz, 0 = every frame (default 15)
//   max_det      maximum detections per frame (default 50)
//
// Engine inputs/outputs may be FP32 or FP16 (e.g. an ONNX model exported with --half).

#include <dlfcn.h>

#include <NvInfer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace
{

// ---------------------------------------------------------------------------------------------
// CUDA runtime, resolved at run time from libcudart.so.10.2
// ---------------------------------------------------------------------------------------------
struct CudaApi
{
  using Malloc = int (*)(void **, size_t);
  using Free = int (*)(void *);
  using HostAlloc = int (*)(void **, size_t, unsigned int);
  using FreeHost = int (*)(void *);
  using MemcpyAsync = int (*)(void *, const void *, size_t, int, cudaStream_t);
  using StreamCreate = int (*)(cudaStream_t *);
  using StreamSync = int (*)(cudaStream_t);
  using StreamDestroy = int (*)(cudaStream_t);
  using ErrorString = const char * (*)(int);

  static constexpr int kHostToDevice = 1;
  static constexpr int kDeviceToHost = 2;

  Malloc malloc = nullptr;
  Free free = nullptr;
  HostAlloc host_alloc = nullptr;
  FreeHost free_host = nullptr;
  MemcpyAsync memcpy_async = nullptr;
  StreamCreate stream_create = nullptr;
  StreamSync stream_sync = nullptr;
  StreamDestroy stream_destroy = nullptr;
  ErrorString error_string = nullptr;

  void load()
  {
    void * lib = dlopen("libcudart.so.10.2", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
      lib = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!lib) {
      throw std::runtime_error(std::string("cannot load the CUDA runtime: ") + dlerror());
    }
    resolve(lib, "cudaMalloc", malloc);
    resolve(lib, "cudaFree", free);
    resolve(lib, "cudaHostAlloc", host_alloc);
    resolve(lib, "cudaFreeHost", free_host);
    resolve(lib, "cudaMemcpyAsync", memcpy_async);
    resolve(lib, "cudaStreamCreate", stream_create);
    resolve(lib, "cudaStreamSynchronize", stream_sync);
    resolve(lib, "cudaStreamDestroy", stream_destroy);
    resolve(lib, "cudaGetErrorString", error_string);
  }

  void check(int status, const char * what) const
  {
    if (status != 0) {
      throw std::runtime_error(std::string(what) + " failed: " + error_string(status));
    }
  }

private:
  template<typename T>
  static void resolve(void * lib, const char * name, T & fn)
  {
    fn = reinterpret_cast<T>(dlsym(lib, name));
    if (!fn) {
      throw std::runtime_error(std::string("missing CUDA symbol ") + name);
    }
  }
};

class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(rclcpp::Logger logger)
  : logger_(logger) {}

  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kERROR) {
      RCLCPP_ERROR(logger_, "TensorRT: %s", msg);
    } else if (severity == Severity::kWARNING) {
      RCLCPP_WARN(logger_, "TensorRT: %s", msg);
    }
  }

private:
  rclcpp::Logger logger_;
};

struct Detection
{
  float x1, y1, x2, y2, score;
  int cls;
};

float iou(const Detection & a, const Detection & b)
{
  const float ix = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float iy = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float inter = ix * iy;
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

}  // namespace

class YoloTrtNode : public rclcpp::Node
{
public:
  YoloTrtNode()
  : Node("yolo_trt"), trt_logger_(get_logger())
  {
    engine_path_ = declare_parameter<std::string>("engine", "/engines/signs.engine");
    std::string labels_path = declare_parameter<std::string>("labels", "");
    conf_ = static_cast<float>(number_parameter("conf", 0.4));
    nms_ = static_cast<float>(number_parameter("nms", 0.45));
    const double max_rate = number_parameter("max_rate", 15.0);
    max_det_ = static_cast<size_t>(declare_parameter<int>("max_det", 50));
    min_period_ = max_rate > 0.0 ? 1.0 / max_rate : 0.0;

    if (labels_path.empty()) {
      labels_path = engine_path_.substr(0, engine_path_.rfind('.')) + ".labels";
    }
    load_labels(labels_path);
    cuda_.load();
    load_engine();

    pub_ = create_publisher<vision_msgs::msg::Detection2DArray>("signs", 10);
    pub_debug_ = create_publisher<sensor_msgs::msg::CompressedImage>("signs/debug/compressed", 2);
    // Depth 1, best effort: only the newest frame waits while an inference runs
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "camera/image_raw", rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {on_image(msg);});
    timer_ = create_wall_timer(std::chrono::seconds(5), [this]() {report();});
  }

  ~YoloTrtNode() override
  {
    if (cuda_.stream_sync && stream_) {
      cuda_.stream_sync(stream_);
      cuda_.stream_destroy(stream_);
    }
    for (void * p : device_) {
      if (p) {cuda_.free(p);}
    }
    if (host_in_) {cuda_.free_host(host_in_);}
    if (host_out_) {cuda_.free_host(host_out_);}
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

private:
  // Numeric parameter that accepts both 15 and 15.0 (values come from an env file)
  double number_parameter(const std::string & name, double default_value)
  {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.dynamic_typing = true;
    const rclcpp::ParameterValue value =
      declare_parameter(name, rclcpp::ParameterValue(default_value), descriptor);
    return value.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER ?
           static_cast<double>(value.get<int64_t>()) : value.get<double>();
  }

  void load_labels(const std::string & path)
  {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      line.erase(line.find_last_not_of(" \r\n\t") + 1);
      if (!line.empty()) {labels_.push_back(line);}
    }
    if (labels_.empty()) {
      RCLCPP_WARN(get_logger(), "No labels in %s: classes are reported as class_<id>", path.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Loaded %zu class names from %s", labels_.size(), path.c_str());
    }
  }

  void load_engine()
  {
    void * nvinfer = dlopen("libnvinfer.so.8", RTLD_NOW | RTLD_GLOBAL);
    if (!nvinfer) {
      throw std::runtime_error(std::string("cannot load TensorRT: ") + dlerror());
    }
    // Plugins are optional (plain YOLO exports do not use them)
    if (void * plugins = dlopen("libnvinfer_plugin.so.8", RTLD_NOW | RTLD_GLOBAL)) {
      using InitPlugins = bool (*)(void *, const char *);
      if (auto init = reinterpret_cast<InitPlugins>(dlsym(plugins, "initLibNvInferPlugins"))) {
        init(&trt_logger_, "");
      }
    }
    using CreateRuntime = void * (*)(void *, int32_t);
    auto create = reinterpret_cast<CreateRuntime>(dlsym(nvinfer, "createInferRuntime_INTERNAL"));
    if (!create) {
      throw std::runtime_error("createInferRuntime_INTERNAL not found in libnvinfer");
    }
    runtime_.reset(static_cast<nvinfer1::IRuntime *>(create(&trt_logger_, NV_TENSORRT_VERSION)));

    std::ifstream f(engine_path_, std::ios::binary);
    if (!f) {
      throw std::runtime_error("cannot open engine " + engine_path_);
    }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) {
      throw std::runtime_error("cannot deserialize " + engine_path_ +
                ": build it on this Jetson with trtexec (TensorRT 8.2)");
    }
    context_.reset(engine_->createExecutionContext());

    const int n = engine_->getNbBindings();
    if (n != 2) {
      throw std::runtime_error("expected 1 input and 1 output binding, got " + std::to_string(n));
    }
    device_.assign(n, nullptr);
    for (int i = 0; i < n; ++i) {
      const nvinfer1::Dims d = engine_->getBindingDimensions(i);
      std::ostringstream shape;
      for (int k = 0; k < d.nbDims; ++k) {shape << (k ? "x" : "") << d.d[k];}
      RCLCPP_INFO(
        get_logger(), "binding %d '%s': %s, %s, data type %d", i, engine_->getBindingName(i),
        engine_->bindingIsInput(i) ? "input" : "output", shape.str().c_str(),
        static_cast<int>(engine_->getBindingDataType(i)));
      const nvinfer1::DataType type = engine_->getBindingDataType(i);
      if (type != nvinfer1::DataType::kFLOAT && type != nvinfer1::DataType::kHALF) {
        throw std::runtime_error("engine I/O must be FP32 or FP16");
      }
      const size_t elem = type == nvinfer1::DataType::kHALF ? 2 : 4;
      size_t volume = 1;
      for (int k = 0; k < d.nbDims; ++k) {volume *= static_cast<size_t>(d.d[k]);}
      cuda_.check(cuda_.malloc(&device_[i], volume * elem), "cudaMalloc");
      if (engine_->bindingIsInput(i)) {
        in_idx_ = i;
        in_elem_ = elem;
        in_h_ = d.d[2];
        in_w_ = d.d[3];
        in_count_ = volume;
      } else {
        out_idx_ = i;
        out_elem_ = elem;
        out_count_ = volume;
        // YOLOv5: [1, boxes, 5 + classes] (objectness); YOLOv8/v11: [1, 4 + classes, boxes]
        if (d.d[1] > d.d[2]) {
          layout_v5_ = true;
          boxes_ = d.d[1];
          channels_ = d.d[2];
        } else {
          layout_v5_ = false;
          channels_ = d.d[1];
          boxes_ = d.d[2];
        }
      }
    }
    cuda_.check(cuda_.host_alloc(&host_in_, in_count_ * in_elem_, 0), "cudaHostAlloc");
    cuda_.check(cuda_.host_alloc(&host_out_, out_count_ * out_elem_, 0), "cudaHostAlloc");
    // FP32 staging buffer for the input, only needed when the engine input is FP16
    if (in_elem_ == 2) {in_float_.create(1, static_cast<int>(in_count_), CV_32F);}

    cuda_.check(cuda_.stream_create(&stream_), "cudaStreamCreate");
    num_classes_ = channels_ - (layout_v5_ ? 5 : 4);
    RCLCPP_INFO(
      get_logger(), "Engine %s: input %dx%d %s, %d boxes, %d classes (%s layout), conf %.2f, rate <= %s",
      engine_path_.c_str(), in_w_, in_h_, in_elem_ == 2 ? "FP16" : "FP32", boxes_, num_classes_,
      layout_v5_ ? "YOLOv5" : "YOLOv8",
      conf_, min_period_ > 0 ? (std::to_string(static_cast<int>(1.0 / min_period_)) + " Hz").c_str() : "camera rate");
    if (!labels_.empty() && static_cast<int>(labels_.size()) != num_classes_) {
      RCLCPP_WARN(get_logger(), "%zu labels for %d classes", labels_.size(), num_classes_);
    }
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    const auto now = std::chrono::steady_clock::now();
    if (min_period_ > 0 &&
      std::chrono::duration<double>(now - last_run_).count() < min_period_ * 0.95)
    {
      return;
    }
    last_run_ = now;
    const bool bgr = msg->encoding == "bgr8";
    if (!bgr && msg->encoding != "rgb8") {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "unsupported encoding %s", msg->encoding.c_str());
      return;
    }
    const cv::Mat image(static_cast<int>(msg->height), static_cast<int>(msg->width), CV_8UC3,
      const_cast<uint8_t *>(msg->data.data()), msg->step);

    // --- Pre-process: letterbox to the network size, RGB planes, scaled to 0..1 ---
    const auto t0 = std::chrono::steady_clock::now();
    const float scale = std::min(static_cast<float>(in_w_) / image.cols, static_cast<float>(in_h_) / image.rows);
    const int nw = static_cast<int>(std::round(image.cols * scale));
    const int nh = static_cast<int>(std::round(image.rows * scale));
    const int dx = (in_w_ - nw) / 2;
    const int dy = (in_h_ - nh) / 2;
    if (canvas_.empty()) {canvas_.create(in_h_, in_w_, CV_8UC3);}
    canvas_.setTo(cv::Scalar(114, 114, 114));
    cv::resize(image, canvas_(cv::Rect(dx, dy, nw, nh)), cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    // FP32 planes go straight into the pinned input buffer, or into a staging buffer for FP16
    float * in = in_elem_ == 2 ? in_float_.ptr<float>() : static_cast<float *>(host_in_);
    const size_t plane = static_cast<size_t>(in_w_) * in_h_;
    std::vector<cv::Mat> planes = {
      cv::Mat(in_h_, in_w_, CV_32F, in + (bgr ? 2 : 0) * plane),
      cv::Mat(in_h_, in_w_, CV_32F, in + plane),
      cv::Mat(in_h_, in_w_, CV_32F, in + (bgr ? 0 : 2) * plane)};
    cv::Mat as_float;
    canvas_.convertTo(as_float, CV_32FC3, 1.0 / 255.0);
    cv::split(as_float, planes);
    if (in_elem_ == 2) {
      cv::Mat half(1, static_cast<int>(in_count_), CV_16F, host_in_);
      in_float_.convertTo(half, CV_16F);
    }

    // --- Inference ---
    const auto t1 = std::chrono::steady_clock::now();
    cuda_.check(cuda_.memcpy_async(device_[in_idx_], host_in_, in_count_ * in_elem_,
      CudaApi::kHostToDevice, stream_), "cudaMemcpyAsync");
    if (!context_->enqueueV2(device_.data(), stream_, nullptr)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "TensorRT enqueue failed");
      return;
    }
    cuda_.check(cuda_.memcpy_async(host_out_, device_[out_idx_], out_count_ * out_elem_,
      CudaApi::kDeviceToHost, stream_), "cudaMemcpyAsync");
    cuda_.check(cuda_.stream_sync(stream_), "cudaStreamSynchronize");

    // --- Post-process: decode, map back to image pixels, NMS ---
    const auto t2 = std::chrono::steady_clock::now();
    // FP16 outputs are read in place: only values that are actually inspected get converted
    const std::vector<Detection> dets = out_elem_ == 2 ?
      decode(static_cast<const cv::float16_t *>(host_out_), scale, dx, dy, image.cols, image.rows) :
      decode(static_cast<const float *>(host_out_), scale, dx, dy, image.cols, image.rows);
    publish(msg, dets);
    if (pub_debug_->get_subscription_count() > 0) {publish_debug(msg, image, bgr, dets);}
    const auto t3 = std::chrono::steady_clock::now();

    frames_++;
    pre_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    inf_ms_ += std::chrono::duration<double, std::milli>(t2 - t1).count();
    post_ms_ += std::chrono::duration<double, std::milli>(t3 - t2).count();
  }

  template<typename T>
  std::vector<Detection> decode(const T * out, float scale, int dx, int dy, int img_w, int img_h)
  {
    std::vector<Detection> cand;
    auto at = [&](int box, int ch) -> float {
        return static_cast<float>(layout_v5_ ? out[static_cast<size_t>(box) * channels_ + ch] :
               out[static_cast<size_t>(ch) * boxes_ + box]);
      };
    const int first_class = layout_v5_ ? 5 : 4;
    for (int b = 0; b < boxes_; ++b) {
      float obj = 1.f;
      if (layout_v5_) {
        obj = at(b, 4);
        if (obj < conf_) {continue;}
      }
      int best = 0;
      float best_score = at(b, first_class);
      for (int c = 1; c < num_classes_; ++c) {
        const float s = at(b, first_class + c);
        if (s > best_score) {best_score = s; best = c;}
      }
      const float score = obj * best_score;
      if (score < conf_) {continue;}
      const float cx = (at(b, 0) - dx) / scale;
      const float cy = (at(b, 1) - dy) / scale;
      const float w = at(b, 2) / scale;
      const float h = at(b, 3) / scale;
      cand.push_back({
          std::max(0.f, cx - w / 2), std::max(0.f, cy - h / 2),
          std::min(static_cast<float>(img_w), cx + w / 2), std::min(static_cast<float>(img_h), cy + h / 2),
          score, best});
    }
    std::sort(cand.begin(), cand.end(), [](const Detection & a, const Detection & b) {return a.score > b.score;});
    std::vector<Detection> keep;
    for (const auto & d : cand) {
      bool suppressed = false;
      for (const auto & k : keep) {
        if (k.cls == d.cls && iou(k, d) > nms_) {suppressed = true; break;}
      }
      if (!suppressed) {
        keep.push_back(d);
        if (keep.size() >= max_det_) {break;}
      }
    }
    return keep;
  }

  std::string label(int cls) const
  {
    return cls >= 0 && cls < static_cast<int>(labels_.size()) ? labels_[cls] : "class_" + std::to_string(cls);
  }

  void publish(const sensor_msgs::msg::Image::ConstSharedPtr & msg, const std::vector<Detection> & dets)
  {
    vision_msgs::msg::Detection2DArray out;
    out.header = msg->header;
    out.detections.reserve(dets.size());
    for (const auto & d : dets) {
      vision_msgs::msg::Detection2D det;
      det.header = msg->header;
      det.id = label(d.cls);
      det.bbox.center.position.x = (d.x1 + d.x2) / 2.0;
      det.bbox.center.position.y = (d.y1 + d.y2) / 2.0;
      det.bbox.size_x = d.x2 - d.x1;
      det.bbox.size_y = d.y2 - d.y1;
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = det.id;
      hyp.hypothesis.score = d.score;
      det.results.push_back(hyp);
      out.detections.push_back(det);
    }
    detections_ += dets.size();
    pub_->publish(out);
  }

  void publish_debug(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg, const cv::Mat & image, bool bgr,
    const std::vector<Detection> & dets)
  {
    cv::Mat view;
    if (bgr) {view = image.clone();} else {cv::cvtColor(image, view, cv::COLOR_RGB2BGR);}
    for (const auto & d : dets) {
      cv::rectangle(view, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2), cv::Scalar(0, 255, 0), 2);
      std::ostringstream text;
      text << label(d.cls) << " " << static_cast<int>(d.score * 100) << "%";
      cv::putText(view, text.str(), cv::Point(d.x1, std::max(12.f, d.y1 - 4)),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    sensor_msgs::msg::CompressedImage out;
    out.header = msg->header;
    out.format = "jpeg";
    cv::imencode(".jpg", view, out.data, {cv::IMWRITE_JPEG_QUALITY, 70});
    pub_debug_->publish(out);
  }

  void report()
  {
    if (frames_ == 0) {
      RCLCPP_WARN(get_logger(), "no images on /camera/image_raw");
      return;
    }
    RCLCPP_INFO(
      get_logger(), "%.1f fps | pre %.1f ms, inference %.1f ms, post %.1f ms | %.1f detections/frame",
      frames_ / 5.0, pre_ms_ / frames_, inf_ms_ / frames_, post_ms_ / frames_,
      static_cast<double>(detections_) / frames_);
    frames_ = 0;
    detections_ = 0;
    pre_ms_ = inf_ms_ = post_ms_ = 0.0;
  }

  struct TrtDelete
  {
    template<typename T>
    void operator()(T * p) const {delete p;}
  };

  TrtLogger trt_logger_;
  CudaApi cuda_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDelete> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDelete> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDelete> context_;
  std::vector<void *> device_;
  void * host_in_ = nullptr;
  void * host_out_ = nullptr;
  cudaStream_t stream_ = nullptr;
  int in_idx_ = 0, out_idx_ = 1, in_w_ = 0, in_h_ = 0;
  size_t in_count_ = 0, out_count_ = 0, in_elem_ = 4, out_elem_ = 4;
  cv::Mat in_float_;
  bool layout_v5_ = true;
  int boxes_ = 0, channels_ = 0, num_classes_ = 0;
  cv::Mat canvas_;

  std::string engine_path_;
  std::vector<std::string> labels_;
  float conf_, nms_;
  size_t max_det_;
  double min_period_;
  std::chrono::steady_clock::time_point last_run_{};

  size_t frames_ = 0, detections_ = 0;
  double pre_ms_ = 0, inf_ms_ = 0, post_ms_ = 0;

  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_debug_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<YoloTrtNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("yolo_trt"), "%s", e.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
