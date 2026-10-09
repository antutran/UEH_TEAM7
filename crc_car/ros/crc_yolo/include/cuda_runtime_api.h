// Minimal stand-in for the CUDA runtime header: the TensorRT 8.2 headers only need these two
// opaque handle types. The CUDA functions themselves are resolved with dlsym at run time
// (see src/yolo_trt_node.cpp), so this package builds without the CUDA toolkit.
#pragma once

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
