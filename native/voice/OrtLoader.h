#ifndef ARI_ORT_LOADER_H
#define ARI_ORT_LOADER_H

#include "onnxruntime_c_api.h"

/**
 * The ONNX Runtime API, from the libonnxruntime.so the host already ships, or
 * null when it cannot be loaded. Loaded once and shared by every engine here.
 */
const OrtApi* ariOrtApi();

#endif // ARI_ORT_LOADER_H
