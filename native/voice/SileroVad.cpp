#include "SileroVad.h"

#include <algorithm>
#include <cstring>

#include "Logging.h"
#include "OrtLoader.h"

static constexpr char LOG_TAG[] = "SileroVad";

SileroVad::SileroVad(const uint8_t* model, size_t modelSize) {
    ort_ = ariOrtApi();
    if (ort_ == nullptr) return;

    if (!check(ort_->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "silero-vad", &env_), "CreateEnv")) return;
    if (!check(ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo_), "CreateCpuMemoryInfo")) {
        return;
    }

    // sherpa's options for a VAD with num_threads = 1: one intra-op and one
    // inter-op thread, everything else ONNX Runtime's default.
    OrtSessionOptions* options = nullptr;
    if (!check(ort_->CreateSessionOptions(&options), "CreateSessionOptions")) return;
    bool ok = check(ort_->SetIntraOpNumThreads(options, 1), "SetIntraOpNumThreads")
        && check(ort_->SetInterOpNumThreads(options, 1), "SetInterOpNumThreads")
        && check(ort_->CreateSessionFromArray(env_, model, modelSize, options, &session_), "CreateSession");
    ort_->ReleaseSessionOptions(options);
    if (!ok) return;

    size_t inputCount = 0;
    size_t outputCount = 0;
    if (!check(ort_->SessionGetInputCount(session_, &inputCount), "SessionGetInputCount")) return;
    if (!check(ort_->SessionGetOutputCount(session_, &outputCount), "SessionGetOutputCount")) return;
    if (inputCount != inputNames_.size() || outputCount != outputNames_.size()) {
        LOGE(LOG_TAG, "Expected silero v4 as k2-fsa exports it (3 inputs, 3 outputs), got %zu and %zu",
             inputCount, outputCount);
        return;
    }
    OrtAllocator* allocator = nullptr;
    if (!check(ort_->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions")) return;
    for (size_t i = 0; i < inputNames_.size(); ++i) {
        if (!check(ort_->SessionGetInputName(session_, i, allocator, &inputNames_[i]), "SessionGetInputName")) return;
    }
    for (size_t i = 0; i < outputNames_.size(); ++i) {
        if (!check(ort_->SessionGetOutputName(session_, i, allocator, &outputNames_[i]), "SessionGetOutputName")) {
            return;
        }
    }

    const int64_t windowShape[] = {1, WINDOW_SAMPLES};
    const int64_t stateShape[] = {2, 1, 64};
    const int64_t probabilityShape[] = {1, 1};
    if (!createTensor(window_.data(), windowShape, 2, &inputs_[0])) return;
    if (!createTensor(h_.data(), stateShape, 3, &inputs_[1])) return;
    if (!createTensor(c_.data(), stateShape, 3, &inputs_[2])) return;
    if (!createTensor(probability_.data(), probabilityShape, 2, &outputs_[0])) return;
    if (!createTensor(hOut_.data(), stateShape, 3, &outputs_[1])) return;
    if (!createTensor(cOut_.data(), stateShape, 3, &outputs_[2])) return;

    reset();
    initialized_ = true;
}

SileroVad::~SileroVad() {
    releaseAll();
}

float SileroVad::compute(const float* window) {
    if (!initialized_) return -1.0f;
    std::copy(window, window + WINDOW_SAMPLES, window_.begin());
    if (!check(ort_->Run(session_, nullptr,
                         inputNames_.data(), inputs_.data(), inputs_.size(),
                         outputNames_.data(), outputs_.size(), outputs_.data()), "Run")) {
        return -1.0f;
    }
    // The outputs land in their own buffers, so the state is carried over here
    // rather than by swapping tensors, as sherpa does.
    h_ = hOut_;
    c_ = cOut_;
    return probability_[0];
}

void SileroVad::reset() {
    h_.fill(0.0f);
    c_.fill(0.0f);
}

bool SileroVad::check(OrtStatus* status, const char* what) const {
    if (status == nullptr) return true;
    LOGE(LOG_TAG, "%s failed: %s", what, ort_->GetErrorMessage(status));
    ort_->ReleaseStatus(status);
    return false;
}

bool SileroVad::createTensor(float* data, const int64_t* shape, size_t rank, OrtValue** out) {
    size_t count = 1;
    for (size_t i = 0; i < rank; ++i) count *= static_cast<size_t>(shape[i]);
    return check(ort_->CreateTensorWithDataAsOrtValue(
        memoryInfo_, data, count * sizeof(float), shape, rank,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, out), "CreateTensorWithDataAsOrtValue");
}

void SileroVad::releaseAll() {
    if (ort_ == nullptr) return;
    for (OrtValue* value : inputs_) {
        if (value != nullptr) ort_->ReleaseValue(value);
    }
    for (OrtValue* value : outputs_) {
        if (value != nullptr) ort_->ReleaseValue(value);
    }
    OrtAllocator* allocator = nullptr;
    if (ort_->GetAllocatorWithDefaultOptions(&allocator) == nullptr) {
        for (char* name : inputNames_) {
            if (name != nullptr) allocator->Free(allocator, name);
        }
        for (char* name : outputNames_) {
            if (name != nullptr) allocator->Free(allocator, name);
        }
    }
    if (session_ != nullptr) ort_->ReleaseSession(session_);
    if (memoryInfo_ != nullptr) ort_->ReleaseMemoryInfo(memoryInfo_);
    if (env_ != nullptr) ort_->ReleaseEnv(env_);
}
