#include "OpenWakeWordEngine.h"

#include <algorithm>
#include <cstring>

#include "Logging.h"
#include "OrtLoader.h"

static constexpr char LOG_TAG[] = "OpenWakeWordEngine";

OpenWakeWordEngine::OpenWakeWordEngine(
    const uint8_t* melspectrogram, size_t melspectrogramSize,
    const uint8_t* embedding, size_t embeddingSize,
    const uint8_t* classifier, size_t classifierSize,
    float probabilityCutoff,
    bool resetOnDetection
)
    : probabilityCutoff_(probabilityCutoff)
    , resetOnDetection_(resetOnDetection)
{
    ort_ = ariOrtApi();
    if (ort_ == nullptr) return;

    if (!check(ort_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "openwakeword", &env_), "CreateEnv")) return;
    if (!check(ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo_), "CreateCpuMemoryInfo")) {
        return;
    }
    if (!createSession(melspectrogram, melspectrogramSize, melspectrogram_, "melspectrogram")) return;
    if (!createSession(embedding, embeddingSize, embedding_, "embedding")) return;
    if (!createSession(classifier, classifierSize, classifier_, "classifier")) return;

    const int64_t melInputFull[] = {1, MEL_HISTORY_SAMPLES + CHUNK_SAMPLES};
    const int64_t melInputFirst[] = {1, CHUNK_SAMPLES};
    const int64_t melOutputFull[] = {1, 1, MEL_FRAMES_PER_CHUNK, MEL_BINS};
    const int64_t melOutputFirst[] = {1, 1, MEL_FRAMES_FIRST_CHUNK, MEL_BINS};
    const int64_t embeddingInput[] = {1, MEL_WINDOW_FRAMES, MEL_BINS, 1};
    const int64_t embeddingOutput[] = {1, 1, 1, EMBEDDING_SIZE};
    const int64_t classifierInput[] = {1, CLASSIFIER_FRAMES, EMBEDDING_SIZE};
    const int64_t scoreShape[] = {1, 1};
    if (!createTensor(melInput_.data(), melInputFull, 2, &melInputFull_)) return;
    // The first chunk after a reset sits where a whole chunk would, after the history.
    if (!createTensor(melInput_.data() + MEL_HISTORY_SAMPLES, melInputFirst, 2, &melInputFirst_)) return;
    if (!createTensor(melOutput_.data(), melOutputFull, 4, &melOutputFull_)) return;
    if (!createTensor(melOutput_.data(), melOutputFirst, 4, &melOutputFirst_)) return;
    if (!createTensor(melWindow_.data(), embeddingInput, 4, &embeddingInput_)) return;
    if (!createTensor(embeddingOutput_.data(), embeddingOutput, 4, &embeddingOutputValue_)) return;
    if (!createTensor(features_.data(), classifierInput, 3, &classifierInput_)) return;
    if (!createTensor(score_.data(), scoreShape, 2, &scoreValue_)) return;

    LOGD(LOG_TAG, "Engine initialized: cutoff=%.3f", probabilityCutoff_);
    initialized_ = true;
}

OpenWakeWordEngine::~OpenWakeWordEngine() {
    releaseAll();
}

bool OpenWakeWordEngine::check(OrtStatus* status, const char* what) const {
    if (status == nullptr) return true;
    LOGE(LOG_TAG, "%s failed: %s", what, ort_->GetErrorMessage(status));
    ort_->ReleaseStatus(status);
    return false;
}

bool OpenWakeWordEngine::createSession(const uint8_t* data, size_t size, Session& out, const char* what) {
    OrtSessionOptions* options = nullptr;
    if (!check(ort_->CreateSessionOptions(&options), "CreateSessionOptions")) return false;
    // One thread, and no pool to spin: three small models every 80 ms gain
    // nothing from parallelism that would pay for the wake-ups.
    bool ok = check(ort_->SetIntraOpNumThreads(options, 1), "SetIntraOpNumThreads")
        && check(ort_->SetInterOpNumThreads(options, 1), "SetInterOpNumThreads")
        && check(ort_->SetSessionExecutionMode(options, ORT_SEQUENTIAL), "SetSessionExecutionMode")
        && check(ort_->CreateSessionFromArray(env_, data, size, options, &out.session), what);
    ort_->ReleaseSessionOptions(options);
    if (!ok) return false;

    OrtAllocator* allocator = nullptr;
    if (!check(ort_->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions")) return false;
    char* name = nullptr;
    if (!check(ort_->SessionGetInputName(out.session, 0, allocator, &name), "SessionGetInputName")) return false;
    out.inputName = name;
    allocator->Free(allocator, name);
    if (!check(ort_->SessionGetOutputName(out.session, 0, allocator, &name), "SessionGetOutputName")) return false;
    out.outputName = name;
    allocator->Free(allocator, name);
    return true;
}

bool OpenWakeWordEngine::createTensor(float* data, const int64_t* shape, size_t rank, OrtValue** out) {
    size_t count = 1;
    for (size_t i = 0; i < rank; i++) count *= static_cast<size_t>(shape[i]);
    return check(ort_->CreateTensorWithDataAsOrtValue(
        memoryInfo_, data, count * sizeof(float), shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, out
    ), "CreateTensorWithDataAsOrtValue");
}

bool OpenWakeWordEngine::run(const Session& session, const OrtValue* input, OrtValue* output) {
    const char* inputName = session.inputName.c_str();
    const char* outputName = session.outputName.c_str();
    return check(ort_->Run(session.session, nullptr, &inputName, &input, 1, &outputName, 1, &output), "Run");
}

bool OpenWakeWordEngine::processAudio(const int16_t* samples, size_t numSamples, std::vector<float>* scores) {
    if (!initialized_) return false;

    bool detected = false;
    size_t offset = 0;
    while (offset < numSamples) {
        size_t take = std::min(numSamples - offset, CHUNK_SAMPLES - pendingSamples_);
        std::copy_n(samples + offset, take, pending_.begin() + static_cast<ptrdiff_t>(pendingSamples_));
        pendingSamples_ += take;
        offset += take;
        if (pendingSamples_ < CHUNK_SAMPLES) break;
        pendingSamples_ = 0;

        float score = 0.0f;
        if (!processChunk(&score)) continue;
        if (scores != nullptr) scores->push_back(score);

        if (score > probabilityCutoff_) {
            if (resetOnDetection_) {
                LOGI(LOG_TAG, "Wake word detected");
                reset();
                return true;
            }
            detected = true;
        }
    }
    return detected;
}

bool OpenWakeWordEngine::processChunk(float* score) {
    // openwakeword feeds the melspectrogram int16 values as floats, unscaled.
    float* chunk = melInput_.data() + MEL_HISTORY_SAMPLES;
    for (size_t i = 0; i < CHUNK_SAMPLES; i++) chunk[i] = static_cast<float>(pending_[i]);

    const bool full = haveHistory_;
    if (!run(melspectrogram_, full ? melInputFull_ : melInputFirst_, full ? melOutputFull_ : melOutputFirst_)) {
        return false;
    }
    std::copy(melInput_.end() - MEL_HISTORY_SAMPLES, melInput_.end(), melInput_.begin());
    haveHistory_ = true;

    const size_t newFrames = full ? MEL_FRAMES_PER_CHUNK : MEL_FRAMES_FIRST_CHUNK;
    const size_t newValues = newFrames * MEL_BINS;
    std::copy(melWindow_.begin() + static_cast<ptrdiff_t>(newValues), melWindow_.end(), melWindow_.begin());
    std::transform(melOutput_.begin(), melOutput_.begin() + static_cast<ptrdiff_t>(newValues),
                   melWindow_.end() - static_cast<ptrdiff_t>(newValues),
                   [](float value) { return value / 10.0f + 2.0f; });
    melFrames_ += newFrames;

    *score = 0.0f;
    if (melFrames_ < MEL_WINDOW_FRAMES) return true;
    if (!run(embedding_, embeddingInput_, embeddingOutputValue_)) return false;
    std::copy(features_.begin() + EMBEDDING_SIZE, features_.end(), features_.begin());
    std::copy(embeddingOutput_.begin(), embeddingOutput_.end(), features_.end() - EMBEDDING_SIZE);
    featureFrames_++;

    // The classifier's 16 embeddings are all audio, rather than the buffers'
    // starting contents, from the 25th chunk on: embeddings need 76 mel frames
    // (5 + 8 * 9, the 10th chunk), and then 16 of them. openwakeword starts its
    // buffers on ones and random noise instead and zeroes only 5 scores; the
    // laptop scorer drops the first 2 s, which this matches.
    if (featureFrames_ < CLASSIFIER_FRAMES) return true;
    if (!run(classifier_, classifierInput_, scoreValue_)) return false;
    *score = score_[0];
    return true;
}

void OpenWakeWordEngine::reset() {
    pendingSamples_ = 0;
    haveHistory_ = false;
    melFrames_ = 0;
    featureFrames_ = 0;
}

void OpenWakeWordEngine::releaseAll() {
    if (ort_ == nullptr) return;
    for (OrtValue* value : {melInputFull_, melInputFirst_, melOutputFull_, melOutputFirst_,
                            embeddingInput_, embeddingOutputValue_, classifierInput_, scoreValue_}) {
        if (value != nullptr) ort_->ReleaseValue(value);
    }
    for (Session* session : {&melspectrogram_, &embedding_, &classifier_}) {
        if (session->session != nullptr) ort_->ReleaseSession(session->session);
    }
    if (memoryInfo_ != nullptr) ort_->ReleaseMemoryInfo(memoryInfo_);
    if (env_ != nullptr) ort_->ReleaseEnv(env_);
}
