#ifndef ARI_SILERO_VAD_H
#define ARI_SILERO_VAD_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "onnxruntime_c_api.h"

/**
 * Silero VAD, one 512-sample window at a time: the probability that the window
 * is speech, carrying the model's LSTM state from one window to the next.
 *
 * A port of sherpa-onnx 2.0.1's `SileroVadModel::Compute` (silero-vad-model.cc,
 * RunV4) for the model Ari bundles: silero-vad v4 re-exported by k2-fsa, inputs
 * x [1, 512], h and c [2, 1, 64], outputs the probability, h and c. Sherpa's
 * C API has no per-window score, only its segmenter, and the speech check and
 * both endpoint rules were measured on the score. Same graph, same zeroed
 * state, same single thread and default optimisation level as sherpa, on the
 * same ONNX Runtime, so the numbers are the same.
 *
 * Not thread-safe: one instance belongs to one thread.
 */
class SileroVad {
public:
    static constexpr size_t WINDOW_SAMPLES = 512;

    /** ONNX Runtime copies the model, so the caller can free it after construction. */
    SileroVad(const uint8_t* model, size_t modelSize);
    ~SileroVad();

    SileroVad(const SileroVad&) = delete;
    SileroVad& operator=(const SileroVad&) = delete;
    SileroVad(SileroVad&&) = delete;
    SileroVad& operator=(SileroVad&&) = delete;

    [[nodiscard]] bool isInitialized() const { return initialized_; }

    /**
     * The speech probability of [window], WINDOW_SAMPLES samples scaled to [-1, 1).
     * Advances the state. Negative if the model failed to run.
     */
    [[nodiscard]] float compute(const float* window);

    /** Zero the state, as for a new stream. */
    void reset();

private:
    static constexpr size_t STATE_SIZE = 2 * 1 * 64;

    [[nodiscard]] bool check(OrtStatus* status, const char* what) const;
    [[nodiscard]] bool createTensor(float* data, const int64_t* shape, size_t rank, OrtValue** out);
    void releaseAll();

    const OrtApi* ort_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtMemoryInfo* memoryInfo_ = nullptr;
    OrtSession* session_ = nullptr;

    // Names in the session's own order, which is the order sherpa feeds them.
    std::array<char*, 3> inputNames_{};
    std::array<char*, 3> outputNames_{};

    std::array<float, WINDOW_SAMPLES> window_{};
    std::array<float, STATE_SIZE> h_{};
    std::array<float, STATE_SIZE> c_{};
    std::array<float, 1> probability_{};
    std::array<float, STATE_SIZE> hOut_{};
    std::array<float, STATE_SIZE> cOut_{};

    // Tensors over the buffers above, made once.
    std::array<OrtValue*, 3> inputs_{};
    std::array<OrtValue*, 3> outputs_{};

    bool initialized_ = false;
};

#endif // ARI_SILERO_VAD_H
