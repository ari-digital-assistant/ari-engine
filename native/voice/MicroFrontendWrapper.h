#ifndef MICRO_FRONTEND_WRAPPER_H
#define MICRO_FRONTEND_WRAPPER_H

#include <array>
#include <cstddef>
#include <cstdint>

extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
}

// Number of mel filterbank features per frame (matches ESPHome PREPROCESSOR_FEATURE_SIZE)
constexpr size_t PREPROCESSOR_FEATURE_SIZE = 40;

/**
 * C++ wrapper for TFLite Micro Frontend audio feature extraction.
 *
 * Configuration matches ESPHome microWakeWord component:
 * https://github.com/esphome/esphome/blob/dev/esphome/components/micro_wake_word/preprocessor_settings.h
 */
class MicroFrontendWrapper {
public:
    MicroFrontendWrapper(int sampleRate, size_t stepSizeMs);
    ~MicroFrontendWrapper();

    // Non-copyable, non-movable
    MicroFrontendWrapper(const MicroFrontendWrapper&) = delete;
    MicroFrontendWrapper& operator=(const MicroFrontendWrapper&) = delete;
    MicroFrontendWrapper(MicroFrontendWrapper&&) = delete;
    MicroFrontendWrapper& operator=(MicroFrontendWrapper&&) = delete;

    [[nodiscard]] bool isInitialized() const { return initialized_; }

    /**
     * Extract the next spectrogram frame from a run of audio samples. Call in a
     * loop, advancing by the returned sample count, until it returns 0.
     *
     * @param samples 16-bit PCM audio samples
     * @param numSamples Number of samples
     * @param frameOut Receives a pointer to PREPROCESSOR_FEATURE_SIZE floats when
     *                 a frame completed, nullptr otherwise. Points at an internal
     *                 buffer that the next call overwrites.
     * @return Number of samples consumed; 0 when the frontend needs more audio
     */
    size_t nextFrame(const int16_t* samples, size_t numSamples, const float** frameOut);

    /**
     * Reset internal state (noise estimates, PCAN state, sample buffer).
     */
    void reset();

private:
    struct FrontendState state_{};
    // Reused across frames — the wake loop runs forever, so allocating one of
    // these per frame is ~100 heap round-trips a second for 40 floats.
    std::array<float, PREPROCESSOR_FEATURE_SIZE> frame_{};
    int sampleRate_;
    size_t stepSizeMs_;
    bool initialized_ = false;
};

#endif // MICRO_FRONTEND_WRAPPER_H
