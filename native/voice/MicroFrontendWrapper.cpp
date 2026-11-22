#include "MicroFrontendWrapper.h"

#include <algorithm>

extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}

#include "Logging.h"

static constexpr char LOG_TAG[] = "MicroFrontendWrapper";

// Configuration matching ESPHome microWakeWord component.
// Source: https://github.com/esphome/esphome/blob/8d1379a2752291d2f4e33d5831d51c2afd59f7ce/esphome/components/micro_wake_word/preprocessor_settings.h
namespace {
    constexpr size_t FEATURE_DURATION_MS = 30;
    constexpr float FILTERBANK_LOWER_BAND_LIMIT = 125.0f;
    constexpr float FILTERBANK_UPPER_BAND_LIMIT = 7500.0f;

    // Noise reduction settings
    constexpr int NOISE_REDUCTION_SMOOTHING_BITS = 10;
    constexpr float NOISE_REDUCTION_EVEN_SMOOTHING = 0.025f;
    constexpr float NOISE_REDUCTION_ODD_SMOOTHING = 0.06f;
    constexpr float NOISE_REDUCTION_MIN_SIGNAL_REMAINING = 0.05f;

    // PCAN gain control settings
    constexpr float PCAN_GAIN_CONTROL_STRENGTH = 0.95f;
    constexpr float PCAN_GAIN_CONTROL_OFFSET = 80.0f;
    constexpr int PCAN_GAIN_CONTROL_GAIN_BITS = 21;

    // Log scale settings
    constexpr int LOG_SCALE_SCALE_SHIFT = 6;

    // Scale factor to convert uint16 MicroFrontend output to float (1/256 * 10)
    // Source: https://github.com/OHF-Voice/micro-wake-word/blob/a70bd740d4e79ee8a8bb3db843fe862b88d5d6b0/microwakeword/inference.py#L94
    constexpr float FLOAT32_SCALE = 0.0390625f;
}

MicroFrontendWrapper::MicroFrontendWrapper(int sampleRate, size_t stepSizeMs)
    : sampleRate_(sampleRate), stepSizeMs_(stepSizeMs) {

    struct FrontendConfig config{};
    FrontendFillConfigWithDefaults(&config);

    // Window config
    config.window.size_ms = FEATURE_DURATION_MS;
    config.window.step_size_ms = stepSizeMs_;

    // Filterbank config
    config.filterbank.num_channels = PREPROCESSOR_FEATURE_SIZE;
    config.filterbank.lower_band_limit = FILTERBANK_LOWER_BAND_LIMIT;
    config.filterbank.upper_band_limit = FILTERBANK_UPPER_BAND_LIMIT;

    // Noise reduction config
    config.noise_reduction.smoothing_bits = NOISE_REDUCTION_SMOOTHING_BITS;
    config.noise_reduction.even_smoothing = NOISE_REDUCTION_EVEN_SMOOTHING;
    config.noise_reduction.odd_smoothing = NOISE_REDUCTION_ODD_SMOOTHING;
    config.noise_reduction.min_signal_remaining = NOISE_REDUCTION_MIN_SIGNAL_REMAINING;

    // PCAN gain control config
    config.pcan_gain_control.enable_pcan = 1;
    config.pcan_gain_control.strength = PCAN_GAIN_CONTROL_STRENGTH;
    config.pcan_gain_control.offset = PCAN_GAIN_CONTROL_OFFSET;
    config.pcan_gain_control.gain_bits = PCAN_GAIN_CONTROL_GAIN_BITS;

    // Log scale config
    config.log_scale.enable_log = 1;
    config.log_scale.scale_shift = LOG_SCALE_SCALE_SHIFT;

    if (!FrontendPopulateState(&config, &state_, sampleRate_)) {
        LOGE(LOG_TAG, "Failed to initialize MicroFrontend state");
        initialized_ = false;
        return;
    }

    initialized_ = true;
    LOGD(LOG_TAG, "MicroFrontend initialized: %d Hz sample rate, %zu mel bins, %.0f-%.0f Hz, %zum window, %zum step",
         sampleRate_, PREPROCESSOR_FEATURE_SIZE, FILTERBANK_LOWER_BAND_LIMIT, FILTERBANK_UPPER_BAND_LIMIT,
         FEATURE_DURATION_MS, stepSizeMs_);
}

MicroFrontendWrapper::~MicroFrontendWrapper() {
    if (initialized_) {
        FrontendFreeStateContents(&state_);
    }
}

size_t MicroFrontendWrapper::nextFrame(const int16_t* samples, size_t numSamples, const float** frameOut) {
    *frameOut = nullptr;

    if (!initialized_) {
        LOGE(LOG_TAG, "MicroFrontend not initialized");
        return 0;
    }

    size_t numSamplesRead = 0;
    struct FrontendOutput output = FrontendProcessSamples(
        &state_, samples, numSamples, &numSamplesRead
    );

    if (numSamplesRead == 0) {
        return 0;
    }

    if (output.values != nullptr && output.size > 0) {
        const size_t channels = std::min(output.size, PREPROCESSOR_FEATURE_SIZE);
        for (size_t index = 0; index < channels; index++) {
            frame_[index] = static_cast<float>(output.values[index]) * FLOAT32_SCALE;
        }
        std::fill(frame_.begin() + channels, frame_.end(), 0.0f);
        *frameOut = frame_.data();
    }

    return numSamplesRead;
}

void MicroFrontendWrapper::reset() {
    if (initialized_) {
        FrontendReset(&state_);
    }
}
