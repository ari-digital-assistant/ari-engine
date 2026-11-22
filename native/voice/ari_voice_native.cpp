#include "ari_voice_native.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "MicroWakeWordEngine.h"
#include "OpenWakeWordEngine.h"
#include "SileroVad.h"

// The opaque handles are the engines themselves.
struct AriMicroWakeWord : MicroWakeWordEngine { using MicroWakeWordEngine::MicroWakeWordEngine; };
struct AriOpenWakeWord : OpenWakeWordEngine { using OpenWakeWordEngine::OpenWakeWordEngine; };
struct AriSileroVad : SileroVad { using SileroVad::SileroVad; };

template <typename T, typename... Args>
static T* createIfInitialized(Args&&... args) {
    auto engine = std::make_unique<T>(std::forward<Args>(args)...);
    return engine->isInitialized() ? engine.release() : nullptr;
}

extern "C" {

int32_t ari_voice_native_api_major(void) { return 1; }

AriMicroWakeWord* ari_mww_create(
    const uint8_t* model, size_t model_size, int32_t sample_rate, int32_t feature_step_ms,
    float probability_cutoff, int32_t sliding_window_size, bool reset_on_detection) {
    if (model == nullptr || model_size == 0) return nullptr;
    return createIfInitialized<AriMicroWakeWord>(
        model, model_size, sample_rate, feature_step_ms, probability_cutoff, sliding_window_size,
        reset_on_detection);
}

bool ari_mww_process(AriMicroWakeWord* engine, const int16_t* samples, size_t count) {
    return engine != nullptr && engine->processAudio(samples, count);
}

void ari_mww_reset(AriMicroWakeWord* engine) {
    if (engine != nullptr) engine->reset();
}

void ari_mww_destroy(AriMicroWakeWord* engine) { delete engine; }

AriOpenWakeWord* ari_oww_create(
    const uint8_t* melspectrogram, size_t melspectrogram_size,
    const uint8_t* embedding, size_t embedding_size,
    const uint8_t* classifier, size_t classifier_size,
    float probability_cutoff, bool reset_on_detection) {
    if (melspectrogram == nullptr || embedding == nullptr || classifier == nullptr) return nullptr;
    if (melspectrogram_size == 0 || embedding_size == 0 || classifier_size == 0) return nullptr;
    return createIfInitialized<AriOpenWakeWord>(
        melspectrogram, melspectrogram_size, embedding, embedding_size, classifier, classifier_size,
        probability_cutoff, reset_on_detection);
}

bool ari_oww_process(
    AriOpenWakeWord* engine, const int16_t* samples, size_t count,
    float* scores, size_t scores_capacity, size_t* scores_count) {
    if (scores_count != nullptr) *scores_count = 0;
    if (engine == nullptr) return false;
    if (scores == nullptr) return engine->processAudio(samples, count, nullptr);
    std::vector<float> frameScores;
    bool detected = engine->processAudio(samples, count, &frameScores);
    size_t written = std::min(frameScores.size(), scores_capacity);
    std::copy_n(frameScores.begin(), written, scores);
    if (scores_count != nullptr) *scores_count = written;
    return detected;
}

void ari_oww_reset(AriOpenWakeWord* engine) {
    if (engine != nullptr) engine->reset();
}

void ari_oww_destroy(AriOpenWakeWord* engine) { delete engine; }

AriSileroVad* ari_vad_create(const uint8_t* model, size_t model_size) {
    if (model == nullptr || model_size == 0) return nullptr;
    return createIfInitialized<AriSileroVad>(model, model_size);
}

size_t ari_vad_window_samples(void) { return SileroVad::WINDOW_SAMPLES; }

float ari_vad_compute(AriSileroVad* vad, const float* window) {
    return vad == nullptr ? -1.0f : vad->compute(window);
}

void ari_vad_reset(AriSileroVad* vad) {
    if (vad != nullptr) vad->reset();
}

void ari_vad_destroy(AriSileroVad* vad) { delete vad; }

}  // extern "C"
