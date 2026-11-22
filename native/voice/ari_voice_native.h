#ifndef ARI_VOICE_NATIVE_H
#define ARI_VOICE_NATIVE_H

/*
 * libari_voice_native: the wake engines and the speech check's VAD, behind a C
 * interface that ari-voice opens at run time. Every create function copies
 * what it is given, returns null on failure, and has a matching destroy.
 * Handles are not thread-safe: each belongs to one thread at a time.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__)
#define ARI_VOICE_EXPORT __attribute__((visibility("default")))
#else
#define ARI_VOICE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this interface. A library with another major is not used. */
ARI_VOICE_EXPORT int32_t ari_voice_native_api_major(void);

/* microWakeWord: a TFLite Micro streaming model over 16 kHz audio. */
typedef struct AriMicroWakeWord AriMicroWakeWord;
ARI_VOICE_EXPORT AriMicroWakeWord* ari_mww_create(
    const uint8_t* model, size_t model_size, int32_t sample_rate, int32_t feature_step_ms,
    float probability_cutoff, int32_t sliding_window_size, bool reset_on_detection);
/* Whether the wake word was detected in these samples. */
ARI_VOICE_EXPORT bool ari_mww_process(AriMicroWakeWord* engine, const int16_t* samples, size_t count);
ARI_VOICE_EXPORT void ari_mww_reset(AriMicroWakeWord* engine);
ARI_VOICE_EXPORT void ari_mww_destroy(AriMicroWakeWord* engine);

/* openWakeWord: two shared feature models and a classifier, on ONNX Runtime. */
typedef struct AriOpenWakeWord AriOpenWakeWord;
ARI_VOICE_EXPORT AriOpenWakeWord* ari_oww_create(
    const uint8_t* melspectrogram, size_t melspectrogram_size,
    const uint8_t* embedding, size_t embedding_size,
    const uint8_t* classifier, size_t classifier_size,
    float probability_cutoff, bool reset_on_detection);
/*
 * Whether the wake word was detected in these samples. When scores is not
 * null, it receives the score of each 80 ms frame the samples complete, up to
 * scores_capacity of them, and *scores_count says how many were written.
 */
ARI_VOICE_EXPORT bool ari_oww_process(
    AriOpenWakeWord* engine, const int16_t* samples, size_t count,
    float* scores, size_t scores_capacity, size_t* scores_count);
ARI_VOICE_EXPORT void ari_oww_reset(AriOpenWakeWord* engine);
ARI_VOICE_EXPORT void ari_oww_destroy(AriOpenWakeWord* engine);

/* Silero VAD (silero_vad.onnx as k2-fsa exports it), one window at a time. */
typedef struct AriSileroVad AriSileroVad;
ARI_VOICE_EXPORT AriSileroVad* ari_vad_create(const uint8_t* model, size_t model_size);
/* The number of samples ari_vad_compute takes. */
ARI_VOICE_EXPORT size_t ari_vad_window_samples(void);
/* The speech probability of one window scaled to [-1, 1); negative on failure. */
ARI_VOICE_EXPORT float ari_vad_compute(AriSileroVad* vad, const float* window);
ARI_VOICE_EXPORT void ari_vad_reset(AriSileroVad* vad);
ARI_VOICE_EXPORT void ari_vad_destroy(AriSileroVad* vad);

#ifdef __cplusplus
}
#endif

#endif /* ARI_VOICE_NATIVE_H */
