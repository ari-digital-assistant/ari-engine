#ifndef OPEN_WAKE_WORD_ENGINE_H
#define OPEN_WAKE_WORD_ENGINE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "onnxruntime_c_api.h"

/**
 * openWakeWord's streaming pipeline, ported from openwakeword 0.6.0
 * (utils.py AudioFeatures, model.py Model.predict):
 * - 1280-sample (80 ms) chunks of 16 kHz int16 audio
 * - melspectrogram over the chunk and the 480 samples before it, then x / 10 + 2,
 *   which openwakeword applies in Python, outside the ONNX graph
 * - one 96-value embedding per chunk, from the last 76 mel frames
 * - the classifier's score over the last 16 embeddings
 *
 * A detection is one score over the cutoff, with no window mean, which is the
 * rule the laptop scorer (ari-tools exam_oww.py) measured it on.
 *
 * Runs on the ONNX Runtime that sherpa-onnx bundles, loaded at runtime, and
 * single-threaded.
 */
class OpenWakeWordEngine {
public:
    /**
     * Each model is copied by ONNX Runtime, so the caller can free them after construction.
     *
     * @param resetOnDetection  reset() the moment a detection is reported. False hands that to
     *                          the caller, which then must call reset() on every detection it
     *                          acts on; until it does, the engine reports every score over the
     *                          cutoff.
     */
    OpenWakeWordEngine(
        const uint8_t* melspectrogram, size_t melspectrogramSize,
        const uint8_t* embedding, size_t embeddingSize,
        const uint8_t* classifier, size_t classifierSize,
        float probabilityCutoff,
        bool resetOnDetection
    );
    ~OpenWakeWordEngine();

    OpenWakeWordEngine(const OpenWakeWordEngine&) = delete;
    OpenWakeWordEngine& operator=(const OpenWakeWordEngine&) = delete;
    OpenWakeWordEngine(OpenWakeWordEngine&&) = delete;
    OpenWakeWordEngine& operator=(OpenWakeWordEngine&&) = delete;

    [[nodiscard]] bool isInitialized() const { return initialized_; }

    /**
     * Process audio samples and check for wake word detection.
     *
     * @param scores  If not null, gets the score of every 80 ms frame these samples
     *                complete, 0 while the engine warms up.
     * @return true if wake word was detected. With resetOnDetection, the rest of the
     *         samples after a detection are discarded.
     */
    [[nodiscard]] bool processAudio(const int16_t* samples, size_t numSamples, std::vector<float>* scores);

    /**
     * Forget all audio. The next 24 frames score 0 (see processChunk).
     */
    void reset();

private:
    static constexpr size_t CHUNK_SAMPLES = 1280;
    static constexpr size_t MEL_HISTORY_SAMPLES = 480;
    static constexpr size_t MEL_BINS = 32;
    static constexpr size_t MEL_WINDOW_FRAMES = 76;
    static constexpr size_t EMBEDDING_SIZE = 96;
    static constexpr size_t CLASSIFIER_FRAMES = 16;

    // Mel frames from one chunk: 8 with its 480-sample history, 5 from the first
    // chunk after a reset, which has none.
    static constexpr size_t MEL_FRAMES_PER_CHUNK = 8;
    static constexpr size_t MEL_FRAMES_FIRST_CHUNK = 5;

    struct Session {
        OrtSession* session = nullptr;
        std::string inputName;
        std::string outputName;
    };

    [[nodiscard]] bool check(OrtStatus* status, const char* what) const;
    [[nodiscard]] bool createSession(const uint8_t* data, size_t size, Session& out, const char* what);
    [[nodiscard]] bool createTensor(float* data, const int64_t* shape, size_t rank, OrtValue** out);
    [[nodiscard]] bool run(const Session& session, const OrtValue* input, OrtValue* output);
    [[nodiscard]] bool processChunk(float* score);
    void releaseAll();

    const OrtApi* ort_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtMemoryInfo* memoryInfo_ = nullptr;
    Session melspectrogram_;
    Session embedding_;
    Session classifier_;

    // Audio waiting to make up a whole chunk.
    std::array<int16_t, CHUNK_SAMPLES> pending_{};
    size_t pendingSamples_ = 0;

    // The melspectrogram's input: the previous chunk's last 480 samples, then this chunk.
    std::array<float, MEL_HISTORY_SAMPLES + CHUNK_SAMPLES> melInput_{};
    bool haveHistory_ = false;
    std::array<float, MEL_FRAMES_PER_CHUNK * MEL_BINS> melOutput_{};

    // The last 76 mel frames, oldest first, and how many have arrived since reset.
    std::array<float, MEL_WINDOW_FRAMES * MEL_BINS> melWindow_{};
    size_t melFrames_ = 0;

    // The last 16 embeddings, oldest first, and how many have arrived since reset.
    std::array<float, EMBEDDING_SIZE> embeddingOutput_{};
    std::array<float, CLASSIFIER_FRAMES * EMBEDDING_SIZE> features_{};
    size_t featureFrames_ = 0;
    std::array<float, 1> score_{};

    // Tensors over the buffers above, made once. The melspectrogram has two of each
    // because the first chunk after a reset is shorter.
    OrtValue* melInputFull_ = nullptr;
    OrtValue* melInputFirst_ = nullptr;
    OrtValue* melOutputFull_ = nullptr;
    OrtValue* melOutputFirst_ = nullptr;
    OrtValue* embeddingInput_ = nullptr;
    OrtValue* embeddingOutputValue_ = nullptr;
    OrtValue* classifierInput_ = nullptr;
    OrtValue* scoreValue_ = nullptr;

    float probabilityCutoff_;
    bool resetOnDetection_;
    bool initialized_ = false;
};

#endif // OPEN_WAKE_WORD_ENGINE_H
