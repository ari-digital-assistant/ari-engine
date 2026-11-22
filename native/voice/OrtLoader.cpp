#include "OrtLoader.h"

#include <dlfcn.h>

#include "Logging.h"

static constexpr char LOG_TAG[] = "OrtLoader";

/**
 * The ONNX Runtime inside sherpa-onnx's AAR, which the app already ships. Loaded
 * rather than linked so the build needs only its headers, and a second copy of
 * libonnxruntime.so never reaches the APK. Its version can move with sherpa-onnx;
 * any runtime at least as new as the headers serves ORT_API_VERSION.
 */
static const OrtApi* loadOrtApi() {
    void* library = dlopen("libonnxruntime.so", RTLD_NOW);
    if (library == nullptr) {
        LOGE(LOG_TAG, "Could not load libonnxruntime.so: %s", dlerror());
        return nullptr;
    }
    using GetApiBase = const OrtApiBase* (*)();
    auto getApiBase = reinterpret_cast<GetApiBase>(dlsym(library, "OrtGetApiBase"));
    if (getApiBase == nullptr) {
        LOGE(LOG_TAG, "libonnxruntime.so has no OrtGetApiBase");
        return nullptr;
    }
    const OrtApi* api = getApiBase()->GetApi(ORT_API_VERSION);
    if (api == nullptr) {
        LOGE(LOG_TAG, "ONNX Runtime %s is older than API version %d",
             getApiBase()->GetVersionString(), ORT_API_VERSION);
    }
    return api;
}

const OrtApi* ariOrtApi() {
    static const OrtApi* api = loadOrtApi();
    return api;
}
