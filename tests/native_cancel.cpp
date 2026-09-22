// Exercise the real render() with a host that cancels inside clipGetImage.
#include "../ofx/src/sam3_ofx.cpp"
#include <stdexcept>

static Instance instance{};
static bool aborted = false, cancelOnFetch = false, failFetch = true, partial = false;
static int targetFetch = 1, fetches = 0, releases = 0, messages = 0;
static float pixels[16] = {};
static std::string lastMessage;
static bool playback = false;
static int sourceFetches = 0;
static double currentFrame = 27;

static OfxStatus properties(OfxImageEffectHandle, OfxPropertySetHandle *out) {
    *out = reinterpret_cast<OfxPropertySetHandle>(&instance); return kOfxStatOK;
}
static OfxStatus pointerProperty(OfxPropertySetHandle, const char *name, int, void **out) {
    *out = strcmp(name, kOfxPropInstanceData) == 0 ? static_cast<void *>(&instance) : pixels;
    return kOfxStatOK;
}
static OfxStatus doubleProperty(OfxPropertySetHandle, const char *, int, double *out) {
    *out = currentFrame; return kOfxStatOK;
}
static OfxStatus doubleProperties(OfxPropertySetHandle, const char *, int count, double *out) {
    for (int i = 0; i < count; ++i) out[i] = 1;
    return kOfxStatOK;
}
static OfxStatus intProperties(OfxPropertySetHandle, const char *, int count, int *out) {
    const int bounds[] = {0, 0, 2, 2};
    for (int i = 0; i < count; ++i) out[i] = bounds[i];
    return kOfxStatOK;
}
static OfxStatus intProperty(OfxPropertySetHandle, const char *, int, int *out) {
    *out = 2 * 4 * sizeof(float); return kOfxStatOK;
}
static OfxStatus stringProperty(OfxPropertySetHandle, const char *name, int, char **out) {
    *out = const_cast<char *>(strcmp(name, kOfxImageEffectPropPixelDepth) == 0
                             ? kOfxBitDepthFloat : kOfxImageComponentRGBA);
    return kOfxStatOK;
}
static OfxStatus parameter(OfxParamHandle handle, OfxTime time, ...) {
    if (!handle) return kOfxStatFailed;
    va_list args; va_start(args, time);
    if (handle == instance.pPlaybackOnly) *va_arg(args, int *) = playback ? 1 : 0;
    else if (handle == instance.pPlaybackToken) *va_arg(args, char **) = const_cast<char *>("batch");
    va_end(args);
    return kOfxStatOK;
}
static int isAborted(OfxImageEffectHandle) { return aborted; }
static OfxStatus fetch(OfxImageClipHandle clip, OfxTime, const OfxRectD *, OfxPropertySetHandle *out) {
    ++fetches;
    if (clip == instance.source) ++sourceFetches;
    const bool targeted = fetches == targetFetch;
    if (targeted && cancelOnFetch) aborted = true;
    *out = targeted && failFetch && !partial ? nullptr : reinterpret_cast<OfxPropertySetHandle>(clip);
    return targeted && failFetch ? kOfxStatFailed : kOfxStatOK;
}
static OfxStatus release(OfxPropertySetHandle) { ++releases; return kOfxStatOK; }
static OfxStatus message(void *, const char *, const char *, const char *format, ...) {
    ++messages;
    char text[2048]; va_list args; va_start(args, format);
    vsnprintf(text, sizeof(text), format, args); va_end(args);
    lastMessage = text; return kOfxStatOK;
}
static void require(bool pass, const char *what) {
    if (!pass) throw std::runtime_error(what);
}
static void scenario(int target, bool cancel, bool failure, bool partialHandle = false) {
    aborted = false; fetches = releases = messages = 0; lastMessage.clear();
    targetFetch = target; cancelOnFetch = cancel; failFetch = failure; partial = partialHandle;
    const auto result = render(instance.effect, nullptr);
    require(result == (cancel ? kOfxStatOK : kOfxStatFailed), "wrong render status");
    require(messages == (cancel ? 0 : 1), "cancelled fetch raised an error, or real error hidden");
    require(releases == target - 1 + ((!failure || partialHandle) ? 1 : 0), "image handle leaked");
    require(!instance.cacheValid, "cancelled/failed render entered cache");
    if (!cancel) require(lastMessage.find("frame 27") != std::string::npos, "missing frame diagnostics");
}
int main() {
    OfxImageEffectSuiteV1 effect{};
    effect.getPropertySet = properties; effect.abort = isAborted;
    effect.clipGetImage = fetch; effect.clipReleaseImage = release;
    OfxPropertySuiteV1 props{};
    props.propGetPointer = pointerProperty; props.propGetDouble = doubleProperty;
    props.propGetDoubleN = doubleProperties;
    props.propGetIntN = intProperties; props.propGetInt = intProperty;
    props.propGetString = stringProperty;
    OfxParameterSuiteV1 params{}; params.paramGetValueAtTime = parameter;
    OfxMessageSuiteV1 msg{}; msg.message = message;
    gEffect = &effect; gProp = &props; gParam = &params; gMessage = &msg;
    instance.effect = reinterpret_cast<OfxImageEffectHandle>(&instance);
    instance.source = reinterpret_cast<OfxImageClipHandle>(&pixels[0]);
    instance.output = reinterpret_cast<OfxImageClipHandle>(&pixels[8]);
    try {
        for (int target : {1, 2}) {
            scenario(target, true, true);
            scenario(target, true, true, true);
            scenario(target, true, false);
            scenario(target, false, true);
        }
        instance.pPlaybackOnly = reinterpret_cast<OfxParamHandle>(&pixels[0]);
        instance.pPlaybackToken = reinterpret_cast<OfxParamHandle>(&pixels[1]);
        playback = true; aborted = false; cancelOnFetch = false; failFetch = false;
        sourceFetches = fetches = releases = messages = 0; targetFetch = 99;
        StoredMask mask; mask.w = 2; mask.h = 2; mask.bits = {9};
        instance.stagedToken = "batch"; instance.stagedMasks[27] = mask;
        require(render(instance.effect, nullptr) == kOfxStatOK, "RAM playback failed");
        require(sourceFetches == 0 && fetches == 1 && releases == 1, "playback fetched source or leaked output");
        require(pixels[0] == 0 && pixels[4] == 1 && pixels[8] == 1 && pixels[12] == 0, "mask coordinates changed");
        currentFrame = 28;
        require(render(instance.effect, nullptr) == kOfxStatOK, "out-of-range playback failed");
        for (float pixel : pixels) require(pixel == 0, "missing RAM frame was not black");
        require(sourceFetches == 0 && messages == 0, "missing frame attempted inference or raised error");
        puts("PASS: cancelled source/output requests are silent, handles released, cache untouched; real failures reported");
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
    return 0;
}
