#version 450

layout(set = 0, binding = 0) uniform sampler2D uTexture;
layout(set = 0, binding = 1, rgba8) uniform readonly image2D u3dImage;
layout(set = 0, binding = 4, rgba8) uniform readonly image2D u3dPreviousTopImage;
layout(set = 0, binding = 6, rgba8) uniform readonly image2D u3dPreviousBottomImage;

layout(set = 0, binding = 2, std430) readonly buffer TopPackedBuffer
{
    uint topPacked[];
};

layout(set = 0, binding = 3, std430) readonly buffer BottomPackedBuffer
{
    uint bottomPacked[];
};

layout(set = 0, binding = 5, std430) readonly buffer Capture3dBuffer
{
    uint capture3dPacked[];
};

layout(push_constant) uniform PresenterPushConstants
{
    uint drawMode;
    uint scale;
    uint rendererWidth;
    uint rendererHeight;
    uint packedStride;
    uint screenSwap;
    uint filtering;
    uint previousTopSourceValid;
    uint previousBottomSourceValid;
    uint captureSourceValid;
    float viewportWidth;
    float viewportHeight;
} pushConstants;

const uint kMetaFlagRegularCaptureUses3d = 1u << 21u;
const uint kMetaFlagVramCaptureUses3d = 1u << 22u;
const uint kMetaFlagForceLive3dCompMode7 = 1u << 18u;
const uint kFilterLinear = 1u;
const uint kFilterSharp2D = 2u;
const uint kFilterXbr2 = 3u;
const uint kFilterHq2x = 4u;
const uint kFilterHq4x = 5u;
const uint kFilterQuilez = 6u;
const uint kFilterLcd = 7u;
const uint kFilterLcdGridDsLite = 8u;
const uint kFilterScanlines = 9u;

layout(location = 0) in vec2 fragUv;
layout(location = 1) in float fragAlpha;

layout(location = 0) out vec4 outColor;

struct Rgba6
{
    int r;
    int g;
    int b;
    int a;
};

bool isPacked3dPlaceholder(Rgba6 color)
{
    return color.r == 0
        && color.g == 0
        && color.b == 0
        && color.a == 0x20;
}

bool hasPackedVisibleColor(Rgba6 color)
{
    return !isPacked3dPlaceholder(color)
        && ((color.r | color.g | color.b) != 0);
}

Rgba6 makeScreenWhite()
{
    Rgba6 color;
    color.r = 63;
    color.g = 63;
    color.b = 63;
    color.a = 0;
    return color;
}

int clampColor6(int value)
{
    return clamp(value, 0, 63);
}

int toColor8(int value6)
{
    int base = clampColor6(value6) << 2;
    return base | (base >> 6);
}

Rgba6 unpackColor6(uint packedColor)
{
    Rgba6 color;
    color.r = int(packedColor & 0xFFu);
    color.g = int((packedColor >> 8u) & 0xFFu);
    color.b = int((packedColor >> 16u) & 0xFFu);
    color.a = int((packedColor >> 24u) & 0xFFu);
    return color;
}

void applyBrightnessUp(inout Rgba6 color, int evy)
{
    color.r = clampColor6(color.r + (((63 - color.r) * evy) >> 4));
    color.g = clampColor6(color.g + (((63 - color.g) * evy) >> 4));
    color.b = clampColor6(color.b + (((63 - color.b) * evy) >> 4));
}

void applyBrightnessDown(inout Rgba6 color, int evy, int roundingBias)
{
    color.r = clampColor6(color.r - (((color.r * evy) + roundingBias) >> 4));
    color.g = clampColor6(color.g - (((color.g * evy) + roundingBias) >> 4));
    color.b = clampColor6(color.b - (((color.b * evy) + roundingBias) >> 4));
}

Rgba6 sample3DColorAtScaledCoord(float scaledX, float scaledY)
{
    Rgba6 zero;
    zero.r = 0;
    zero.g = 0;
    zero.b = 0;
    zero.a = 0;

    if (scaledX < 0.0
        || scaledX >= float(pushConstants.rendererWidth)
        || scaledY < 0.0
        || scaledY >= float(pushConstants.rendererHeight))
        return zero;

    vec4 color3d = imageLoad(u3dImage, ivec2(int(scaledX), int(scaledY)));

    Rgba6 color;
    color.r = int(clamp(color3d.r * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.g = int(clamp(color3d.g * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.b = int(clamp(color3d.b * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.a = int(clamp(color3d.a * 255.0 + 0.5, 0.0, 255.0)) >> 3;
    return color;
}

Rgba6 samplePreviousTop3DColorAtScaledCoord(float scaledX, float scaledY)
{
    Rgba6 zero;
    zero.r = 0;
    zero.g = 0;
    zero.b = 0;
    zero.a = 0;

    if (scaledX < 0.0
        || scaledX >= float(pushConstants.rendererWidth)
        || scaledY < 0.0
        || scaledY >= float(pushConstants.rendererHeight))
        return zero;

    vec4 color3d = imageLoad(u3dPreviousTopImage, ivec2(int(scaledX), int(scaledY)));

    Rgba6 color;
    color.r = int(clamp(color3d.r * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.g = int(clamp(color3d.g * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.b = int(clamp(color3d.b * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.a = int(clamp(color3d.a * 255.0 + 0.5, 0.0, 255.0)) >> 3;
    return color;
}

Rgba6 samplePreviousBottom3DColorAtScaledCoord(float scaledX, float scaledY)
{
    Rgba6 zero;
    zero.r = 0;
    zero.g = 0;
    zero.b = 0;
    zero.a = 0;

    if (scaledX < 0.0
        || scaledX >= float(pushConstants.rendererWidth)
        || scaledY < 0.0
        || scaledY >= float(pushConstants.rendererHeight))
        return zero;

    vec4 color3d = imageLoad(u3dPreviousBottomImage, ivec2(int(scaledX), int(scaledY)));

    Rgba6 color;
    color.r = int(clamp(color3d.r * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.g = int(clamp(color3d.g * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.b = int(clamp(color3d.b * 255.0 + 0.5, 0.0, 255.0)) >> 2;
    color.a = int(clamp(color3d.a * 255.0 + 0.5, 0.0, 255.0)) >> 3;
    return color;
}

Rgba6 sampleCapture3DColorAtDsPixel(int dsX, int dsY)
{
    Rgba6 zero;
    zero.r = 0;
    zero.g = 0;
    zero.b = 0;
    zero.a = 0;

    if (pushConstants.captureSourceValid == 0u
        || dsX < 0
        || dsX >= 256
        || dsY < 0
        || dsY >= 192)
        return zero;

    return unpackColor6(capture3dPacked[uint(dsY) * 256u + uint(dsX)]);
}

uint readTopPacked(int y, int x)
{
    uint offset = uint(y) * pushConstants.packedStride + uint(x);
    return topPacked[offset];
}

uint readBottomPacked(int y, int x)
{
    uint offset = uint(y) * pushConstants.packedStride + uint(x);
    return bottomPacked[offset];
}

vec3 color6ToRgb01(Rgba6 color)
{
    return vec3(
        float(toColor8(color.r)) * (1.0 / 255.0),
        float(toColor8(color.g)) * (1.0 / 255.0),
        float(toColor8(color.b)) * (1.0 / 255.0)
    );
}

vec3 color6ToVec3(Rgba6 color)
{
    return vec3(float(color.r), float(color.g), float(color.b));
}

Rgba6 vec3ToColor6(vec3 color, int alpha)
{
    Rgba6 result;
    result.r = clampColor6(int(floor(color.r + 0.5)));
    result.g = clampColor6(int(floor(color.g + 0.5)));
    result.b = clampColor6(int(floor(color.b + 0.5)));
    result.a = alpha;
    return result;
}

bool packedTapIsUnsafeFor2DFilter(Rgba6 color)
{
    return isPacked3dPlaceholder(color);
}

#define DEFINE_SAMPLE_PACKED_WITH_BRIGHTNESS(FUNC_NAME, READ_PACKED_FUNC) \
Rgba6 FUNC_NAME( \
    int sourceX, \
    int sourceY, \
    int displayMode, \
    int brightnessMode, \
    int brightnessFactor) \
{ \
    Rgba6 pixel = unpackColor6(READ_PACKED_FUNC(sourceY, sourceX)); \
 \
    if (displayMode != 0) \
    { \
        if (brightnessMode == 1) \
            applyBrightnessUp(pixel, brightnessFactor); \
        else if (brightnessMode == 2) \
            applyBrightnessDown(pixel, brightnessFactor, 0xF); \
    } \
 \
    return pixel; \
}

DEFINE_SAMPLE_PACKED_WITH_BRIGHTNESS(sampleTopPackedWithBrightness, readTopPacked)
DEFINE_SAMPLE_PACKED_WITH_BRIGHTNESS(sampleBottomPackedWithBrightness, readBottomPacked)

#define DEFINE_SAMPLE_FILTERED_PACKED_LAYER(FUNC_NAME, READ_PACKED_FUNC) \
Rgba6 FUNC_NAME(int sourceX, int sourceY, float sourceXFloat, float sourceYFloat, int layerOffset) \
{ \
    Rgba6 nearest = unpackColor6(READ_PACKED_FUNC(sourceY, layerOffset + sourceX)); \
    if (pushConstants.filtering != kFilterLinear && pushConstants.filtering != kFilterSharp2D) \
        return nearest; \
    float sharpX = clamp(sourceXFloat - 0.5, 0.0, 255.0); \
    float sharpY = clamp(sourceYFloat - 0.5, 0.0, 191.0); \
    int x0 = int(floor(sharpX)); \
    int y0 = int(floor(sharpY)); \
    int x1 = min(x0 + 1, 255); \
    int y1 = min(y0 + 1, 191); \
    float tx = sharpX - float(x0); \
    float ty = sharpY - float(y0); \
    Rgba6 c00 = unpackColor6(READ_PACKED_FUNC(y0, layerOffset + x0)); \
    Rgba6 c10 = unpackColor6(READ_PACKED_FUNC(y0, layerOffset + x1)); \
    Rgba6 c01 = unpackColor6(READ_PACKED_FUNC(y1, layerOffset + x0)); \
    Rgba6 c11 = unpackColor6(READ_PACKED_FUNC(y1, layerOffset + x1)); \
    if (packedTapIsUnsafeFor2DFilter(c00) \
        || packedTapIsUnsafeFor2DFilter(c10) \
        || packedTapIsUnsafeFor2DFilter(c01) \
        || packedTapIsUnsafeFor2DFilter(c11)) \
        return nearest; \
    vec3 v00 = color6ToVec3(c00); \
    vec3 v10 = color6ToVec3(c10); \
    vec3 v01 = color6ToVec3(c01); \
    vec3 v11 = color6ToVec3(c11); \
    vec3 blended = mix(mix(v00, v10, tx), mix(v01, v11, tx), ty); \
    if (pushConstants.filtering == kFilterLinear) \
        return vec3ToColor6(blended, nearest.a); \
    vec3 sharpened = mix(blended, color6ToVec3(nearest), 0.45); \
    vec3 minColor = min(min(v00, v10), min(v01, v11)); \
    vec3 maxColor = max(max(v00, v10), max(v01, v11)); \
    return vec3ToColor6(clamp(sharpened, minColor, maxColor), nearest.a); \
}

DEFINE_SAMPLE_FILTERED_PACKED_LAYER(sampleTopFilteredPackedLayer, readTopPacked)
DEFINE_SAMPLE_FILTERED_PACKED_LAYER(sampleBottomFilteredPackedLayer, readBottomPacked)

#undef DEFINE_SAMPLE_FILTERED_PACKED_LAYER

#define DEFINE_COMPOSE_SCREEN_COLOR(FUNC_NAME, READ_PACKED_FUNC, SAMPLE_PACKED_FUNC, SCREEN_IS_TOP) \
vec4 FUNC_NAME() \
{ \
    float sourceXFloat = clamp(fragUv.x * 256.0, 0.0, 255.0); \
    float sourceYFloat = clamp((1.0 - fragUv.y) * 192.0, 0.0, 191.0); \
    float scaledXFloat = clamp(fragUv.x * float(pushConstants.rendererWidth), 0.0, float(pushConstants.rendererWidth - 1u)); \
    float scaledYFloat = clamp((1.0 - fragUv.y) * float(pushConstants.rendererHeight), 0.0, float(pushConstants.rendererHeight - 1u)); \
    int sourceX = int(sourceXFloat); \
    int sourceY = int(sourceYFloat); \
    const bool packedTopScreen = SCREEN_IS_TOP; \
 \
    uint masterBrightness = READ_PACKED_FUNC(sourceY, 256 * 3); \
    int displayMode = int((masterBrightness >> 16u) & 0x3u); \
    int brightnessMode = int(((masterBrightness >> 8u) & 0xFFu) >> 6u); \
    int brightnessFactor = min(16, int(masterBrightness & 0x1Fu)); \
    int xOffset = int((masterBrightness >> 24u) & 0xFFu) \
        - ((((masterBrightness >> 16u) & 0x80u) != 0u) ? 256 : 0); \
    bool regularCaptureUses3d = (masterBrightness & kMetaFlagRegularCaptureUses3d) != 0u; \
    bool vramCaptureUses3d = (masterBrightness & kMetaFlagVramCaptureUses3d) != 0u; \
    bool forceLive3dCompMode7 = (masterBrightness & kMetaFlagForceLive3dCompMode7) != 0u; \
    bool screenOwnsLive3D = SCREEN_IS_TOP ? (pushConstants.screenSwap != 0u) : (pushConstants.screenSwap == 0u); \
\
	    Rgba6 pixel = SCREEN_IS_TOP \
            ? sampleTopFilteredPackedLayer(sourceX, sourceY, sourceXFloat, sourceYFloat, 0) \
            : sampleBottomFilteredPackedLayer(sourceX, sourceY, sourceXFloat, sourceYFloat, 0); \
\
	    if (displayMode == 1) \
	    { \
	        Rgba6 val1 = pixel; \
	        Rgba6 val2 = SCREEN_IS_TOP \
                ? sampleTopFilteredPackedLayer(sourceX, sourceY, sourceXFloat, sourceYFloat, 256) \
                : sampleBottomFilteredPackedLayer(sourceX, sourceY, sourceXFloat, sourceYFloat, 256); \
	        Rgba6 val3 = unpackColor6(READ_PACKED_FUNC(sourceY, 512 + sourceX)); \
\
        int compMode = val3.a & 0xF; \
        bool both3dPlaceholders = isPacked3dPlaceholder(val1) && isPacked3dPlaceholder(val2); \
        bool captureBackedComp4 = compMode == 4 && both3dPlaceholders; \
        bool temporalCompMode7Uses3D = compMode == 7 && (regularCaptureUses3d || forceLive3dCompMode7); \
        bool compModeSamples3D = compMode <= 4 || temporalCompMode7Uses3D; \
        bool screenHasPrevious3D = SCREEN_IS_TOP ? (pushConstants.previousTopSourceValid != 0u) : (pushConstants.previousBottomSourceValid != 0u); \
        Rgba6 pixel3D; \
        pixel3D.r = 0; \
        pixel3D.g = 0; \
        pixel3D.b = 0; \
        pixel3D.a = 0; \
        if (compModeSamples3D && screenOwnsLive3D) \
        { \
            pixel3D = sample3DColorAtScaledCoord( \
                scaledXFloat + (float(xOffset) * float(max(pushConstants.scale, 1u))), \
                scaledYFloat \
            ); \
        } \
        else if (compModeSamples3D && screenHasPrevious3D) \
        { \
            pixel3D = SCREEN_IS_TOP \
                ? samplePreviousTop3DColorAtScaledCoord( \
                    scaledXFloat + (float(xOffset) * float(max(pushConstants.scale, 1u))), \
                    scaledYFloat \
                ) \
                : samplePreviousBottom3DColorAtScaledCoord( \
                    scaledXFloat + (float(xOffset) * float(max(pushConstants.scale, 1u))), \
                    scaledYFloat \
                ); \
        } \
        Rgba6 capture3D = sampleCapture3DColorAtDsPixel(sourceX, sourceY); \
        bool pixel3DHasUsefulColor = ((pixel3D.a & 0x1F) > 0) || hasPackedVisibleColor(pixel3D); \
        bool captureBackedComp4Valid = captureBackedComp4 && ((capture3D.a & 0x1F) > 0); \
 \
        if (compMode == 4 && both3dPlaceholders && pixel3DHasUsefulColor) \
        { \
            val1 = pixel3D; \
        } \
        else if (compMode == 4 && both3dPlaceholders) \
        { \
            if (captureBackedComp4Valid) \
            { \
                pixel3D = capture3D; \
            } \
        } \
        else if ((pixel3D.a & 0x1F) == 0 && captureBackedComp4Valid) \
        { \
            pixel3D = capture3D; \
        } \
 \
        if (compMode == 4) \
        { \
            if (both3dPlaceholders && pixel3DHasUsefulColor) \
            { \
                val1 = pixel3D; \
            } \
            else if (both3dPlaceholders && captureBackedComp4Valid) \
            { \
                val1 = capture3D; \
            } \
            else if (both3dPlaceholders && brightnessFactor > 0) \
            { \
                val1 = val2; \
            } \
            else if ((pixel3D.a & 0x1F) > 0) \
            { \
                int eva = (pixel3D.a & 0x1F) + 1; \
                int evb = 32 - eva; \
                val1.r = clampColor6(((pixel3D.r * eva) + (val1.r * evb) + 0x10) >> 5); \
                val1.g = clampColor6(((pixel3D.g * eva) + (val1.g * evb) + 0x10) >> 5); \
                val1.b = clampColor6(((pixel3D.b * eva) + (val1.b * evb) + 0x10) >> 5); \
            } \
            else \
            { \
                val1 = val2; \
            } \
        } \
        else if (compMode == 1) \
        { \
            if ((pixel3D.a & 0x1F) > 0) \
            { \
                int eva = val3.g; \
                int evb = val3.b; \
                val1.r = clampColor6(((val1.r * eva) + (pixel3D.r * evb) + 0x8) >> 4); \
                val1.g = clampColor6(((val1.g * eva) + (pixel3D.g * evb) + 0x8) >> 4); \
                val1.b = clampColor6(((val1.b * eva) + (pixel3D.b * evb) + 0x8) >> 4); \
            } \
            else \
            { \
                val1 = val2; \
            } \
        } \
        else if (compMode <= 3) \
        { \
            if ((pixel3D.a & 0x1F) > 0) \
            { \
                val1 = pixel3D; \
                int evy = val3.g; \
                if (compMode == 2) \
                { \
                    val1.r = clampColor6(val1.r + ((((63 - val1.r) * evy) + 0x8) >> 4)); \
                    val1.g = clampColor6(val1.g + ((((63 - val1.g) * evy) + 0x8) >> 4)); \
                    val1.b = clampColor6(val1.b + ((((63 - val1.b) * evy) + 0x8) >> 4)); \
                } \
                else if (compMode == 3) \
                { \
                    applyBrightnessDown(val1, evy, 0x7); \
                } \
            } \
            else \
            { \
                val1 = val2; \
            } \
        } \
        else if (compMode == 7) \
        { \
            if (temporalCompMode7Uses3D && (pixel3D.a & 0x1F) > 0) \
                val1 = pixel3D; \
        } \
        pixel = val1; \
    } \
 \
    if (displayMode != 0) \
    { \
        if (brightnessMode == 1) \
            applyBrightnessUp(pixel, brightnessFactor); \
        else if (brightnessMode == 2) \
            applyBrightnessDown(pixel, brightnessFactor, 0xF); \
    } \
 \
    if (displayMode == 1 || (displayMode == 2 && vramCaptureUses3d)) \
        return vec4(color6ToRgb01(pixel), fragAlpha); \
 \
	    if (pushConstants.filtering == kFilterLinear) \
    { \
        float linearX = clamp(sourceXFloat - 0.5, 0.0, 255.0); \
        float linearY = clamp(sourceYFloat - 0.5, 0.0, 191.0); \
        int x0 = int(floor(linearX)); \
        int y0 = int(floor(linearY)); \
        int x1 = min(x0 + 1, 255); \
        int y1 = min(y0 + 1, 191); \
        float tx = linearX - float(x0); \
        float ty = linearY - float(y0); \
 \
        vec3 c00 = color6ToRgb01(SAMPLE_PACKED_FUNC(x0, y0, displayMode, brightnessMode, brightnessFactor)); \
        vec3 c10 = color6ToRgb01(SAMPLE_PACKED_FUNC(x1, y0, displayMode, brightnessMode, brightnessFactor)); \
        vec3 c01 = color6ToRgb01(SAMPLE_PACKED_FUNC(x0, y1, displayMode, brightnessMode, brightnessFactor)); \
        vec3 c11 = color6ToRgb01(SAMPLE_PACKED_FUNC(x1, y1, displayMode, brightnessMode, brightnessFactor)); \
        vec3 cx0 = mix(c00, c10, tx); \
        vec3 cx1 = mix(c01, c11, tx); \
        vec3 finalColor = mix(cx0, cx1, ty); \
        return vec4(finalColor, fragAlpha); \
    } \
 \
    pixel = SAMPLE_PACKED_FUNC(sourceX, sourceY, displayMode, brightnessMode, brightnessFactor); \
    return vec4(color6ToRgb01(pixel), fragAlpha); \
}

DEFINE_COMPOSE_SCREEN_COLOR(composeTopScreenColor, readTopPacked, sampleTopPackedWithBrightness, true)
DEFINE_COMPOSE_SCREEN_COLOR(composeBottomScreenColor, readBottomPacked, sampleBottomPackedWithBrightness, false)

vec2 compositeTexelSize()
{
    uint safeScale = max(pushConstants.scale, 1u);
    return vec2(
        1.0 / float(256u * safeScale),
        1.0 / float((384u + 2u) * safeScale)
    );
}

vec2 compositeDsTexelSize()
{
    return compositeTexelSize() * float(max(pushConstants.scale, 1u));
}

vec4 screenUvBounds(bool topScreen)
{
    float gapUv = 1.0 / 386.0;
    float minY = topScreen ? 0.0 : (0.5 + gapUv);
    float maxY = topScreen ? (0.5 - gapUv) : 1.0;
    return vec4(0.0, minY, 1.0, maxY);
}

vec2 clampCompositeUvToScreen(vec2 uv, bool topScreen)
{
    vec2 texel = compositeTexelSize();
    vec4 bounds = screenUvBounds(topScreen);
    return vec2(
        clamp(uv.x, texel.x * 0.5, 1.0 - texel.x * 0.5),
        clamp(uv.y, bounds.y + (texel.y * 0.5), bounds.w - (texel.y * 0.5))
    );
}

vec3 sampleCompositeRgb(vec2 uv, bool topScreen)
{
    return texture(uTexture, clampCompositeUvToScreen(uv, topScreen)).bgr;
}

vec2 screenLocalCoord(vec2 uv, bool topScreen)
{
    vec4 bounds = screenUvBounds(topScreen);
    vec2 clamped = clampCompositeUvToScreen(uv, topScreen);
    return vec2(clamped.x, clamp((clamped.y - bounds.y) / max(bounds.w - bounds.y, 0.0001), 0.0, 1.0));
}

vec2 uvFromScreenLocal(vec2 local, bool topScreen)
{
    vec4 bounds = screenUvBounds(topScreen);
    return vec2(clamp(local.x, 0.0, 1.0), mix(bounds.y, bounds.w, clamp(local.y, 0.0, 1.0)));
}

vec2 uvFromScreenTexel(vec2 texelCoord, bool topScreen)
{
    return uvFromScreenLocal((texelCoord + vec2(0.5)) / vec2(256.0, 192.0), topScreen);
}

vec3 filterQuilez(vec2 uv, bool topScreen)
{
    vec2 size = vec2(256.0, 192.0);
    vec2 local = screenLocalCoord(uv, topScreen);
    vec2 p = (local * size) + vec2(0.5);
    vec2 i = floor(p);
    vec2 f = p - i;
    f = f * f * f * ((f * ((f * 6.0) - vec2(15.0))) + vec2(10.0));
    vec2 filteredLocal = (i + f - vec2(0.5)) / size;
    return sampleCompositeRgb(uvFromScreenLocal(filteredLocal, topScreen), topScreen);
}

vec3 filterScanlines(vec2 uv, bool topScreen)
{
    vec3 color = sampleCompositeRgb(uv, topScreen);
    vec2 local = screenLocalCoord(uv, topScreen);
    vec2 omega = vec2(3.1415 * 256.0, 2.0 * 3.1415 * 192.0);
    const float baseBrightness = 0.95;
    const vec2 sineComp = vec2(0.05, 0.15);
    return clamp(color * (baseBrightness + dot(sineComp * sin(local * omega), vec2(1.0))), 0.0, 1.0);
}

vec3 filterLcd(vec2 uv, bool topScreen)
{
    vec3 color = sampleCompositeRgb(uv, topScreen);
    vec2 local = screenLocalCoord(uv, topScreen);
    vec2 angle = local * (3.141592654 * 2.0 * vec2(256.0, 192.0));
    const float brightenScanlines = 16.0;
    const float brightenLcd = 4.0;
    const vec3 offsets = 3.141592654 * vec3(0.5, 0.5 - (2.0 / 3.0), 0.5 - (4.0 / 3.0));
    float yFactor = (brightenScanlines + sin(angle.y)) / (brightenScanlines + 1.0);
    vec3 xFactors = (brightenLcd + sin(angle.x + offsets)) / (brightenLcd + 1.0);
    return clamp(color * yFactor * xFactors, 0.0, 1.0);
}

float intsmearFuncX(float z)
{
    float z2 = z * z;
    float zn = z;
    float ret = 0.0;
    ret += zn * 1.0;
    zn *= z2;
    ret += zn * -0.6666667;
    zn *= z2;
    ret += zn * -0.2;
    zn *= z2;
    ret += zn * 0.5714286;
    zn *= z2;
    ret += zn * -0.1111111;
    zn *= z2;
    ret += zn * -0.1818182;
    zn *= z2;
    ret += zn * 0.0769231;
    return ret;
}

float intsmearFuncY(float z)
{
    float z2 = z * z;
    float zn = z;
    float ret = 0.0;
    ret += zn * 1.0;
    zn *= z2;
    ret += zn * 0.0;
    zn *= z2;
    ret += zn * -0.8;
    zn *= z2;
    ret += zn * 0.2857143;
    zn *= z2;
    ret += zn * 0.4444444;
    zn *= z2;
    ret += zn * -0.3636364;
    zn *= z2;
    ret += zn * 0.0769231;
    return ret;
}

float intsmearX(float x, float dx, float d)
{
    float safeDx = max(dx, 0.000001);
    float zl = clamp((x - safeDx * 0.5) / d, -1.0, 1.0);
    float zh = clamp((x + safeDx * 0.5) / d, -1.0, 1.0);
    return d * (intsmearFuncX(zh) - intsmearFuncX(zl)) / safeDx;
}

float intsmearY(float x, float dx, float d)
{
    float safeDx = max(dx, 0.000001);
    float zl = clamp((x - safeDx * 0.5) / d, -1.0, 1.0);
    float zh = clamp((x + safeDx * 0.5) / d, -1.0, 1.0);
    return d * (intsmearFuncY(zh) - intsmearFuncY(zl)) / safeDx;
}

vec3 filterLcdGridDsLite(vec2 uv, bool topScreen)
{
    const float brightenScanlines = 16.0;
    const float brightenLcd = 4.0;
    const vec3 offsets = 3.141592654 * vec3(0.5, 0.5 - 0.6666667, 0.5 - 1.3333333);
    const float gain = 1.0;
    const float gamma = 2.2;
    const float blacklevel = 0.0;
    const float ambient = 0.0;
    const float outgamma = 2.2;
    const float maskContrast = 0.8;
    const float detailBlend = 0.3;
    const vec3 postBias = vec3(0.0);
    const vec2 lowerScreenPhase = vec2(0.15, 0.10);
    const vec3 channelGain = vec3(1.07, 0.97, 1.05);
    const float saturationBoost = 1.12;
    const vec3 rSubpixel = vec3(1.0, 0.0, 0.0);
    const vec3 gSubpixel = vec3(0.0, 1.0, 0.0);
    const vec3 bSubpixel = vec3(0.0, 0.0, 1.0);
    const float targetGamma = 2.2;
    const float displayGamma = 2.2;
    const float dslLuminance = 0.955;
    const mat3 dslMatrix = mat3(
        0.965, 0.11, -0.065,
        0.02, 0.925, 0.055,
        0.01, -0.02, 1.03
    );

    vec2 local = screenLocalCoord(uv, topScreen);
    vec2 screenPixelCoord = local * vec2(256.0, 192.0);
    vec2 pixelCoord = screenPixelCoord - vec2(0.4999);
    ivec2 tli = ivec2(floor(pixelCoord));
    float subpix = (pixelCoord.x - float(tli.x)) * 3.0;
    vec2 viewport = max(vec2(pushConstants.viewportWidth, pushConstants.viewportHeight), vec2(1.0));
    float rsubpix = (256.0 / viewport.x) * 3.0;
    vec3 lcol = vec3(
        intsmearX(subpix + 1.0, rsubpix, 1.5),
        intsmearX(subpix, rsubpix, 1.5),
        intsmearX(subpix - 1.0, rsubpix, 1.5)
    ).bgr;
    vec3 rcol = vec3(
        intsmearX(subpix - 2.0, rsubpix, 1.5),
        intsmearX(subpix - 3.0, rsubpix, 1.5),
        intsmearX(subpix - 4.0, rsubpix, 1.5)
    ).bgr;
    float subpixY = pixelCoord.y - float(tli.y);
    float rsubpixY = 192.0 / viewport.y;
    float tcol = intsmearY(subpixY, rsubpixY, 0.63);
    float bcol = intsmearY(subpixY - 1.0, rsubpixY, 0.63);

    vec2 baseTexel = vec2(tli);
    vec3 topLeftColor = pow(gain * sampleCompositeRgb(uvFromScreenTexel(clamp(baseTexel + vec2(0.0, 0.0), vec2(0.0), vec2(255.0, 191.0)), topScreen), topScreen) + vec3(blacklevel), vec3(gamma)) + vec3(ambient);
    vec3 bottomRightColor = pow(gain * sampleCompositeRgb(uvFromScreenTexel(clamp(baseTexel + vec2(1.0, 1.0), vec2(0.0), vec2(255.0, 191.0)), topScreen), topScreen) + vec3(blacklevel), vec3(gamma)) + vec3(ambient);
    vec3 bottomLeftColor = pow(gain * sampleCompositeRgb(uvFromScreenTexel(clamp(baseTexel + vec2(0.0, 1.0), vec2(0.0), vec2(255.0, 191.0)), topScreen), topScreen) + vec3(blacklevel), vec3(gamma)) + vec3(ambient);
    vec3 topRightColor = pow(gain * sampleCompositeRgb(uvFromScreenTexel(clamp(baseTexel + vec2(1.0, 0.0), vec2(0.0), vec2(255.0, 191.0)), topScreen), topScreen) + vec3(blacklevel), vec3(gamma)) + vec3(ambient);
    vec3 averageColor = topLeftColor * lcol * vec3(tcol)
        + bottomRightColor * rcol * vec3(bcol)
        + bottomLeftColor * lcol * vec3(bcol)
        + topRightColor * rcol * vec3(tcol);

    vec2 angle = screenPixelCoord * 6.28318530718;
    angle += (topScreen ? 0.0 : 1.0) * lowerScreenPhase;
    float yfactor = (brightenScanlines + sin(angle.y)) / (brightenScanlines + 1.0);
    vec3 xfactors = (brightenLcd + sin(angle.x + offsets)) / (brightenLcd + 1.0);
    vec3 mask = yfactor * xfactors;
    vec3 softenedMask = mix(vec3(1.0), mask, maskContrast);
    vec3 maskedColor = averageColor * softenedMask;
    averageColor = mix(maskedColor, averageColor, detailBlend);
    averageColor = mat3(pow(rSubpixel, vec3(outgamma)), pow(gSubpixel, vec3(outgamma)), pow(bSubpixel, vec3(outgamma))) * averageColor;
    vec3 baseColor = pow(max(averageColor, vec3(0.0)), vec3(1.0 / outgamma));
    vec3 dslLinear = pow(baseColor, vec3(targetGamma));
    dslLinear = clamp(dslLinear * dslLuminance, 0.0, 1.0);
    vec3 corrected = dslMatrix * dslLinear;
    corrected *= channelGain;
    float gray = dot(corrected, vec3(0.299, 0.587, 0.114));
    corrected = mix(vec3(gray), corrected, saturationBoost);
    vec3 finalColor = pow(max(corrected, vec3(0.0)), vec3(1.0 / displayGamma));
    return clamp(finalColor + postBias, 0.0, 1.0);
}

vec3 filterXbr2(vec2 uv, bool topScreen)
{
    vec2 texel = compositeDsTexelSize();
    vec2 local = screenLocalCoord(uv, topScreen);
    vec2 fp = fract(local * vec2(256.0, 192.0));
    vec2 g1 = vec2(0.0, -texel.y) * (step(0.5, fp.x) + step(0.5, fp.y) - 1.0)
        + vec2(-texel.x, 0.0) * (step(0.5, fp.x) - step(0.5, fp.y));
    vec2 g2 = vec2(0.0, -texel.y) * (step(0.5, fp.y) - step(0.5, fp.x))
        + vec2(-texel.x, 0.0) * (step(0.5, fp.x) + step(0.5, fp.y) - 1.0);
    vec3 c = sampleCompositeRgb(uv + g1 - g2, topScreen);
    vec3 e = sampleCompositeRgb(uv, topScreen);
    vec3 f = sampleCompositeRgb(uv - g2, topScreen);
    vec3 h = sampleCompositeRgb(uv - g1, topScreen);
    vec3 i = sampleCompositeRgb(uv - g1 - g2, topScreen);
    float de = length(e - f) + length(e - h);
    float edge = step(0.015, de) * step(length(h - f), 0.015) * step(0.015, length(h - e));
    vec3 blended = mix(e, mix(f, h, 0.5), 0.5);
    return mix(e, blended, edge * step(length(e - c), length(e - i) + 0.0001));
}

vec3 filterHq2x(vec2 uv, bool topScreen)
{
    vec2 dg1 = 0.5 * compositeDsTexelSize();
    vec2 dg2 = vec2(-dg1.x, dg1.y);
    vec2 dx = vec2(dg1.x, 0.0);
    vec2 dy = vec2(0.0, dg1.y);
    vec3 c00 = sampleCompositeRgb(uv - dg1, topScreen);
    vec3 c10 = sampleCompositeRgb(uv - dy, topScreen);
    vec3 c20 = sampleCompositeRgb(uv - dg2, topScreen);
    vec3 c01 = sampleCompositeRgb(uv - dx, topScreen);
    vec3 c11 = sampleCompositeRgb(uv, topScreen);
    vec3 c21 = sampleCompositeRgb(uv + dx, topScreen);
    vec3 c02 = sampleCompositeRgb(uv + dg2, topScreen);
    vec3 c12 = sampleCompositeRgb(uv + dy, topScreen);
    vec3 c22 = sampleCompositeRgb(uv + dg1, topScreen);
    vec3 dt = vec3(1.0);
    const float mx = 0.325;
    const float k = -0.250;
    const float maxW = 0.25;
    const float minW = -0.05;
    const float lumAdd = 0.25;
    float md1 = dot(abs(c00 - c22), dt);
    float md2 = dot(abs(c02 - c20), dt);
    float w1 = dot(abs(c22 - c11), dt) * md2;
    float w2 = dot(abs(c02 - c11), dt) * md1;
    float w3 = dot(abs(c00 - c11), dt) * md2;
    float w4 = dot(abs(c20 - c11), dt) * md1;
    float t1 = w1 + w3;
    float t2 = w2 + w4;
    float ww = max(t1, t2) + 0.001;
    c11 = (w1 * c00 + w2 * c20 + w3 * c22 + w4 * c02 + ww * c11) / (t1 + t2 + ww);
    float lc1 = k / (0.12 * dot(c10 + c12 + c11, dt) + lumAdd);
    float lc2 = k / (0.12 * dot(c01 + c21 + c11, dt) + lumAdd);
    w1 = clamp(lc1 * dot(abs(c11 - c10), dt) + mx, minW, maxW);
    w2 = clamp(lc2 * dot(abs(c11 - c21), dt) + mx, minW, maxW);
    w3 = clamp(lc1 * dot(abs(c11 - c12), dt) + mx, minW, maxW);
    w4 = clamp(lc2 * dot(abs(c11 - c01), dt) + mx, minW, maxW);
    return clamp(w1 * c10 + w2 * c21 + w3 * c12 + w4 * c01 + (1.0 - w1 - w2 - w3 - w4) * c11, 0.0, 1.0);
}

vec3 filterHq4x(vec2 uv, bool topScreen)
{
    vec2 dg1 = 0.5 * compositeDsTexelSize();
    vec2 dg2 = vec2(-dg1.x, dg1.y);
    vec2 sd1 = dg1 * 0.5;
    vec2 sd2 = dg2 * 0.5;
    vec2 ddx = vec2(dg1.x, 0.0);
    vec2 ddy = vec2(0.0, dg1.y);
    vec3 c = sampleCompositeRgb(uv, topScreen);
    vec3 i1 = sampleCompositeRgb(uv - sd1, topScreen);
    vec3 i2 = sampleCompositeRgb(uv - sd2, topScreen);
    vec3 i3 = sampleCompositeRgb(uv + sd1, topScreen);
    vec3 i4 = sampleCompositeRgb(uv + sd2, topScreen);
    vec3 o1 = sampleCompositeRgb(uv - dg1, topScreen);
    vec3 o3 = sampleCompositeRgb(uv + dg1, topScreen);
    vec3 o2 = sampleCompositeRgb(uv - dg2, topScreen);
    vec3 o4 = sampleCompositeRgb(uv + dg2, topScreen);
    vec3 s1 = sampleCompositeRgb(uv - ddy, topScreen);
    vec3 s2 = sampleCompositeRgb(uv + ddx, topScreen);
    vec3 s3 = sampleCompositeRgb(uv + ddy, topScreen);
    vec3 s4 = sampleCompositeRgb(uv - ddx, topScreen);
    vec3 dt = vec3(1.0);
    const float mx = 1.00;
    const float k = -1.10;
    const float maxW = 0.75;
    const float minW = 0.03;
    const float lumAdd = 0.33;
    float ko1 = dot(abs(o1 - c), dt);
    float ko2 = dot(abs(o2 - c), dt);
    float ko3 = dot(abs(o3 - c), dt);
    float ko4 = dot(abs(o4 - c), dt);
    float k1 = min(dot(abs(i1 - i3), dt), max(ko1, ko3));
    float k2 = min(dot(abs(i2 - i4), dt), max(ko2, ko4));
    float w1 = k2;
    if (ko3 < ko1)
        w1 *= ko3 / max(ko1, 0.000001);
    float w2 = k1;
    if (ko4 < ko2)
        w2 *= ko4 / max(ko2, 0.000001);
    float w3 = k2;
    if (ko1 < ko3)
        w3 *= ko1 / max(ko3, 0.000001);
    float w4 = k1;
    if (ko2 < ko4)
        w4 *= ko2 / max(ko4, 0.000001);
    c = (w1 * o1 + w2 * o2 + w3 * o3 + w4 * o4 + 0.001 * c) / (w1 + w2 + w3 + w4 + 0.001);
    w1 = k * dot(abs(i1 - c) + abs(i3 - c), dt) / (0.125 * dot(i1 + i3, dt) + lumAdd);
    w2 = k * dot(abs(i2 - c) + abs(i4 - c), dt) / (0.125 * dot(i2 + i4, dt) + lumAdd);
    w3 = k * dot(abs(s1 - c) + abs(s3 - c), dt) / (0.125 * dot(s1 + s3, dt) + lumAdd);
    w4 = k * dot(abs(s2 - c) + abs(s4 - c), dt) / (0.125 * dot(s2 + s4, dt) + lumAdd);
    w1 = clamp(w1 + mx, minW, maxW);
    w2 = clamp(w2 + mx, minW, maxW);
    w3 = clamp(w3 + mx, minW, maxW);
    w4 = clamp(w4 + mx, minW, maxW);
    return clamp((w1 * (i1 + i3) + w2 * (i2 + i4) + w3 * (s1 + s3) + w4 * (s2 + s4) + c) / (2.0 * (w1 + w2 + w3 + w4) + 1.0), 0.0, 1.0);
}

vec3 applyCompositePostFilter(vec2 uv, bool topScreen)
{
    if (pushConstants.filtering == kFilterQuilez)
        return filterQuilez(uv, topScreen);
    if (pushConstants.filtering == kFilterXbr2)
        return filterXbr2(uv, topScreen);
    if (pushConstants.filtering == kFilterHq2x)
        return filterHq2x(uv, topScreen);
    if (pushConstants.filtering == kFilterHq4x)
        return filterHq4x(uv, topScreen);
    if (pushConstants.filtering == kFilterLcd)
        return filterLcd(uv, topScreen);
    if (pushConstants.filtering == kFilterLcdGridDsLite)
        return filterLcdGridDsLite(uv, topScreen);
    if (pushConstants.filtering == kFilterScanlines)
        return filterScanlines(uv, topScreen);
    return sampleCompositeRgb(uv, topScreen);
}

void main()
{
    if (pushConstants.drawMode == 0u)
    {
        vec4 sampledColor = texture(uTexture, fragUv);
        outColor = vec4(sampledColor.rgb, fragAlpha);
        return;
    }

    if (pushConstants.drawMode == 1u)
    {
        vec4 sampledColor = texture(uTexture, fragUv);
        outColor = vec4(sampledColor.bgr, fragAlpha);
        return;
    }

    if (pushConstants.drawMode == 4u || pushConstants.drawMode == 5u)
    {
        bool topScreen = pushConstants.drawMode == 4u;
        outColor = vec4(applyCompositePostFilter(fragUv, topScreen), fragAlpha);
        return;
    }

    if (pushConstants.drawMode == 2u)
    {
        outColor = composeTopScreenColor();
        return;
    }

    outColor = composeBottomScreenColor();
}
