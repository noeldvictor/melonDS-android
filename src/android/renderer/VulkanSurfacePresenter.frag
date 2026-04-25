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
} pushConstants;

const uint kMetaFlagRegularCaptureUses3d = 1u << 21u;
const uint kMetaFlagVramCaptureUses3d = 1u << 22u;
const uint kMetaFlagForceLive3dCompMode7 = 1u << 18u;

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
    Rgba6 pixel = unpackColor6(READ_PACKED_FUNC(sourceY, sourceX)); \
\
    if (displayMode == 1) \
    { \
        Rgba6 val1 = pixel; \
        Rgba6 val2 = unpackColor6(READ_PACKED_FUNC(sourceY, 256 + sourceX)); \
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
    if (pushConstants.filtering != 0u) \
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

    if (pushConstants.drawMode == 2u)
    {
        outColor = composeTopScreenColor();
        return;
    }

    outColor = composeBottomScreenColor();
}
