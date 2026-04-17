#version 450

layout(set = 0, binding = 0) uniform sampler2D uTexture;
layout(set = 0, binding = 1, rgba8) uniform readonly image2D u3dImage;

layout(set = 0, binding = 2, std430) readonly buffer TopPackedBuffer
{
    uint topPacked[];
};

layout(set = 0, binding = 3, std430) readonly buffer BottomPackedBuffer
{
    uint bottomPacked[];
};

layout(push_constant) uniform PresenterPushConstants
{
    uint drawMode;
    uint scale;
    uint rendererWidth;
    uint rendererHeight;
    uint packedStride;
    uint filtering;
} pushConstants;

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

uint readPacked(bool topScreen, int y, int x)
{
    uint offset = uint(y) * pushConstants.packedStride + uint(x);
    return topScreen ? topPacked[offset] : bottomPacked[offset];
}

vec3 color6ToRgb01(Rgba6 color)
{
    return vec3(
        float(toColor8(color.r)) * (1.0 / 255.0),
        float(toColor8(color.g)) * (1.0 / 255.0),
        float(toColor8(color.b)) * (1.0 / 255.0)
    );
}

Rgba6 samplePackedWithBrightness(
    bool topScreen,
    int sourceX,
    int sourceY,
    int displayMode,
    int brightnessMode,
    int brightnessFactor)
{
    Rgba6 pixel = unpackColor6(readPacked(topScreen, sourceY, sourceX));

    if (displayMode != 0)
    {
        if (brightnessMode == 1)
            applyBrightnessUp(pixel, brightnessFactor);
        else if (brightnessMode == 2)
            applyBrightnessDown(pixel, brightnessFactor, 0xF);
    }

    return pixel;
}

vec4 composeScreenColor(bool topScreen)
{
    float sourceXFloat = clamp(fragUv.x * 256.0, 0.0, 255.0);
    float sourceYFloat = clamp((1.0 - fragUv.y) * 192.0, 0.0, 191.0);
    float scaledXFloat = clamp(fragUv.x * float(pushConstants.rendererWidth), 0.0, float(pushConstants.rendererWidth - 1u));
    float scaledYFloat = clamp((1.0 - fragUv.y) * float(pushConstants.rendererHeight), 0.0, float(pushConstants.rendererHeight - 1u));
    int sourceX = int(sourceXFloat);
    int sourceY = int(sourceYFloat);

    uint masterBrightness = readPacked(topScreen, sourceY, 256 * 3);
    int displayMode = int((masterBrightness >> 16u) & 0x3u);
    int brightnessMode = int(((masterBrightness >> 8u) & 0xFFu) >> 6u);
    int brightnessFactor = min(16, int(masterBrightness & 0x1Fu));
    int xOffset = int((masterBrightness >> 24u) & 0xFFu)
        - ((((masterBrightness >> 16u) & 0x80u) != 0u) ? 256 : 0);

    Rgba6 pixel = unpackColor6(readPacked(topScreen, sourceY, sourceX));

    if (displayMode == 1)
    {
        Rgba6 val1 = pixel;
        Rgba6 val2 = unpackColor6(readPacked(topScreen, sourceY, 256 + sourceX));
        Rgba6 val3 = unpackColor6(readPacked(topScreen, sourceY, 512 + sourceX));

        int compMode = val3.a & 0xF;
        // Match the offscreen compositor contract: packed buffers stay DS-sized,
        // while the 3D sample comes from the true IR-scaled pixel position.
        Rgba6 pixel3D = sample3DColorAtScaledCoord(
            scaledXFloat + (float(xOffset) * float(max(pushConstants.scale, 1u))),
            scaledYFloat
        );

        if (compMode == 4)
        {
            if ((pixel3D.a & 0x1F) > 0)
            {
                int eva = (pixel3D.a & 0x1F) + 1;
                int evb = 32 - eva;
                val1.r = clampColor6(((pixel3D.r * eva) + (val1.r * evb) + 0x10) >> 5);
                val1.g = clampColor6(((pixel3D.g * eva) + (val1.g * evb) + 0x10) >> 5);
                val1.b = clampColor6(((pixel3D.b * eva) + (val1.b * evb) + 0x10) >> 5);
            }
            else
            {
                val1 = val2;
            }
        }
        else if (compMode == 1)
        {
            if ((pixel3D.a & 0x1F) > 0)
            {
                int eva = val3.g;
                int evb = val3.b;
                val1.r = clampColor6(((val1.r * eva) + (pixel3D.r * evb) + 0x8) >> 4);
                val1.g = clampColor6(((val1.g * eva) + (pixel3D.g * evb) + 0x8) >> 4);
                val1.b = clampColor6(((val1.b * eva) + (pixel3D.b * evb) + 0x8) >> 4);
            }
            else
            {
                val1 = val2;
            }
        }
        else if (compMode <= 3)
        {
            if ((pixel3D.a & 0x1F) > 0)
            {
                val1 = pixel3D;
                int evy = val3.g;
                if (compMode == 2)
                {
                    val1.r = clampColor6(val1.r + ((((63 - val1.r) * evy) + 0x8) >> 4));
                    val1.g = clampColor6(val1.g + ((((63 - val1.g) * evy) + 0x8) >> 4));
                    val1.b = clampColor6(val1.b + ((((63 - val1.b) * evy) + 0x8) >> 4));
                }
                else if (compMode == 3)
                {
                    applyBrightnessDown(val1, evy, 0x7);
                }
            }
            else
            {
                val1 = val2;
            }
        }

        pixel = val1;
    }

    if (displayMode == 1)
        return vec4(color6ToRgb01(pixel), fragAlpha);

    if (pushConstants.filtering != 0u)
    {
        float linearX = clamp(sourceXFloat - 0.5, 0.0, 255.0);
        float linearY = clamp(sourceYFloat - 0.5, 0.0, 191.0);
        int x0 = int(floor(linearX));
        int y0 = int(floor(linearY));
        int x1 = min(x0 + 1, 255);
        int y1 = min(y0 + 1, 191);
        float tx = linearX - float(x0);
        float ty = linearY - float(y0);

        vec3 c00 = color6ToRgb01(samplePackedWithBrightness(topScreen, x0, y0, displayMode, brightnessMode, brightnessFactor));
        vec3 c10 = color6ToRgb01(samplePackedWithBrightness(topScreen, x1, y0, displayMode, brightnessMode, brightnessFactor));
        vec3 c01 = color6ToRgb01(samplePackedWithBrightness(topScreen, x0, y1, displayMode, brightnessMode, brightnessFactor));
        vec3 c11 = color6ToRgb01(samplePackedWithBrightness(topScreen, x1, y1, displayMode, brightnessMode, brightnessFactor));
        vec3 cx0 = mix(c00, c10, tx);
        vec3 cx1 = mix(c01, c11, tx);
        vec3 finalColor = mix(cx0, cx1, ty);
        return vec4(finalColor, fragAlpha);
    }

    pixel = samplePackedWithBrightness(topScreen, sourceX, sourceY, displayMode, brightnessMode, brightnessFactor);
    return vec4(color6ToRgb01(pixel), fragAlpha);
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

    outColor = composeScreenColor(pushConstants.drawMode == 2u);
}
