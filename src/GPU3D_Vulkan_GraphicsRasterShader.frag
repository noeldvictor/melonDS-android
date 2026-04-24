#version 450

const uint MAX_TEXTURE_DESCRIPTORS = 128u;

layout(constant_id = 0) const uint DEPTH_INTERPOLATION_MODE = 0u;
layout(constant_id = 1) const uint TRANSLUCENT_PASS = 0u;
layout(constant_id = 2) const uint EDGE_MARK_PASS = 0u;

layout(set = 0, binding = 1) uniform usampler2DArray texArrays[MAX_TEXTURE_DESCRIPTORS];

layout(set = 0, binding = 2, std430) readonly buffer ToonTableBuffer
{
    uint toonValues[];
};

layout(push_constant) uniform PushConsts
{
    uint width;
    uint height;
    uint clearColor;
    uint clearDepth;
    uint triangleCount;
    uint dispCnt;
    uint alphaRef;
    uint fogColor;
    uint fogOffset;
    uint fogShift;
    uint clearAttr;
    uint fogDensityPacked[9];
    uint edgeColorPacked[8];
    uint variantKey;
    uint passIndex;
    uint triangleBase;
    uint depthBlendMode;
} pc;

layout(location = 0) smooth in vec3 fColor;
layout(location = 1) smooth in vec2 fTexcoord;
layout(location = 2) noperspective in float fDepthLinear;
layout(location = 3) smooth in float fDepthPerspective;
layout(location = 4) flat in uvec4 fTriInfo0;
layout(location = 5) flat in uvec4 fTriInfo1;

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oAttr;
layout(location = 2) out float oDepthValue;

const uint TRI_FLAG_TEXTURED = 1u << 1u;
const uint TRI_FLAG_DECAL = 1u << 2u;

struct Color6A5
{
    int r;
    int g;
    int b;
    int a;
};

int clamp6(int value)
{
    return clamp(value, 0, 63);
}

int clamp5(int value)
{
    return clamp(value, 0, 31);
}

Color6A5 decodeTexelRgb6a5(uvec4 texel)
{
    Color6A5 color;
    color.r = int(texel.r & 0x3Fu);
    color.g = int(texel.g & 0x3Fu);
    color.b = int(texel.b & 0x3Fu);
    color.a = int(texel.a & 0x1Fu);
    return color;
}

uvec4 fetchTextureArrayTexel(uint descriptorIndex, ivec3 coord)
{
    switch (descriptorIndex)
    {
        case 0u: return texelFetch(texArrays[0], coord, 0);
        case 1u: return texelFetch(texArrays[1], coord, 0);
        case 2u: return texelFetch(texArrays[2], coord, 0);
        case 3u: return texelFetch(texArrays[3], coord, 0);
        case 4u: return texelFetch(texArrays[4], coord, 0);
        case 5u: return texelFetch(texArrays[5], coord, 0);
        case 6u: return texelFetch(texArrays[6], coord, 0);
        case 7u: return texelFetch(texArrays[7], coord, 0);
        case 8u: return texelFetch(texArrays[8], coord, 0);
        case 9u: return texelFetch(texArrays[9], coord, 0);
        case 10u: return texelFetch(texArrays[10], coord, 0);
        case 11u: return texelFetch(texArrays[11], coord, 0);
        case 12u: return texelFetch(texArrays[12], coord, 0);
        case 13u: return texelFetch(texArrays[13], coord, 0);
        case 14u: return texelFetch(texArrays[14], coord, 0);
        case 15u: return texelFetch(texArrays[15], coord, 0);
        case 16u: return texelFetch(texArrays[16], coord, 0);
        case 17u: return texelFetch(texArrays[17], coord, 0);
        case 18u: return texelFetch(texArrays[18], coord, 0);
        case 19u: return texelFetch(texArrays[19], coord, 0);
        case 20u: return texelFetch(texArrays[20], coord, 0);
        case 21u: return texelFetch(texArrays[21], coord, 0);
        case 22u: return texelFetch(texArrays[22], coord, 0);
        case 23u: return texelFetch(texArrays[23], coord, 0);
        case 24u: return texelFetch(texArrays[24], coord, 0);
        case 25u: return texelFetch(texArrays[25], coord, 0);
        case 26u: return texelFetch(texArrays[26], coord, 0);
        case 27u: return texelFetch(texArrays[27], coord, 0);
        case 28u: return texelFetch(texArrays[28], coord, 0);
        case 29u: return texelFetch(texArrays[29], coord, 0);
        case 30u: return texelFetch(texArrays[30], coord, 0);
        case 31u: return texelFetch(texArrays[31], coord, 0);
        case 32u: return texelFetch(texArrays[32], coord, 0);
        case 33u: return texelFetch(texArrays[33], coord, 0);
        case 34u: return texelFetch(texArrays[34], coord, 0);
        case 35u: return texelFetch(texArrays[35], coord, 0);
        case 36u: return texelFetch(texArrays[36], coord, 0);
        case 37u: return texelFetch(texArrays[37], coord, 0);
        case 38u: return texelFetch(texArrays[38], coord, 0);
        case 39u: return texelFetch(texArrays[39], coord, 0);
        case 40u: return texelFetch(texArrays[40], coord, 0);
        case 41u: return texelFetch(texArrays[41], coord, 0);
        case 42u: return texelFetch(texArrays[42], coord, 0);
        case 43u: return texelFetch(texArrays[43], coord, 0);
        case 44u: return texelFetch(texArrays[44], coord, 0);
        case 45u: return texelFetch(texArrays[45], coord, 0);
        case 46u: return texelFetch(texArrays[46], coord, 0);
        case 47u: return texelFetch(texArrays[47], coord, 0);
        case 48u: return texelFetch(texArrays[48], coord, 0);
        case 49u: return texelFetch(texArrays[49], coord, 0);
        case 50u: return texelFetch(texArrays[50], coord, 0);
        case 51u: return texelFetch(texArrays[51], coord, 0);
        case 52u: return texelFetch(texArrays[52], coord, 0);
        case 53u: return texelFetch(texArrays[53], coord, 0);
        case 54u: return texelFetch(texArrays[54], coord, 0);
        case 55u: return texelFetch(texArrays[55], coord, 0);
        case 56u: return texelFetch(texArrays[56], coord, 0);
        case 57u: return texelFetch(texArrays[57], coord, 0);
        case 58u: return texelFetch(texArrays[58], coord, 0);
        case 59u: return texelFetch(texArrays[59], coord, 0);
        case 60u: return texelFetch(texArrays[60], coord, 0);
        case 61u: return texelFetch(texArrays[61], coord, 0);
        case 62u: return texelFetch(texArrays[62], coord, 0);
        case 63u: return texelFetch(texArrays[63], coord, 0);
        case 64u: return texelFetch(texArrays[64], coord, 0);
        case 65u: return texelFetch(texArrays[65], coord, 0);
        case 66u: return texelFetch(texArrays[66], coord, 0);
        case 67u: return texelFetch(texArrays[67], coord, 0);
        case 68u: return texelFetch(texArrays[68], coord, 0);
        case 69u: return texelFetch(texArrays[69], coord, 0);
        case 70u: return texelFetch(texArrays[70], coord, 0);
        case 71u: return texelFetch(texArrays[71], coord, 0);
        case 72u: return texelFetch(texArrays[72], coord, 0);
        case 73u: return texelFetch(texArrays[73], coord, 0);
        case 74u: return texelFetch(texArrays[74], coord, 0);
        case 75u: return texelFetch(texArrays[75], coord, 0);
        case 76u: return texelFetch(texArrays[76], coord, 0);
        case 77u: return texelFetch(texArrays[77], coord, 0);
        case 78u: return texelFetch(texArrays[78], coord, 0);
        case 79u: return texelFetch(texArrays[79], coord, 0);
        case 80u: return texelFetch(texArrays[80], coord, 0);
        case 81u: return texelFetch(texArrays[81], coord, 0);
        case 82u: return texelFetch(texArrays[82], coord, 0);
        case 83u: return texelFetch(texArrays[83], coord, 0);
        case 84u: return texelFetch(texArrays[84], coord, 0);
        case 85u: return texelFetch(texArrays[85], coord, 0);
        case 86u: return texelFetch(texArrays[86], coord, 0);
        case 87u: return texelFetch(texArrays[87], coord, 0);
        case 88u: return texelFetch(texArrays[88], coord, 0);
        case 89u: return texelFetch(texArrays[89], coord, 0);
        case 90u: return texelFetch(texArrays[90], coord, 0);
        case 91u: return texelFetch(texArrays[91], coord, 0);
        case 92u: return texelFetch(texArrays[92], coord, 0);
        case 93u: return texelFetch(texArrays[93], coord, 0);
        case 94u: return texelFetch(texArrays[94], coord, 0);
        case 95u: return texelFetch(texArrays[95], coord, 0);
        case 96u: return texelFetch(texArrays[96], coord, 0);
        case 97u: return texelFetch(texArrays[97], coord, 0);
        case 98u: return texelFetch(texArrays[98], coord, 0);
        case 99u: return texelFetch(texArrays[99], coord, 0);
        case 100u: return texelFetch(texArrays[100], coord, 0);
        case 101u: return texelFetch(texArrays[101], coord, 0);
        case 102u: return texelFetch(texArrays[102], coord, 0);
        case 103u: return texelFetch(texArrays[103], coord, 0);
        case 104u: return texelFetch(texArrays[104], coord, 0);
        case 105u: return texelFetch(texArrays[105], coord, 0);
        case 106u: return texelFetch(texArrays[106], coord, 0);
        case 107u: return texelFetch(texArrays[107], coord, 0);
        case 108u: return texelFetch(texArrays[108], coord, 0);
        case 109u: return texelFetch(texArrays[109], coord, 0);
        case 110u: return texelFetch(texArrays[110], coord, 0);
        case 111u: return texelFetch(texArrays[111], coord, 0);
        case 112u: return texelFetch(texArrays[112], coord, 0);
        case 113u: return texelFetch(texArrays[113], coord, 0);
        case 114u: return texelFetch(texArrays[114], coord, 0);
        case 115u: return texelFetch(texArrays[115], coord, 0);
        case 116u: return texelFetch(texArrays[116], coord, 0);
        case 117u: return texelFetch(texArrays[117], coord, 0);
        case 118u: return texelFetch(texArrays[118], coord, 0);
        case 119u: return texelFetch(texArrays[119], coord, 0);
        case 120u: return texelFetch(texArrays[120], coord, 0);
        case 121u: return texelFetch(texArrays[121], coord, 0);
        case 122u: return texelFetch(texArrays[122], coord, 0);
        case 123u: return texelFetch(texArrays[123], coord, 0);
        case 124u: return texelFetch(texArrays[124], coord, 0);
        case 125u: return texelFetch(texArrays[125], coord, 0);
        case 126u: return texelFetch(texArrays[126], coord, 0);
        case 127u: return texelFetch(texArrays[127], coord, 0);
        default: return texelFetch(texArrays[127], coord, 0);
    }
}

int wrapTexelCoord(int coord, int size, bool repeat, bool mirror)
{
    if (size <= 0)
        return 0;

    if (repeat)
    {
        if (mirror)
        {
            if ((coord & size) != 0)
                coord = (size - 1) - (coord & (size - 1));
            else
                coord = coord & (size - 1);
        }
        else
        {
            coord = coord & (size - 1);
        }
    }
    else
    {
        coord = clamp(coord, 0, size - 1);
    }

    return coord;
}

Color6A5 unpackToonColor(uint shadeIndex)
{
    uint clampedIndex = min(shadeIndex, 31u);
    uint packedColor = toonValues[clampedIndex];

    Color6A5 color;
    color.r = int(packedColor & 0x3Fu);
    color.g = int((packedColor >> 8u) & 0x3Fu);
    color.b = int((packedColor >> 16u) & 0x3Fu);
    color.a = 31;
    return color;
}

Color6A5 sampleTexture()
{
    Color6A5 whiteTexel;
    whiteTexel.r = 63;
    whiteTexel.g = 63;
    whiteTexel.b = 63;
    whiteTexel.a = 31;

    uint flags = fTriInfo0.x;
    uint texLayer = fTriInfo0.y;
    uint texArrayIndex = fTriInfo0.z;
    uint texWidth = fTriInfo0.w;
    uint texHeight = fTriInfo1.x;
    uint texParam = fTriInfo1.y;

    if ((flags & TRI_FLAG_TEXTURED) == 0u
        || texWidth == 0u
        || texHeight == 0u
        || texArrayIndex >= MAX_TEXTURE_DESCRIPTORS)
    {
        return whiteTexel;
    }

    int sampleS = int(floor(fTexcoord.x));
    int sampleT = int(floor(fTexcoord.y));
    bool repeatS = (texParam & (1u << 16u)) != 0u;
    bool repeatT = (texParam & (1u << 17u)) != 0u;
    bool mirrorS = (texParam & (1u << 18u)) != 0u;
    bool mirrorT = (texParam & (1u << 19u)) != 0u;

    sampleS = wrapTexelCoord(sampleS, int(texWidth), repeatS, mirrorS);
    sampleT = wrapTexelCoord(sampleT, int(texHeight), repeatT, mirrorT);

    uvec4 texel = fetchTextureArrayTexel(texArrayIndex, ivec3(sampleS, sampleT, int(texLayer)));
    return decodeTexelRgb6a5(texel);
}

vec4 encodeColor(Color6A5 color)
{
    return vec4(
        float(clamp6(color.r)) * (1.0 / 63.0),
        float(clamp6(color.g)) * (1.0 / 63.0),
        float(clamp6(color.b)) * (1.0 / 63.0),
        float(clamp5(color.a)) * (1.0 / 31.0));
}

void main()
{
    uint flags = fTriInfo0.x;
    uint polyAttr = fTriInfo1.z;
    uint polyAlpha = (polyAttr >> 16u) & 0x1Fu;
    uint blendMode = (polyAttr >> 4u) & 0x3u;
    bool highlightEnabled = (pc.dispCnt & (1u << 1u)) != 0u;
    bool textureMapsEnabled = (pc.dispCnt & (1u << 0u)) != 0u;

    Color6A5 sourceColor;
    sourceColor.r = clamp6(int(clamp(fColor.r * 63.0 + 0.5, 0.0, 63.0)));
    sourceColor.g = clamp6(int(clamp(fColor.g * 63.0 + 0.5, 0.0, 63.0)));
    sourceColor.b = clamp6(int(clamp(fColor.b * 63.0 + 0.5, 0.0, 63.0)));
    sourceColor.a = clamp5(int(polyAlpha));

    int highlightShade = sourceColor.r;
    if (blendMode == 2u)
    {
        if (highlightEnabled)
        {
            sourceColor.g = sourceColor.r;
            sourceColor.b = sourceColor.r;
            highlightShade = sourceColor.r;
        }
        else
        {
            Color6A5 toonColor = unpackToonColor(uint(clamp(sourceColor.r >> 1, 0, 31)));
            sourceColor.r = toonColor.r;
            sourceColor.g = toonColor.g;
            sourceColor.b = toonColor.b;
        }
    }

    if (textureMapsEnabled && (flags & TRI_FLAG_TEXTURED) != 0u)
    {
        Color6A5 texelColor = sampleTexture();
        if ((flags & TRI_FLAG_DECAL) != 0u)
        {
            if (texelColor.a >= 31)
            {
                sourceColor.r = texelColor.r;
                sourceColor.g = texelColor.g;
                sourceColor.b = texelColor.b;
            }
            else if (texelColor.a > 0)
            {
                sourceColor.r = clamp6(((texelColor.r * texelColor.a) + (sourceColor.r * (31 - texelColor.a))) >> 5);
                sourceColor.g = clamp6(((texelColor.g * texelColor.a) + (sourceColor.g * (31 - texelColor.a))) >> 5);
                sourceColor.b = clamp6(((texelColor.b * texelColor.a) + (sourceColor.b * (31 - texelColor.a))) >> 5);
            }
        }
        else
        {
            sourceColor.r = clamp6((((texelColor.r + 1) * (sourceColor.r + 1)) - 1) >> 6);
            sourceColor.g = clamp6((((texelColor.g + 1) * (sourceColor.g + 1)) - 1) >> 6);
            sourceColor.b = clamp6((((texelColor.b + 1) * (sourceColor.b + 1)) - 1) >> 6);
            sourceColor.a = clamp5((((texelColor.a + 1) * (sourceColor.a + 1)) - 1) >> 5);
        }
    }

    if (blendMode == 2u && highlightEnabled)
    {
        Color6A5 highlightColor = unpackToonColor(uint(clamp(highlightShade >> 1, 0, 31)));
        sourceColor.r = clamp6(sourceColor.r + highlightColor.r);
        sourceColor.g = clamp6(sourceColor.g + highlightColor.g);
        sourceColor.b = clamp6(sourceColor.b + highlightColor.b);
    }

    if (polyAlpha == 0u)
        sourceColor.a = 31;

    if (sourceColor.a <= int(pc.alphaRef))
        discard;

    if (EDGE_MARK_PASS != 0u)
    {
        if (sourceColor.a < 31)
            discard;

        oColor = vec4(0.0);
        oAttr = vec4(0.0, 1.0, 0.0, 1.0);

        float edgeDepth = DEPTH_INTERPOLATION_MODE != 0u ? fDepthPerspective : fDepthLinear;
        oDepthValue = edgeDepth;
        gl_FragDepth = edgeDepth;
        return;
    }

    if (TRANSLUCENT_PASS != 0u)
    {
        if (sourceColor.a <= 0 || sourceColor.a >= 31)
            discard;

        oColor = encodeColor(sourceColor);
        oAttr = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
        if (sourceColor.a < 31)
            discard;

        uint polyId = (polyAttr >> 24u) & 0x3Fu;
        float fogFlag = ((polyAttr & (1u << 15u)) != 0u) ? 1.0 : 0.0;
        oColor = encodeColor(sourceColor);
        oAttr = vec4(float(polyId) * (1.0 / 63.0), 0.0, fogFlag, 1.0);
    }

    float depth = DEPTH_INTERPOLATION_MODE != 0u ? fDepthPerspective : fDepthLinear;
    oDepthValue = depth;
    gl_FragDepth = depth;
}
