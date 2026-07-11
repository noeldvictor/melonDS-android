/*
    Copyright 2016-2025 melonDS team

    This file is part of melonDS.

    melonDS is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    melonDS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
    details.

    You should have received a copy of the GNU General Public License along
    with melonDS. If not, see http://www.gnu.org/licenses/.
*/

#include "HDTextureFilter.h"

#include <algorithm>
#include <cmath>

namespace melonDS
{

namespace
{
u8 Channel(u32 color, int shift)
{
    return static_cast<u8>((color >> shift) & 0xFF);
}

u32 Pack(u8 r, u8 g, u8 b, u8 a)
{
    return static_cast<u32>(r) |
        (static_cast<u32>(g) << 8) |
        (static_cast<u32>(b) << 16) |
        (static_cast<u32>(a) << 24);
}

u8 BlendChannel(u8 a, u8 b, int weight)
{
    return static_cast<u8>((static_cast<int>(a) * (256 - weight) + static_cast<int>(b) * weight + 128) >> 8);
}

u8 ClampChannel(int value)
{
    return static_cast<u8>(std::clamp(value, 0, 255));
}

u32 Blend(u32 a, u32 b, int weight)
{
    weight = std::clamp(weight, 0, 256);
    return Pack(BlendChannel(Channel(a, 0), Channel(b, 0), weight),
                BlendChannel(Channel(a, 8), Channel(b, 8), weight),
                BlendChannel(Channel(a, 16), Channel(b, 16), weight),
                BlendChannel(Channel(a, 24), Channel(b, 24), weight));
}

u32 Average(u32 a, u32 b)
{
    return Blend(a, b, 128);
}

u32 Average4(u32 a, u32 b, u32 c, u32 d)
{
    return Pack(static_cast<u8>((static_cast<int>(Channel(a, 0)) + Channel(b, 0) + Channel(c, 0) + Channel(d, 0) + 2) >> 2),
                static_cast<u8>((static_cast<int>(Channel(a, 8)) + Channel(b, 8) + Channel(c, 8) + Channel(d, 8) + 2) >> 2),
                static_cast<u8>((static_cast<int>(Channel(a, 16)) + Channel(b, 16) + Channel(c, 16) + Channel(d, 16) + 2) >> 2),
                static_cast<u8>((static_cast<int>(Channel(a, 24)) + Channel(b, 24) + Channel(c, 24) + Channel(d, 24) + 2) >> 2));
}

u32 Average9(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i)
{
    return Pack(static_cast<u8>((static_cast<int>(Channel(a, 0)) + Channel(b, 0) + Channel(c, 0) + Channel(d, 0) + Channel(e, 0) + Channel(f, 0) + Channel(g, 0) + Channel(h, 0) + Channel(i, 0) + 4) / 9),
                static_cast<u8>((static_cast<int>(Channel(a, 8)) + Channel(b, 8) + Channel(c, 8) + Channel(d, 8) + Channel(e, 8) + Channel(f, 8) + Channel(g, 8) + Channel(h, 8) + Channel(i, 8) + 4) / 9),
                static_cast<u8>((static_cast<int>(Channel(a, 16)) + Channel(b, 16) + Channel(c, 16) + Channel(d, 16) + Channel(e, 16) + Channel(f, 16) + Channel(g, 16) + Channel(h, 16) + Channel(i, 16) + 4) / 9),
                static_cast<u8>((static_cast<int>(Channel(a, 24)) + Channel(b, 24) + Channel(c, 24) + Channel(d, 24) + Channel(e, 24) + Channel(f, 24) + Channel(g, 24) + Channel(h, 24) + Channel(i, 24) + 4) / 9));
}

int ColorDistance(u32 a, u32 b)
{
    return std::max({
        std::abs(static_cast<int>(Channel(a, 0)) - static_cast<int>(Channel(b, 0))),
        std::abs(static_cast<int>(Channel(a, 8)) - static_cast<int>(Channel(b, 8))),
        std::abs(static_cast<int>(Channel(a, 16)) - static_cast<int>(Channel(b, 16))),
        std::abs(static_cast<int>(Channel(a, 24)) - static_cast<int>(Channel(b, 24))),
    });
}

bool Similar(u32 a, u32 b)
{
    return ColorDistance(a, b) < 2;
}

bool SoftSimilar(u32 a, u32 b)
{
    return ColorDistance(a, b) < 24;
}

int FloorDiv2(int value)
{
    return (value >= 0) ? (value / 2) : ((value - 1) / 2);
}

int PositiveMod2(int value)
{
    int ret = value % 2;
    return ret < 0 ? ret + 2 : ret;
}

bool IsThinContrastDetail(u32 e, u32 b, u32 d, u32 f, u32 h)
{
    if (Channel(e, 24) <= 2)
        return false;

    int support = 0;
    support += Similar(e, b) ? 1 : 0;
    support += Similar(e, d) ? 1 : 0;
    support += Similar(e, f) ? 1 : 0;
    support += Similar(e, h) ? 1 : 0;

    int farthest = std::max({
        ColorDistance(e, b),
        ColorDistance(e, d),
        ColorDistance(e, f),
        ColorDistance(e, h),
    });
    return support <= 2 && farthest > 48;
}

u32 ProtectThinDetail(u32 candidate, u32 b, u32 d, u32 e, u32 f, u32 h)
{
    if (IsThinContrastDetail(e, b, d, f, h) && ColorDistance(candidate, e) > 8)
        return e;
    return candidate;
}

u32 FetchTexel(const u32* src, u32 width, u32 height, int x, int y)
{
    x = std::clamp(x, 0, static_cast<int>(width) - 1);
    y = std::clamp(y, 0, static_cast<int>(height) - 1);
    return src[x + y * width];
}

u32 Scale2xTexel(u32 b, u32 d, u32 e, u32 f, u32 h, int subx, int suby)
{
    if (Similar(b, h) || Similar(d, f))
        return e;

    if (subx == 0 && suby == 0)
        return Similar(d, b) ? d : e;
    if (subx == 1 && suby == 0)
        return Similar(b, f) ? f : e;
    if (subx == 0 && suby == 1)
        return Similar(d, h) ? d : e;
    return Similar(h, f) ? f : e;
}

u32 HQ2xLiteTexel(u32 b, u32 d, u32 e, u32 f, u32 h, int subx, int suby)
{
    u32 scaled = Scale2xTexel(b, d, e, f, h, subx, suby);
    if (Similar(e, scaled))
        return e;

    u32 cardinal = Average4(b, d, f, h);
    int edge = std::max(ColorDistance(b, h), ColorDistance(d, f));
    u32 softened = Blend(e, scaled, 166);
    return Blend(softened, cardinal, std::min(edge * 3, 166) / 3);
}

u32 TwoXSaILiteTexel(u32 b, u32 d, u32 e, u32 f, u32 h, int subx, int suby)
{
    u32 corner = e;
    if (subx == 0 && suby == 0)
        corner = (Similar(d, b) && !Similar(d, h) && !Similar(b, f)) ? Average(d, b) : e;
    else if (subx == 1 && suby == 0)
        corner = (Similar(b, f) && !Similar(b, d) && !Similar(f, h)) ? Average(b, f) : e;
    else if (subx == 0 && suby == 1)
        corner = (Similar(d, h) && !Similar(d, b) && !Similar(h, f)) ? Average(d, h) : e;
    else
        corner = (Similar(h, f) && !Similar(h, d) && !Similar(f, b)) ? Average(h, f) : e;

    return Blend(e, corner, 192);
}

u32 SuperEagleCorner(u32 e, u32 edge0, u32 edge1, u32 diagonal, u32 opposite0, u32 opposite1)
{
    if (Similar(edge0, edge1) && !Similar(opposite0, opposite1))
        return Blend(e, Average(edge0, edge1), 225);
    if (Similar(edge0, diagonal) && Similar(edge1, diagonal))
        return Blend(e, diagonal, 200);
    if (Similar(edge0, edge1))
        return Blend(e, Average(edge0, edge1), 141);
    return e;
}

u32 SuperEagleLiteTexel(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby)
{
    u32 tl = SuperEagleCorner(e, d, b, a, f, h);
    u32 tr = SuperEagleCorner(e, b, f, c, d, h);
    u32 bl = SuperEagleCorner(e, d, h, g, b, f);
    u32 br = SuperEagleCorner(e, h, f, i, d, b);

    if (Similar(a, i) && !Similar(c, g))
    {
        tl = Blend(tl, a, 115);
        br = Blend(br, i, 115);
    }
    else if (Similar(c, g) && !Similar(a, i))
    {
        tr = Blend(tr, c, 115);
        bl = Blend(bl, g, 115);
    }

    u32 corner = (subx == 0 && suby == 0) ? tl :
                 (subx == 1 && suby == 0) ? tr :
                 (subx == 0 && suby == 1) ? bl : br;

    if (Similar(corner, e))
    {
        bool vertical = Similar(b, h);
        bool horizontal = Similar(d, f);
        if (vertical && !horizontal)
            corner = Blend(e, Average(b, h), 46);
        else if (horizontal && !vertical)
            corner = Blend(e, Average(d, f), 46);
    }

    return corner;
}

u32 MMPXLiteCorner(u32 e, u32 edge0, u32 edge1, u32 diagonal, u32 opposite0, u32 opposite1)
{
    if (SoftSimilar(edge0, edge1) && !SoftSimilar(opposite0, opposite1))
        return Blend(e, Average(edge0, edge1), 235);
    if (SoftSimilar(edge0, diagonal) && SoftSimilar(edge1, diagonal))
        return Blend(e, diagonal, 215);
    if (SoftSimilar(edge0, diagonal) && !SoftSimilar(edge1, opposite1))
        return Blend(e, edge0, 184);
    if (SoftSimilar(edge1, diagonal) && !SoftSimilar(edge0, opposite0))
        return Blend(e, edge1, 184);
    if (SoftSimilar(edge0, edge1))
        return Blend(e, Average(edge0, edge1), 159);
    return e;
}

u32 MMPXLiteTexel(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby)
{
    u32 tl = MMPXLiteCorner(e, d, b, a, f, h);
    u32 tr = MMPXLiteCorner(e, b, f, c, d, h);
    u32 bl = MMPXLiteCorner(e, d, h, g, b, f);
    u32 br = MMPXLiteCorner(e, h, f, i, d, b);

    if (SoftSimilar(a, i) && !SoftSimilar(c, g))
    {
        tl = Blend(tl, a, 90);
        br = Blend(br, i, 90);
    }
    else if (SoftSimilar(c, g) && !SoftSimilar(a, i))
    {
        tr = Blend(tr, c, 90);
        bl = Blend(bl, g, 90);
    }

    u32 corner = (subx == 0 && suby == 0) ? tl :
                 (subx == 1 && suby == 0) ? tr :
                 (subx == 0 && suby == 1) ? bl : br;
    return Blend(corner, Average4(b, d, f, h), 46);
}

int SmoothStepWeight(int value, int low, int high)
{
    if (high <= low)
        return value >= high ? 256 : 0;

    int t = std::clamp(((value - low) * 256) / (high - low), 0, 256);
    return (t * t * (768 - (2 * t)) + 32768) >> 16;
}

u32 Sharpen(u32 color, u32 reference, int weight)
{
    weight = std::clamp(weight, 0, 256);
    return Pack(ClampChannel(Channel(color, 0) + (((static_cast<int>(Channel(color, 0)) - Channel(reference, 0)) * weight + 128) >> 8)),
                ClampChannel(Channel(color, 8) + (((static_cast<int>(Channel(color, 8)) - Channel(reference, 8)) * weight + 128) >> 8)),
                ClampChannel(Channel(color, 16) + (((static_cast<int>(Channel(color, 16)) - Channel(reference, 16)) * weight + 128) >> 8)),
                ClampChannel(Channel(color, 24) + (((static_cast<int>(Channel(color, 24)) - Channel(reference, 24)) * weight + 128) >> 8)));
}

u32 Anime4KLiteTexel(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby)
{
    u32 base = MMPXLiteTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    if (IsThinContrastDetail(e, b, d, f, h))
        return ProtectThinDetail(base, b, d, e, f, h);

    u32 cardinal = Average4(b, d, f, h);
    u32 pair = Average(d, f);
    int pairDist = ColorDistance(d, f);
    int verticalDist = ColorDistance(b, h);
    if (verticalDist < pairDist)
    {
        pair = Average(b, h);
        pairDist = verticalDist;
    }

    int diagDownDist = ColorDistance(a, i);
    if (diagDownDist < pairDist)
    {
        pair = Average(a, i);
        pairDist = diagDownDist;
    }

    int diagUpDist = ColorDistance(c, g);
    if (diagUpDist < pairDist)
        pair = Average(c, g);

    int contrast = std::max({
        ColorDistance(e, b),
        ColorDistance(e, d),
        ColorDistance(e, f),
        ColorDistance(e, h),
    });
    int edgeWeight = SmoothStepWeight(contrast, 4, 56);
    u32 smoothed = Blend(base, pair, (edgeWeight * 184 + 128) >> 8);
    u32 sharpened = Sharpen(smoothed, cardinal, (edgeWeight * 77 + 128) >> 8);
    return ProtectThinDetail(sharpened, b, d, e, f, h);
}

u32 SuperSAIStrongCorner(u32 e, u32 edge0, u32 edge1, u32 diagonal, u32 opposite0, u32 opposite1)
{
    int pairDist = ColorDistance(edge0, edge1);
    int oppositeDist = ColorDistance(opposite0, opposite1);
    int diag0 = ColorDistance(edge0, diagonal);
    int diag1 = ColorDistance(edge1, diagonal);

    if (pairDist < 40 && oppositeDist > pairDist + 10)
        return Blend(e, Average(edge0, edge1), 246);
    if (std::max(diag0, diag1) < 42)
        return Blend(e, diagonal, 236);
    if (diag0 < 36 && ColorDistance(edge1, opposite1) > diag0 + 8)
        return Blend(e, edge0, 220);
    if (diag1 < 36 && ColorDistance(edge0, opposite0) > diag1 + 8)
        return Blend(e, edge1, 220);
    if (pairDist < 56)
        return Blend(e, Average(edge0, edge1), 189);
    return e;
}

u32 SuperSAIStrongTexel(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby)
{
    u32 tl = SuperSAIStrongCorner(e, d, b, a, f, h);
    u32 tr = SuperSAIStrongCorner(e, b, f, c, d, h);
    u32 bl = SuperSAIStrongCorner(e, d, h, g, b, f);
    u32 br = SuperSAIStrongCorner(e, h, f, i, d, b);

    int diagDown = ColorDistance(a, i);
    int diagUp = ColorDistance(c, g);
    if (diagDown < diagUp + 8 && diagUp >= diagDown + 8)
    {
        tl = Blend(tl, a, 128);
        br = Blend(br, i, 128);
    }
    else if (diagUp < diagDown + 8 && diagDown >= diagUp + 8)
    {
        tr = Blend(tr, c, 128);
        bl = Blend(bl, g, 128);
    }

    u32 corner = (subx == 0 && suby == 0) ? tl :
                 (subx == 1 && suby == 0) ? tr :
                 (subx == 0 && suby == 1) ? bl : br;

    if (Similar(corner, e))
    {
        int horizontal = ColorDistance(d, f);
        int vertical = ColorDistance(b, h);
        int contrast = std::max({
            ColorDistance(e, b),
            ColorDistance(e, d),
            ColorDistance(e, f),
            ColorDistance(e, h),
        });
        int weight = (SmoothStepWeight(contrast, 6, 80) * 82 + 128) >> 8;
        corner = (horizontal < vertical)
            ? Blend(e, Average(d, f), weight)
            : Blend(e, Average(b, h), weight);
    }

    return Pack(Channel(corner, 0), Channel(corner, 8), Channel(corner, 16), Channel(e, 24));
}

u32 SuperEagleSmoothTexel(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby)
{
    u32 eagle = SuperEagleLiteTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    u32 sai = SuperSAIStrongTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    u32 guide = Blend(eagle, sai, 160);

    u32 neighborhood = Average9(a, b, c, d, e, f, g, h, i);
    u32 cardinal = Average4(b, d, f, h);
    u32 pair = Average(d, f);
    int pairDist = ColorDistance(d, f);

    int verticalDist = ColorDistance(b, h);
    if (verticalDist < pairDist)
    {
        pair = Average(b, h);
        pairDist = verticalDist;
    }

    int diagDownDist = ColorDistance(a, i);
    if (diagDownDist < pairDist)
    {
        pair = Average(a, i);
        pairDist = diagDownDist;
    }

    int diagUpDist = ColorDistance(c, g);
    if (diagUpDist < pairDist)
        pair = Average(c, g);

    int contrast = std::max({
        ColorDistance(e, b),
        ColorDistance(e, d),
        ColorDistance(e, f),
        ColorDistance(e, h),
    });
    int smoothWeight = 144 + ((SmoothStepWeight(contrast, 2, 64) * 96 + 128) >> 8);
    u32 soft = Blend(Blend(neighborhood, pair, 96), cardinal, 64);
    u32 smoothed = Blend(guide, soft, smoothWeight);
    return Pack(Channel(smoothed, 0), Channel(smoothed, 8), Channel(smoothed, 16), Channel(e, 24));
}

u32 CrispGradientTexel(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby)
{
    u32 anime = Anime4KLiteTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    u32 sai = SuperSAIStrongTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    u32 guide = Blend(anime, sai, 96);

    u32 neighborhood = Average9(a, b, c, d, e, f, g, h, i);
    u32 cardinal = Average4(b, d, f, h);
    u32 reference = Blend(neighborhood, cardinal, 90);

    int contrast = std::max({
        ColorDistance(e, b),
        ColorDistance(e, d),
        ColorDistance(e, f),
        ColorDistance(e, h),
    });
    int edgeWeight = SmoothStepWeight(contrast, 3, 72);
    int sharpenWeight = 51 + ((edgeWeight * 108 + 128) >> 8);
    u32 crisp = Sharpen(guide, reference, sharpenWeight);

    if (edgeWeight < 128)
        crisp = Blend(crisp, e, ((128 - edgeWeight) * 26 + 64) >> 7);

    return Pack(Channel(crisp, 0), Channel(crisp, 8), Channel(crisp, 16), Channel(e, 24));
}

u32 CrispEdgeAATexel(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby)
{
    u32 base = CrispGradientTexel(a, b, c, d, e, f, g, h, i, subx, suby);

    u32 xNeighbor = subx ? f : d;
    u32 yNeighbor = suby ? h : b;
    u32 diagonal = (subx && suby) ? i : (subx ? c : (suby ? g : a));
    u32 xSupport0 = subx ? c : a;
    u32 xSupport1 = subx ? i : g;
    u32 ySupport0 = suby ? g : a;
    u32 ySupport1 = suby ? i : c;

    int xWeight = SmoothStepWeight(ColorDistance(e, xNeighbor), 18, 92);
    int yWeight = SmoothStepWeight(ColorDistance(e, yNeighbor), 18, 92);
    int xSupport = (SoftSimilar(xNeighbor, xSupport0) ? 1 : 0) + (SoftSimilar(xNeighbor, xSupport1) ? 1 : 0);
    int ySupport = (SoftSimilar(yNeighbor, ySupport0) ? 1 : 0) + (SoftSimilar(yNeighbor, ySupport1) ? 1 : 0);
    xWeight = (xWeight * (154 + (xSupport * 51)) + 128) >> 8;
    yWeight = (yWeight * (154 + (ySupport * 51)) + 128) >> 8;

    u32 edge = Blend(base, xNeighbor, (xWeight * 72 + 128) >> 8);
    edge = Blend(edge, yNeighbor, (yWeight * 72 + 128) >> 8);

    int diagonalSupport = (SoftSimilar(diagonal, xNeighbor) ? 1 : 0) + (SoftSimilar(diagonal, yNeighbor) ? 1 : 0);
    int diagonalWeight = SmoothStepWeight(ColorDistance(e, diagonal), 22, 110);
    diagonalWeight = (diagonalWeight * diagonalSupport * 36 + 128) >> 8;
    edge = Blend(edge, diagonal, diagonalWeight);

    int maxEdge = std::max({xWeight, yWeight, diagonalWeight * 3});
    u32 reference = Blend(Average4(b, d, f, h), e, 80);
    u32 crisp = Sharpen(edge, reference, 18 + ((maxEdge * 46 + 128) >> 8));
    crisp = Pack(Channel(crisp, 0), Channel(crisp, 8), Channel(crisp, 16), Channel(e, 24));
    return ProtectThinDetail(crisp, b, d, e, f, h);
}

u32 HeavyFilterCorner(u32 a, u32 b, u32 c, u32 d, u32 e, u32 f, u32 g, u32 h, u32 i, int subx, int suby, int mode)
{
    u32 filtered;
    if (mode == 4)
        filtered = HQ2xLiteTexel(b, d, e, f, h, subx, suby);
    else if (mode == 5)
        filtered = TwoXSaILiteTexel(b, d, e, f, h, subx, suby);
    else if (mode == 6)
        filtered = SuperEagleLiteTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    else if (mode == 7)
        filtered = MMPXLiteTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    else if (mode == 8)
        filtered = Anime4KLiteTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    else if (mode == 9)
        filtered = SuperSAIStrongTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    else if (mode == 10)
        filtered = SuperEagleSmoothTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    else if (mode == 11)
        filtered = CrispGradientTexel(a, b, c, d, e, f, g, h, i, subx, suby);
    else if (mode == 12)
        filtered = CrispEdgeAATexel(a, b, c, d, e, f, g, h, i, subx, suby);
    else
        filtered = Scale2xTexel(b, d, e, f, h, subx, suby);

    return (mode >= 4 && mode != 10) ? ProtectThinDetail(filtered, b, d, e, f, h) : filtered;
}

u32 HeavyFilterTexel(const u32* src, u32 width, u32 height, int x, int y, u32 subx, u32 suby, u32 scale, int mode)
{
    u32 a = FetchTexel(src, width, height, x - 1, y - 1);
    u32 b = FetchTexel(src, width, height, x, y - 1);
    u32 c = FetchTexel(src, width, height, x + 1, y - 1);
    u32 d = FetchTexel(src, width, height, x - 1, y);
    u32 e = FetchTexel(src, width, height, x, y);
    u32 f = FetchTexel(src, width, height, x + 1, y);
    u32 g = FetchTexel(src, width, height, x - 1, y + 1);
    u32 h = FetchTexel(src, width, height, x, y + 1);
    u32 i = FetchTexel(src, width, height, x + 1, y + 1);

    if (scale <= 1)
        return e;
    if (scale <= 2)
        return HeavyFilterCorner(a, b, c, d, e, f, g, h, i, subx >= 1, suby >= 1, mode);

    u32 tl = HeavyFilterCorner(a, b, c, d, e, f, g, h, i, 0, 0, mode);
    u32 tr = HeavyFilterCorner(a, b, c, d, e, f, g, h, i, 1, 0, mode);
    u32 bl = HeavyFilterCorner(a, b, c, d, e, f, g, h, i, 0, 1, mode);
    u32 br = HeavyFilterCorner(a, b, c, d, e, f, g, h, i, 1, 1, mode);
    int wx = static_cast<int>(((subx * 256u) + (scale / 2u)) / scale);
    int wy = static_cast<int>(((suby * 256u) + (scale / 2u)) / scale);
    u32 filtered = Blend(Blend(tl, tr, wx), Blend(bl, br, wx), wy);
    return (mode >= 4 && mode != 10) ? ProtectThinDetail(filtered, b, d, e, f, h) : filtered;
}

u32 FirstStageHeavyTexel(const u32* src, u32 width, u32 height, int midX, int midY, int mode)
{
    int x = FloorDiv2(midX);
    int y = FloorDiv2(midY);
    int subx = PositiveMod2(midX);
    int suby = PositiveMod2(midY);

    u32 a = FetchTexel(src, width, height, x - 1, y - 1);
    u32 b = FetchTexel(src, width, height, x, y - 1);
    u32 c = FetchTexel(src, width, height, x + 1, y - 1);
    u32 d = FetchTexel(src, width, height, x - 1, y);
    u32 e = FetchTexel(src, width, height, x, y);
    u32 f = FetchTexel(src, width, height, x + 1, y);
    u32 g = FetchTexel(src, width, height, x - 1, y + 1);
    u32 h = FetchTexel(src, width, height, x, y + 1);
    u32 i = FetchTexel(src, width, height, x + 1, y + 1);

    return HeavyFilterCorner(a, b, c, d, e, f, g, h, i, subx, suby, mode);
}

u32 Recursive4xHeavyTexel(const u32* src, u32 width, u32 height, int x, int y, u32 subx, u32 suby, u32 scale, int mode)
{
    // the two chained 2x stages produce a 4x phase grid; map the requested
    // subtexel to its nearest phase center so any scale >= 3 stays inside
    // the source texel (identity at 4x, no neighbour bleed at 8x)
    const u32 phaseX = std::min(3u, (subx * 4u + 2u) / scale);
    const u32 phaseY = std::min(3u, (suby * 4u + 2u) / scale);
    int midX = (x * 2) + static_cast<int>(phaseX / 2);
    int midY = (y * 2) + static_cast<int>(phaseY / 2);
    int secondSubX = static_cast<int>(phaseX % 2);
    int secondSubY = static_cast<int>(phaseY % 2);

    u32 a = FirstStageHeavyTexel(src, width, height, midX - 1, midY - 1, mode);
    u32 b = FirstStageHeavyTexel(src, width, height, midX, midY - 1, mode);
    u32 c = FirstStageHeavyTexel(src, width, height, midX + 1, midY - 1, mode);
    u32 d = FirstStageHeavyTexel(src, width, height, midX - 1, midY, mode);
    u32 e = FirstStageHeavyTexel(src, width, height, midX, midY, mode);
    u32 f = FirstStageHeavyTexel(src, width, height, midX + 1, midY, mode);
    u32 g = FirstStageHeavyTexel(src, width, height, midX - 1, midY + 1, mode);
    u32 h = FirstStageHeavyTexel(src, width, height, midX, midY + 1, mode);
    u32 i = FirstStageHeavyTexel(src, width, height, midX + 1, midY + 1, mode);

    return HeavyFilterCorner(a, b, c, d, e, f, g, h, i, secondSubX, secondSubY, mode);
}

// ---- ScaleFX (Sp00kyFox) CPU port, mode 13 ----
// Faithful whole-image translation of the libretro scalefx 5-pass pipeline at
// its native 3x: pass 0 color-metric map, pass 1 corner strengths, pass 2
// dominance/edge classification, pass 3 candidate resolution up to level 6,
// pass 4 subpixel emission using only original source colors. Fully
// transparent texels form their own color class so silhouette edges against
// transparency get the same corner treatment as opaque edges.

struct ScaleFXVec4
{
    float v[4];
};

struct ScaleFXFlags
{
    // bits 0..3 corners, 4..7 horizontal edges, 8..11 vertical edges, 12..15 orientation
    u16 bits = 0;
    bool Crn(int k) const { return ((bits >> k) & 1) != 0; }
    bool Hori(int k) const { return ((bits >> (4 + k)) & 1) != 0; }
    bool Vert(int k) const { return ((bits >> (8 + k)) & 1) != 0; }
    bool Or(int k) const { return ((bits >> (12 + k)) & 1) != 0; }
};

struct ScaleFXCandidates
{
    u8 crn[4];
    u8 mid[4];
};

float ScaleFXGE(float x, float y)
{
    return (x > y) ? 1.0f : 0.0f;
}

float ScaleFXLEQ(float x, float y)
{
    return (x <= y) ? 1.0f : 0.0f;
}

// scalefx-pass0 dist(): compuphase perceptual RGB metric, with fully
// transparent texels treated as one dedicated color class
float ScaleFXMetric(u32 a, u32 b)
{
    const bool transparentA = Channel(a, 24) == 0;
    const bool transparentB = Channel(b, 24) == 0;
    if (transparentA || transparentB)
        return (transparentA && transparentB) ? 0.0f : 1.0f;

    const float ra = Channel(a, 0) * (1.0f / 63.0f);
    const float ga = Channel(a, 8) * (1.0f / 63.0f);
    const float ba = Channel(a, 16) * (1.0f / 63.0f);
    const float rb = Channel(b, 0) * (1.0f / 63.0f);
    const float gb = Channel(b, 8) * (1.0f / 63.0f);
    const float bb = Channel(b, 16) * (1.0f / 63.0f);

    const float r = 0.5f * (ra + rb);
    const float dr = ra - rb;
    const float dg = ga - gb;
    const float db = ba - bb;
    return std::sqrt((2.0f + r) * dr * dr + 4.0f * dg * dg + (3.0f - r) * db * db) * (1.0f / 3.0f);
}

// scalefx-pass1 str(): corner strength at a 2x2 junction (SFX_CLR 0.5, SFX_SAA 1)
float ScaleFXStrength(float d, float ax, float ay, float bx, float by)
{
    const float diff = ax - ay;
    const float wght1 = std::max(0.5f - d, 0.0f) * 2.0f;
    const float dir = (std::min(ax, bx) + ax > std::min(ay, by) + ay) ? diff : -diff;
    const float wght2 = std::clamp((1.0f - d) + dir, 0.0f, 1.0f);
    return wght1 * wght2 * ax * ay;
}

void ScaleFXPass0(const u32* src, u32 width, u32 height, std::vector<ScaleFXVec4>& metric)
{
    metric.resize(static_cast<size_t>(width) * height);
    for (u32 y = 0; y < height; y++)
    {
        for (u32 x = 0; x < width; x++)
        {
            const int ix = static_cast<int>(x);
            const int iy = static_cast<int>(y);
            const u32 e = FetchTexel(src, width, height, ix, iy);
            ScaleFXVec4& out = metric[x + y * width];
            out.v[0] = ScaleFXMetric(e, FetchTexel(src, width, height, ix - 1, iy - 1));
            out.v[1] = ScaleFXMetric(e, FetchTexel(src, width, height, ix, iy - 1));
            out.v[2] = ScaleFXMetric(e, FetchTexel(src, width, height, ix + 1, iy - 1));
            out.v[3] = ScaleFXMetric(e, FetchTexel(src, width, height, ix + 1, iy));
        }
    }
}

const ScaleFXVec4& ScaleFXAt(const std::vector<ScaleFXVec4>& data, u32 width, u32 height, int x, int y)
{
    x = std::clamp(x, 0, static_cast<int>(width) - 1);
    y = std::clamp(y, 0, static_cast<int>(height) - 1);
    return data[static_cast<size_t>(x) + static_cast<size_t>(y) * width];
}

void ScaleFXPass1(const std::vector<ScaleFXVec4>& metric, u32 width, u32 height, std::vector<ScaleFXVec4>& strength)
{
    strength.resize(static_cast<size_t>(width) * height);
    for (u32 y = 0; y < height; y++)
    {
        for (u32 x = 0; x < width; x++)
        {
            const int ix = static_cast<int>(x);
            const int iy = static_cast<int>(y);
            const ScaleFXVec4& mA = ScaleFXAt(metric, width, height, ix - 1, iy - 1);
            const ScaleFXVec4& mB = ScaleFXAt(metric, width, height, ix, iy - 1);
            const ScaleFXVec4& mD = ScaleFXAt(metric, width, height, ix - 1, iy);
            const ScaleFXVec4& mE = ScaleFXAt(metric, width, height, ix, iy);
            const ScaleFXVec4& mF = ScaleFXAt(metric, width, height, ix + 1, iy);
            const ScaleFXVec4& mG = ScaleFXAt(metric, width, height, ix - 1, iy + 1);
            const ScaleFXVec4& mH = ScaleFXAt(metric, width, height, ix, iy + 1);
            const ScaleFXVec4& mI = ScaleFXAt(metric, width, height, ix + 1, iy + 1);

            ScaleFXVec4& out = strength[x + y * width];
            out.v[0] = ScaleFXStrength(mD.v[2], mD.v[3], mE.v[1], mA.v[3], mD.v[1]);
            out.v[1] = ScaleFXStrength(mF.v[0], mE.v[3], mE.v[1], mB.v[3], mF.v[1]);
            out.v[2] = ScaleFXStrength(mH.v[2], mE.v[3], mH.v[1], mH.v[3], mI.v[1]);
            out.v[3] = ScaleFXStrength(mH.v[0], mD.v[3], mH.v[1], mG.v[3], mG.v[1]);
        }
    }
}

// scalefx-pass2 clear(): necessary junction condition for orthogonal edges
bool ScaleFXClear(float crn0, float crn1, float a0, float a1, float b0, float b1)
{
    return (crn0 >= std::max(std::min(a0, a1), std::min(b0, b1))) &&
           (crn1 >= std::max(std::min(a0, b1), std::min(b0, a1)));
}

void ScaleFXPass2(const std::vector<ScaleFXVec4>& metric, const std::vector<ScaleFXVec4>& strength,
                  u32 width, u32 height, std::vector<ScaleFXFlags>& flags)
{
    flags.resize(static_cast<size_t>(width) * height);
    for (u32 y = 0; y < height; y++)
    {
        for (u32 x = 0; x < width; x++)
        {
            const int ix = static_cast<int>(x);
            const int iy = static_cast<int>(y);
            const ScaleFXVec4& mA = ScaleFXAt(metric, width, height, ix - 1, iy - 1);
            const ScaleFXVec4& mB = ScaleFXAt(metric, width, height, ix, iy - 1);
            const ScaleFXVec4& mD = ScaleFXAt(metric, width, height, ix - 1, iy);
            const ScaleFXVec4& mE = ScaleFXAt(metric, width, height, ix, iy);
            const ScaleFXVec4& mF = ScaleFXAt(metric, width, height, ix + 1, iy);
            const ScaleFXVec4& mG = ScaleFXAt(metric, width, height, ix - 1, iy + 1);
            const ScaleFXVec4& mH = ScaleFXAt(metric, width, height, ix, iy + 1);
            const ScaleFXVec4& mI = ScaleFXAt(metric, width, height, ix + 1, iy + 1);

            const ScaleFXVec4& sA = ScaleFXAt(strength, width, height, ix - 1, iy - 1);
            const ScaleFXVec4& sB = ScaleFXAt(strength, width, height, ix, iy - 1);
            const ScaleFXVec4& sC = ScaleFXAt(strength, width, height, ix + 1, iy - 1);
            const ScaleFXVec4& sD = ScaleFXAt(strength, width, height, ix - 1, iy);
            const ScaleFXVec4& sE = ScaleFXAt(strength, width, height, ix, iy);
            const ScaleFXVec4& sF = ScaleFXAt(strength, width, height, ix + 1, iy);
            const ScaleFXVec4& sG = ScaleFXAt(strength, width, height, ix - 1, iy + 1);
            const ScaleFXVec4& sH = ScaleFXAt(strength, width, height, ix, iy + 1);
            const ScaleFXVec4& sI = ScaleFXAt(strength, width, height, ix + 1, iy + 1);

            // strength & dominance junctions
            const float jS[4][4] = {
                { sA.v[2], sB.v[3], sE.v[0], sD.v[1] },
                { sB.v[2], sC.v[3], sF.v[0], sE.v[1] },
                { sE.v[2], sF.v[3], sI.v[0], sH.v[1] },
                { sD.v[2], sE.v[3], sH.v[0], sG.v[1] },
            };
            const float jD[4][4] = {
                { 2.0f * sA.v[2] - (sA.v[1] + sA.v[3]), 2.0f * sB.v[3] - (sB.v[2] + sB.v[0]),
                  2.0f * sE.v[0] - (sE.v[3] + sE.v[1]), 2.0f * sD.v[1] - (sD.v[0] + sD.v[2]) },
                { 2.0f * sB.v[2] - (sB.v[1] + sB.v[3]), 2.0f * sC.v[3] - (sC.v[2] + sC.v[0]),
                  2.0f * sF.v[0] - (sF.v[3] + sF.v[1]), 2.0f * sE.v[1] - (sE.v[0] + sE.v[2]) },
                { 2.0f * sE.v[2] - (sE.v[1] + sE.v[3]), 2.0f * sF.v[3] - (sF.v[2] + sF.v[0]),
                  2.0f * sI.v[0] - (sI.v[3] + sI.v[1]), 2.0f * sH.v[1] - (sH.v[0] + sH.v[2]) },
                { 2.0f * sD.v[2] - (sD.v[1] + sD.v[3]), 2.0f * sE.v[3] - (sE.v[2] + sE.v[0]),
                  2.0f * sH.v[0] - (sH.v[3] + sH.v[1]), 2.0f * sG.v[1] - (sG.v[0] + sG.v[2]) },
            };

            // majority vote for ambiguous dominance junctions
            float j[4][4];
            for (int n = 0; n < 4; n++)
            {
                for (int k = 0; k < 4; k++)
                {
                    const float* row = jD[n];
                    const float term = ScaleFXLEQ(row[(k + 1) & 3], 0.0f) * ScaleFXLEQ(row[(k + 3) & 3], 0.0f) +
                                       ScaleFXGE(row[k] + row[(k + 2) & 3], row[(k + 1) & 3] + row[(k + 3) & 3]);
                    j[n][k] = std::min(ScaleFXGE(row[k], 0.0f) * term, 1.0f);
                }
            }

            // inject strength without creating new contradictions
            float res[4];
            res[0] = std::min(j[0][2] + (1.0f - j[0][1]) * (1.0f - j[0][3]) * ScaleFXGE(jS[0][2], 0.0f) *
                              (j[0][0] + ScaleFXGE(jS[0][0] + jS[0][2], jS[0][1] + jS[0][3])), 1.0f);
            res[1] = std::min(j[1][3] + (1.0f - j[1][2]) * (1.0f - j[1][0]) * ScaleFXGE(jS[1][3], 0.0f) *
                              (j[1][1] + ScaleFXGE(jS[1][1] + jS[1][3], jS[1][0] + jS[1][2])), 1.0f);
            res[2] = std::min(j[2][0] + (1.0f - j[2][3]) * (1.0f - j[2][1]) * ScaleFXGE(jS[2][0], 0.0f) *
                              (j[2][2] + ScaleFXGE(jS[2][0] + jS[2][2], jS[2][1] + jS[2][3])), 1.0f);
            res[3] = std::min(j[3][1] + (1.0f - j[3][0]) * (1.0f - j[3][2]) * ScaleFXGE(jS[3][1], 0.0f) *
                              (j[3][3] + ScaleFXGE(jS[3][1] + jS[3][3], jS[3][0] + jS[3][2])), 1.0f);

            // single pixel & end of line detection
            const float jc[4] = { j[0][2], j[1][3], j[2][0], j[3][1] };
            float corner[4];
            for (int k = 0; k < 4; k++)
                corner[k] = std::min(res[k] * (jc[k] + (1.0f - res[(k + 3) & 3] * res[(k + 1) & 3])), 1.0f);

            const bool clr[4] = {
                ScaleFXClear(mD.v[2], mE.v[0], mD.v[3], mE.v[1], mA.v[3], mD.v[1]),
                ScaleFXClear(mF.v[0], mE.v[2], mE.v[3], mE.v[1], mB.v[3], mF.v[1]),
                ScaleFXClear(mH.v[2], mI.v[0], mE.v[3], mH.v[1], mH.v[3], mI.v[1]),
                ScaleFXClear(mH.v[0], mG.v[2], mD.v[3], mH.v[1], mG.v[3], mG.v[1]),
            };

            const float hEdge[4] = { std::min(mD.v[3], mA.v[3]), std::min(mE.v[3], mB.v[3]),
                                     std::min(mE.v[3], mH.v[3]), std::min(mD.v[3], mG.v[3]) };
            const float vEdge[4] = { std::min(mE.v[1], mD.v[1]), std::min(mE.v[1], mF.v[1]),
                                     std::min(mH.v[1], mI.v[1]), std::min(mH.v[1], mG.v[1]) };
            const float hAdd[4] = { mD.v[3], mE.v[3], mE.v[3], mD.v[3] };
            const float vAdd[4] = { mE.v[1], mE.v[1], mH.v[1], mH.v[1] };

            u16 bits = 0;
            for (int k = 0; k < 4; k++)
            {
                if (corner[k] > 0.0f)
                    bits |= static_cast<u16>(1u << k);
                if (hEdge[k] < vEdge[k] && clr[k])
                    bits |= static_cast<u16>(1u << (4 + k));
                if (hEdge[k] > vEdge[k] && clr[k])
                    bits |= static_cast<u16>(1u << (8 + k));
                if (hEdge[k] + hAdd[k] > vEdge[k] + vAdd[k])
                    bits |= static_cast<u16>(1u << (12 + k));
            }
            flags[x + y * width].bits = bits;
        }
    }
}

const ScaleFXFlags& ScaleFXFlagsAt(const std::vector<ScaleFXFlags>& data, u32 width, u32 height, int x, int y)
{
    x = std::clamp(x, 0, static_cast<int>(width) - 1);
    y = std::clamp(y, 0, static_cast<int>(height) - 1);
    return data[static_cast<size_t>(x) + static_cast<size_t>(y) * width];
}

void ScaleFXPass3(const std::vector<ScaleFXFlags>& flags, u32 width, u32 height,
                  std::vector<ScaleFXCandidates>& candidates)
{
    candidates.resize(static_cast<size_t>(width) * height);
    for (u32 y = 0; y < height; y++)
    {
        for (u32 x = 0; x < width; x++)
        {
            const int ix = static_cast<int>(x);
            const int iy = static_cast<int>(y);
            const ScaleFXFlags& fE = ScaleFXFlagsAt(flags, width, height, ix, iy);
            const ScaleFXFlags& fD = ScaleFXFlagsAt(flags, width, height, ix - 1, iy);
            const ScaleFXFlags& fD0 = ScaleFXFlagsAt(flags, width, height, ix - 2, iy);
            const ScaleFXFlags& fD1 = ScaleFXFlagsAt(flags, width, height, ix - 3, iy);
            const ScaleFXFlags& fF = ScaleFXFlagsAt(flags, width, height, ix + 1, iy);
            const ScaleFXFlags& fF0 = ScaleFXFlagsAt(flags, width, height, ix + 2, iy);
            const ScaleFXFlags& fF1 = ScaleFXFlagsAt(flags, width, height, ix + 3, iy);
            const ScaleFXFlags& fB = ScaleFXFlagsAt(flags, width, height, ix, iy - 1);
            const ScaleFXFlags& fB0 = ScaleFXFlagsAt(flags, width, height, ix, iy - 2);
            const ScaleFXFlags& fB1 = ScaleFXFlagsAt(flags, width, height, ix, iy - 3);
            const ScaleFXFlags& fH = ScaleFXFlagsAt(flags, width, height, ix, iy + 1);
            const ScaleFXFlags& fH0 = ScaleFXFlagsAt(flags, width, height, ix, iy + 2);
            const ScaleFXFlags& fH1 = ScaleFXFlagsAt(flags, width, height, ix, iy + 3);

            // lvl1 corners (SFX_SCN == 1 keeps plain corners active)
            const bool lvl1x = fE.Crn(0);
            const bool lvl1y = fE.Crn(1);
            const bool lvl1z = fE.Crn(2);
            const bool lvl1w = fE.Crn(3);

            // lvl2 mid (left, right / up, down)
            const bool lvl2x0 = (fE.Crn(0) && fE.Hori(1)) && fD.Crn(2);
            const bool lvl2x1 = (fE.Crn(1) && fE.Hori(0)) && fF.Crn(3);
            const bool lvl2y0 = (fE.Crn(1) && fE.Vert(2)) && fB.Crn(3);
            const bool lvl2y1 = (fE.Crn(2) && fE.Vert(1)) && fH.Crn(0);
            const bool lvl2z0 = (fE.Crn(3) && fE.Hori(2)) && fD.Crn(1);
            const bool lvl2z1 = (fE.Crn(2) && fE.Hori(3)) && fF.Crn(0);
            const bool lvl2w0 = (fE.Crn(0) && fE.Vert(3)) && fB.Crn(2);
            const bool lvl2w1 = (fE.Crn(3) && fE.Vert(0)) && fH.Crn(1);

            // lvl3 corners (hori, vert)
            const bool lvl3x0 = lvl2x1 && (fD.Hori(1) && fD.Hori(0)) && fF.Hori(2);
            const bool lvl3x1 = lvl2w1 && (fB.Vert(3) && fB.Vert(0)) && fH.Vert(2);
            const bool lvl3y0 = lvl2x0 && (fF.Hori(0) && fF.Hori(1)) && fD.Hori(3);
            const bool lvl3y1 = lvl2y1 && (fB.Vert(2) && fB.Vert(1)) && fH.Vert(3);
            const bool lvl3z0 = lvl2z0 && (fF.Hori(3) && fF.Hori(2)) && fD.Hori(0);
            const bool lvl3z1 = lvl2y0 && (fH.Vert(1) && fH.Vert(2)) && fB.Vert(0);
            const bool lvl3w0 = lvl2z1 && (fD.Hori(2) && fD.Hori(3)) && fF.Hori(1);
            const bool lvl3w1 = lvl2w0 && (fH.Vert(0) && fH.Vert(3)) && fB.Vert(1);

            // lvl4 corners (hori, vert)
            const bool lvl4x0 = (fD.Crn(0) && fD.Hori(1) && fE.Hori(0) && fE.Hori(1) && fF.Hori(0) && fF.Hori(1)) &&
                                (fD0.Crn(2) && fD0.Hori(3));
            const bool lvl4x1 = (fB.Crn(0) && fB.Vert(3) && fE.Vert(0) && fE.Vert(3) && fH.Vert(0) && fH.Vert(3)) &&
                                (fB0.Crn(2) && fB0.Vert(1));
            const bool lvl4y0 = (fF.Crn(1) && fF.Hori(0) && fE.Hori(1) && fE.Hori(0) && fD.Hori(1) && fD.Hori(0)) &&
                                (fF0.Crn(3) && fF0.Hori(2));
            const bool lvl4y1 = (fB.Crn(1) && fB.Vert(2) && fE.Vert(1) && fE.Vert(2) && fH.Vert(1) && fH.Vert(2)) &&
                                (fB0.Crn(3) && fB0.Vert(0));
            const bool lvl4z0 = (fF.Crn(2) && fF.Hori(3) && fE.Hori(2) && fE.Hori(3) && fD.Hori(2) && fD.Hori(3)) &&
                                (fF0.Crn(0) && fF0.Hori(1));
            const bool lvl4z1 = (fH.Crn(2) && fH.Vert(1) && fE.Vert(2) && fE.Vert(1) && fB.Vert(2) && fB.Vert(1)) &&
                                (fH0.Crn(0) && fH0.Vert(3));
            const bool lvl4w0 = (fD.Crn(3) && fD.Hori(2) && fE.Hori(3) && fE.Hori(2) && fF.Hori(3) && fF.Hori(2)) &&
                                (fD0.Crn(1) && fD0.Hori(0));
            const bool lvl4w1 = (fH.Crn(3) && fH.Vert(0) && fE.Vert(3) && fE.Vert(0) && fB.Vert(3) && fB.Vert(0)) &&
                                (fH0.Crn(1) && fH0.Vert(2));

            // lvl5 mid (left, right / up, down)
            const bool lvl5x0 = lvl4x0 && (fF0.Hori(0) && fF0.Hori(1)) && (fD1.Hori(2) && fD1.Hori(3));
            const bool lvl5x1 = lvl4y0 && (fD0.Hori(1) && fD0.Hori(0)) && (fF1.Hori(3) && fF1.Hori(2));
            const bool lvl5y0 = lvl4y1 && (fH0.Vert(1) && fH0.Vert(2)) && (fB1.Vert(3) && fB1.Vert(0));
            const bool lvl5y1 = lvl4z1 && (fB0.Vert(2) && fB0.Vert(1)) && (fH1.Vert(0) && fH1.Vert(3));
            const bool lvl5z0 = lvl4w0 && (fF0.Hori(3) && fF0.Hori(2)) && (fD1.Hori(1) && fD1.Hori(0));
            const bool lvl5z1 = lvl4z0 && (fD0.Hori(2) && fD0.Hori(3)) && (fF1.Hori(0) && fF1.Hori(1));
            const bool lvl5w0 = lvl4x1 && (fH0.Vert(0) && fH0.Vert(3)) && (fB1.Vert(2) && fB1.Vert(1));
            const bool lvl5w1 = lvl4w1 && (fB0.Vert(3) && fB0.Vert(0)) && (fH1.Vert(1) && fH1.Vert(2));

            // lvl6 corners (hori, vert)
            const bool lvl6x0 = lvl5x1 && (fD1.Hori(1) && fD1.Hori(0));
            const bool lvl6x1 = lvl5w1 && (fB1.Vert(3) && fB1.Vert(0));
            const bool lvl6y0 = lvl5x0 && (fF1.Hori(0) && fF1.Hori(1));
            const bool lvl6y1 = lvl5y1 && (fB1.Vert(2) && fB1.Vert(1));
            const bool lvl6z0 = lvl5z0 && (fF1.Hori(3) && fF1.Hori(2));
            const bool lvl6z1 = lvl5y0 && (fH1.Vert(1) && fH1.Vert(2));
            const bool lvl6w0 = lvl5z1 && (fD1.Hori(2) && fD1.Hori(3));
            const bool lvl6w1 = lvl5w0 && (fH1.Vert(0) && fH1.Vert(3));

            // subpixel candidates - 0 = E, 1 = D, 2 = D0, 3 = F, 4 = F0, 5 = B, 6 = B0, 7 = H, 8 = H0
            ScaleFXCandidates& out = candidates[x + y * width];
            out.crn[0] = ((lvl1x && fE.Or(0)) || (lvl3x0 && fE.Or(1)) || (lvl4x0 && fD.Or(0)) || (lvl6x0 && fF.Or(1))) ? 5
                       : (lvl1x || (lvl3x1 && !fE.Or(3)) || (lvl4x1 && !fB.Or(0)) || (lvl6x1 && !fH.Or(3))) ? 1
                       : lvl3x0 ? 3 : lvl3x1 ? 7 : lvl4x0 ? 2 : lvl4x1 ? 6 : lvl6x0 ? 4 : lvl6x1 ? 8 : 0;
            out.crn[1] = ((lvl1y && fE.Or(1)) || (lvl3y0 && fE.Or(0)) || (lvl4y0 && fF.Or(1)) || (lvl6y0 && fD.Or(0))) ? 5
                       : (lvl1y || (lvl3y1 && !fE.Or(2)) || (lvl4y1 && !fB.Or(1)) || (lvl6y1 && !fH.Or(2))) ? 3
                       : lvl3y0 ? 1 : lvl3y1 ? 7 : lvl4y0 ? 4 : lvl4y1 ? 6 : lvl6y0 ? 2 : lvl6y1 ? 8 : 0;
            out.crn[2] = ((lvl1z && fE.Or(2)) || (lvl3z0 && fE.Or(3)) || (lvl4z0 && fF.Or(2)) || (lvl6z0 && fD.Or(3))) ? 7
                       : (lvl1z || (lvl3z1 && !fE.Or(1)) || (lvl4z1 && !fH.Or(2)) || (lvl6z1 && !fB.Or(1))) ? 3
                       : lvl3z0 ? 1 : lvl3z1 ? 5 : lvl4z0 ? 4 : lvl4z1 ? 8 : lvl6z0 ? 2 : lvl6z1 ? 6 : 0;
            out.crn[3] = ((lvl1w && fE.Or(3)) || (lvl3w0 && fE.Or(2)) || (lvl4w0 && fD.Or(3)) || (lvl6w0 && fF.Or(2))) ? 7
                       : (lvl1w || (lvl3w1 && !fE.Or(0)) || (lvl4w1 && !fH.Or(3)) || (lvl6w1 && !fB.Or(0))) ? 1
                       : lvl3w0 ? 3 : lvl3w1 ? 5 : lvl4w0 ? 2 : lvl4w1 ? 8 : lvl6w0 ? 4 : lvl6w1 ? 6 : 0;

            out.mid[0] = ((lvl2x0 && fE.Or(0)) || (lvl2x1 && fE.Or(1)) || (lvl5x0 && fD.Or(0)) || (lvl5x1 && fF.Or(1))) ? 5
                       : lvl2x0 ? 1 : lvl2x1 ? 3 : lvl5x0 ? 2 : lvl5x1 ? 4
                       : (fE.Crn(0) && fD.Crn(2) && fE.Crn(1) && fF.Crn(3)) ? (fE.Or(0) ? (fE.Or(1) ? 5 : 3) : 1) : 0;
            out.mid[1] = ((lvl2y0 && !fE.Or(1)) || (lvl2y1 && !fE.Or(2)) || (lvl5y0 && !fB.Or(1)) || (lvl5y1 && !fH.Or(2))) ? 3
                       : lvl2y0 ? 5 : lvl2y1 ? 7 : lvl5y0 ? 6 : lvl5y1 ? 8
                       : (fE.Crn(1) && fB.Crn(3) && fE.Crn(2) && fH.Crn(0)) ? (!fE.Or(1) ? (!fE.Or(2) ? 3 : 7) : 5) : 0;
            out.mid[2] = ((lvl2z0 && fE.Or(3)) || (lvl2z1 && fE.Or(2)) || (lvl5z0 && fD.Or(3)) || (lvl5z1 && fF.Or(2))) ? 7
                       : lvl2z0 ? 1 : lvl2z1 ? 3 : lvl5z0 ? 2 : lvl5z1 ? 4
                       : (fE.Crn(2) && fF.Crn(0) && fE.Crn(3) && fD.Crn(1)) ? (fE.Or(2) ? (fE.Or(3) ? 7 : 1) : 3) : 0;
            out.mid[3] = ((lvl2w0 && !fE.Or(0)) || (lvl2w1 && !fE.Or(3)) || (lvl5w0 && !fB.Or(0)) || (lvl5w1 && !fH.Or(3))) ? 1
                       : lvl2w0 ? 5 : lvl2w1 ? 7 : lvl5w0 ? 6 : lvl5w1 ? 8
                       : (fE.Crn(3) && fH.Crn(1) && fE.Crn(0) && fB.Crn(2)) ? (!fE.Or(3) ? (!fE.Or(0) ? 1 : 5) : 7) : 0;
        }
    }
}

// scalefx-pass4: emit the 3x3 subpixel block for every source texel using
// only original source colors
void ScaleFX3x(const u32* src, u32 width, u32 height, std::vector<u32>& out)
{
    std::vector<ScaleFXCandidates> candidates;
    {
        std::vector<ScaleFXFlags> flags;
        {
            std::vector<ScaleFXVec4> metric;
            std::vector<ScaleFXVec4> strength;
            ScaleFXPass0(src, width, height, metric);
            ScaleFXPass1(metric, width, height, strength);
            ScaleFXPass2(metric, strength, width, height, flags);
        }
        ScaleFXPass3(flags, width, height, candidates);
    }

    static const int offsets[9][2] = {
        { 0, 0 }, { -1, 0 }, { -2, 0 }, { 1, 0 }, { 2, 0 },
        { 0, -1 }, { 0, -2 }, { 0, 1 }, { 0, 2 },
    };

    const u32 outWidth = width * 3;
    const u32 outHeight = height * 3;
    out.resize(static_cast<size_t>(outWidth) * outHeight);
    for (u32 y = 0; y < height; y++)
    {
        for (u32 x = 0; x < width; x++)
        {
            const ScaleFXCandidates& cand = candidates[x + y * width];
            const u8 subpixels[3][3] = {
                { cand.crn[0], cand.mid[0], cand.crn[1] },
                { cand.mid[3], 0, cand.mid[1] },
                { cand.crn[3], cand.mid[2], cand.crn[2] },
            };
            for (int sy = 0; sy < 3; sy++)
            {
                for (int sx = 0; sx < 3; sx++)
                {
                    const u8 sp = subpixels[sy][sx];
                    const u32 color = FetchTexel(src, width, height,
                                                 static_cast<int>(x) + offsets[sp][0],
                                                 static_cast<int>(y) + offsets[sp][1]);
                    out[(x * 3 + sx) + (y * 3 + sy) * outWidth] = color;
                }
            }
        }
    }
}
}

namespace HDTextureFilter
{

u32 ClampScale(int scale)
{
    if (scale < 1) return 1;
    if (scale > 4) return 4;
    return static_cast<u32>(scale);
}

int ClampMode(int mode)
{
    if (mode < 0) return 0;
    if (mode > 13) return 13;
    return mode;
}

void ScaleFXImage(const u32* src, u32 width, u32 height, u32 scale, std::vector<u32>& dst)
{
    const u32 dstWidth = width * scale;
    const u32 dstHeight = height * scale;

    std::vector<u32> stage;
    ScaleFX3x(src, width, height, stage);
    u32 stageWidth = width * 3;
    u32 stageHeight = height * 3;
    if (scale >= 6)
    {
        // run the native 3x pass twice (9x) before resampling down to scale
        std::vector<u32> stage2;
        ScaleFX3x(stage.data(), stageWidth, stageHeight, stage2);
        stage.swap(stage2);
        stageWidth *= 3;
        stageHeight *= 3;
    }

    if (stageWidth == dstWidth && stageHeight == dstHeight)
    {
        dst = std::move(stage);
        return;
    }

    // center-based nearest resample 3x/9x -> requested scale keeps the
    // ScaleFX color classes exact instead of introducing new blend colors
    dst.resize(static_cast<size_t>(dstWidth) * dstHeight);
    for (u32 y = 0; y < dstHeight; y++)
    {
        const u32 sy = static_cast<u32>(((2ull * y + 1ull) * stageHeight) / (2ull * dstHeight));
        for (u32 x = 0; x < dstWidth; x++)
        {
            const u32 sx = static_cast<u32>(((2ull * x + 1ull) * stageWidth) / (2ull * dstWidth));
            dst[x + y * dstWidth] = stage[sx + sy * stageWidth];
        }
    }
}

void UpscaleTexture(const u32* src, u32 width, u32 height, u32 scale, int mode, std::vector<u32>& dst)
{
    const u32 dstWidth = width * scale;
    const u32 dstHeight = height * scale;
    if (mode == 13)
    {
        // whole-image multi-pass ScaleFX instead of the per-texel heavy path
        ScaleFXImage(src, width, height, scale, dst);
        return;
    }
    dst.resize(static_cast<size_t>(dstWidth) * dstHeight);

    for (u32 y = 0; y < dstHeight; y++)
    {
        const u32 srcY = y / scale;
        const u32 subY = y - srcY * scale;
        for (u32 x = 0; x < dstWidth; x++)
        {
            const u32 srcX = x / scale;
            const u32 subX = x - srcX * scale;
            if (mode == 0)
            {
                // nearest: texture-pack storage scaling without a filter active
                dst[x + y * dstWidth] = src[srcX + srcY * width];
                continue;
            }
            if (mode >= 3)
            {
                dst[x + y * dstWidth] = (scale >= 3)
                    ? Recursive4xHeavyTexel(src, width, height,
                                            static_cast<int>(srcX), static_cast<int>(srcY),
                                            subX, subY, scale, mode)
                    : HeavyFilterTexel(src, width, height,
                                       static_cast<int>(srcX), static_cast<int>(srcY),
                                       subX, subY, scale, mode);
                continue;
            }

            float fx = (static_cast<float>(x) + 0.5f) / static_cast<float>(scale) - 0.5f;
            float fy = (static_cast<float>(y) + 0.5f) / static_cast<float>(scale) - 0.5f;
            int baseX = static_cast<int>(std::floor(fx));
            int baseY = static_cast<int>(std::floor(fy));
            float fracX = fx - static_cast<float>(baseX);
            float fracY = fy - static_cast<float>(baseY);
            if (mode == 2)
            {
                fracX = fracX * fracX * (3.0f - 2.0f * fracX);
                fracY = fracY * fracY * (3.0f - 2.0f * fracY);
            }
            int wx = static_cast<int>(std::clamp(fracX, 0.0f, 1.0f) * 256.0f + 0.5f);
            int wy = static_cast<int>(std::clamp(fracY, 0.0f, 1.0f) * 256.0f + 0.5f);
            u32 c00 = FetchTexel(src, width, height, baseX, baseY);
            u32 c10 = FetchTexel(src, width, height, baseX + 1, baseY);
            u32 c01 = FetchTexel(src, width, height, baseX, baseY + 1);
            u32 c11 = FetchTexel(src, width, height, baseX + 1, baseY + 1);
            dst[x + y * dstWidth] = Blend(Blend(c00, c10, wx), Blend(c01, c11, wx), wy);
        }
    }
}

}

}
