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

#ifndef GPU2D_HDPACK_H
#define GPU2D_HDPACK_H

#include "types.h"

#include <unordered_set>
#include <vector>

namespace melonDS
{

class GPU;
class HDTexPack;
struct HDTexPackImage;

// One replaced 2D asset occurrence for the current frame: an OBJ sprite or a
// text BG tile whose content hash matched a pack entry. The presenter
// composites Image over the packed plane pixels the producer won, so
// occlusion, priority and blending stay with the original composition.
struct HDPack2DInstance
{
    const HDTexPackImage* Image;
    u8 Engine;       // 0 = engine A, 1 = engine B
    u8 RequireMask;  // packed flag bits that must be set (0x90 OBJ, 1<<n BG n)
    u8 RejectMask;   // packed flag bits that must be clear (0x90 for BG tiles)
    u8 Flip;         // bit 0 horizontal, bit 1 vertical
    s16 X, Y;        // native screen coordinates, may be negative
    u16 W, H;        // native pixel size
};

// CPU-side 2D asset walker: decodes active OBJ sprites and text BG tiles
// straight from OAM/VRAM after a frame has been rendered, dumping them
// through HDTexPack at a throttled cadence and building the per-frame
// replacement instance list. Identity hashing matches the desktop dumper
// (chained XXH64 over the encoded guest bytes plus a used-range palette
// hash) so packs are interchangeable between platforms.
//
// v1 limits (frame-level walker by design):
//  - replacement skips rotscale sprites and non-text BG layers
//  - OAM, scroll and palette state are sampled once at end of frame; games
//    that rewrite BG scroll in HBlank or use mosaic get replacements
//    positioned from the final register state, which can misplace them on
//    such frames (they are not detected and do not fall back)
//  - overlapping OBJ replacements share the generic OBJ producer mask; the
//    instance carries no OAM slot or priority, so where two sprites overlap
//    the presenter cannot tell which sprite actually won a pixel
class HDPack2D
{
public:
    // call once per emulated frame on the emu thread, after RunFrame
    void ProcessFrame(GPU& gpu, HDTexPack* pack);

    std::vector<HDPack2DInstance> Instances;

private:
    void WalkSprites(GPU& gpu, int num, HDTexPack* pack, bool dump, bool load);
    void WalkBGLayers(GPU& gpu, int num, HDTexPack* pack, bool dump, bool load);
    void EmitSpriteInstance(const HDTexPackImage* img, int num, u8 flip,
                            s32 xpos, s32 ypos, int width, int height);

    u32 FrameCounter = 0;
    u32 WalkBatch = 0;

    // bitmap sprites can alias volatile display-capture VRAM; only dump
    // content whose hash survives across two sampled walks
    std::unordered_set<u64> PrevBitmapKeys, CurBitmapKeys;

    std::vector<u32> PixelScratch;
};

}

#endif
