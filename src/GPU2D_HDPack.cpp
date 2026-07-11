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

#include "GPU2D_HDPack.h"

#include "GPU.h"
#include "HDTexPack.h"

#define XXH_STATIC_LINKING_ONLY
#include "xxhash/xxhash.h"

namespace melonDS
{

namespace
{

constexpr size_t kMaxInstances = 4096;
constexpr u32 kSpriteDumpInterval = 8;
constexpr u32 kBGDumpInterval = 16;

// chained XXH64 over a VRAM byte range, wrap-masked like VRAMRead8
u64 HashVRAMRange(const u8* vram, u32 vrammask, u64 hash, u32 addr, u32 size)
{
    addr &= vrammask;
    if (addr + size > vrammask + 1)
    {
        u32 first = vrammask + 1 - addr;
        hash = XXH64(&vram[addr], first, hash);
        return XXH64(&vram[0], size - first, hash);
    }
    return XXH64(&vram[addr], size, hash);
}

// desktop prerender parity: 5-bit channels expand through the 6-bit DS space
// (v5*2, green folds in the palette A1 bit) and quantize as round(v6*255/63)
u32 Expand6To8(u32 v6)
{
    return (v6 * 510 + 63) / 126;
}

u32 Pal555ToRGBA8(u16 entry, bool opaque)
{
    u32 r6 = (entry & 0x1F) << 1;
    u32 g6 = (((entry >> 5) & 0x1F) << 1) + (entry >> 15);
    u32 b6 = ((entry >> 10) & 0x1F) << 1;
    return Expand6To8(r6) | (Expand6To8(g6) << 8) | (Expand6To8(b6) << 16)
        | (opaque ? 0xFF000000u : 0u);
}

u32 Bitmap555ToRGBA8(u16 col)
{
    u32 r6 = (col << 1) & 0x3E;
    u32 g6 = (col >> 4) & 0x3E;
    u32 b6 = (col >> 9) & 0x3E;
    return Expand6To8(r6) | (Expand6To8(g6) << 8) | (Expand6To8(b6) << 16)
        | ((col & 0x8000) ? 0xFF000000u : 0u);
}

// chained XXH64 over the exact OBJ VRAM bytes the sprite decodes from,
// pre-flip; identical to the desktop dumper's SpriteTileHash
u64 SpriteTileHash(const u8* vram, u32 vrammask, int type,
                   int tileOffset, int tileStride, int w, int h)
{
    u64 hash = 0;
    if (type == 2)
    {
        // direct-color bitmap: rows of w*2 bytes at TileStride pitch
        for (int y = 0; y < h; y++)
            hash = HashVRAMRange(vram, vrammask, hash,
                                 (u32)(tileOffset + y * tileStride), (u32)w * 2);
    }
    else
    {
        const u32 tileBytes = (type == 0) ? 32 : 64;
        for (int ty = 0; ty < h / 8; ty++)
            for (int tx = 0; tx < w / 8; tx++)
                hash = HashVRAMRange(vram, vrammask, hash,
                                     (u32)(tileOffset + tx * (int)tileBytes + ty * tileStride),
                                     tileBytes);
    }
    return hash;
}

// hash of the palette range the sprite can address; matches the desktop
// dumper's SpritePalHash
u64 SpritePalHash(const u16* stdPal, const u16* extPal, int type, int palOffset, bool& hasPal)
{
    hasPal = (type != 2);
    if (!hasPal)
        return 0;

    if (type == 0)
    {
        int start = palOffset & 0xF0;
        return XXH64(&stdPal[start], 16 * 2, 0);
    }

    if (palOffset == 0)
        return XXH64(stdPal, 256 * 2, 0);
    return XXH64(&extPal[(palOffset - 1) * 256], 256 * 2, 0);
}

}

void HDPack2D::ProcessFrame(GPU& gpu, HDTexPack* pack)
{
    Instances.clear();
    if (!pack)
        return;

    const bool load = pack->LoadActive() && pack->Has2DEntries();
    const bool dumpEnabled = pack->DumpActive();
    if (!load && !dumpEnabled)
        return;

    FrameCounter++;
    const bool dumpSprites = dumpEnabled && (FrameCounter % kSpriteDumpInterval) == 0;
    const bool dumpBG = dumpEnabled && (FrameCounter % kBGDumpInterval) == 0;
    if (!load && !dumpSprites && !dumpBG)
        return;

    if (dumpSprites || dumpBG)
        WalkBatch++;
    if (dumpSprites)
        CurBitmapKeys.clear();

    for (int num = 0; num < 2; num++)
    {
        const GPU2D::Unit& unit = num ? (const GPU2D::Unit&)gpu.GPU2D_B : gpu.GPU2D_A;
        u32 dispmode = (unit.DispCnt >> 16) & (num ? 0x1 : 0x3);
        if (dispmode != 1)
            continue;

        if (load || dumpSprites)
            WalkSprites(gpu, num, pack, dumpSprites, load);
        if (load || dumpBG)
            WalkBGLayers(gpu, num, pack, dumpBG, load);
    }

    if (dumpSprites)
    {
        std::swap(PrevBitmapKeys, CurBitmapKeys);
        CurBitmapKeys.clear();
    }

    if (Instances.size() > kMaxInstances)
        Instances.resize(kMaxInstances);
}

void HDPack2D::EmitSpriteInstance(const HDTexPackImage* img, int num, u8 flip,
                                  s32 xpos, s32 ypos, int width, int height)
{
    // the renderer draws sprite row r at scanline (ypos + r) & 0xFF, so a
    // sprite near the bottom edge also wraps to the top of the screen
    s32 baseY = ypos & 0xFF;
    const s32 candidates[2] = { baseY, baseY - 256 };
    for (s32 y : candidates)
    {
        if (y + height <= 0 || y >= 192)
            continue;
        if (Instances.size() >= kMaxInstances)
            return;
        HDPack2DInstance inst;
        inst.Image = img;
        inst.Engine = (u8)num;
        inst.RequireMask = 0x90;
        inst.RejectMask = 0;
        inst.Flip = flip;
        inst.X = (s16)xpos;
        inst.Y = (s16)y;
        inst.W = (u16)width;
        inst.H = (u16)height;
        Instances.push_back(inst);
    }
}

void HDPack2D::WalkSprites(GPU& gpu, int num, HDTexPack* pack, bool dump, bool load)
{
    const GPU2D::Unit& unit = num ? (const GPU2D::Unit&)gpu.GPU2D_B : gpu.GPU2D_A;
    if (!(unit.DispCnt & 0x1000))
        return;

    const u16* oam = (const u16*)&gpu.OAM[num ? 0x400 : 0];
    u8* objvram;
    u32 objvrammask;
    unit.GetOBJVRAM(objvram, objvrammask);
    const u16* stdPal = (const u16*)&gpu.Palette[num ? 0x600 : 0x200];
    const u16* extPal = const_cast<GPU2D::Unit&>(unit).GetOBJExtPal();
    const char screen = num ? 'B' : 'A';

    static const u8 spritewidth[16] =
    {
        8, 16, 8, 8,
        16, 32, 8, 8,
        32, 32, 16, 8,
        64, 64, 32, 8
    };
    static const u8 spriteheight[16] =
    {
        8, 8, 16, 8,
        16, 8, 32, 8,
        32, 16, 32, 8,
        64, 32, 64, 8
    };

    for (int i = 0; i < 128; i++)
    {
        const u16* attrib = &oam[i * 4];

        u32 sprtype = (attrib[0] >> 8) & 0x3;
        if (sprtype == 2) // sprite disabled
            continue;

        u32 sprmode = (attrib[0] >> 10) & 0x3;

        // sign-extend the 9-bit X / 8-bit Y fields without shifting other
        // attribute bits into the sign position (undefined behaviour)
        s32 xpos = (s32)(attrib[1] & 0x1FF) - (s32)((attrib[1] & 0x100) << 1);
        s32 ypos = (s32)(attrib[0] & 0xFF) - (s32)((attrib[0] & 0x80) << 1);

        u32 sizeparam = (attrib[0] >> 14) | ((attrib[1] & 0xC000) >> 12);
        s32 width = spritewidth[sizeparam];
        s32 height = spriteheight[sizeparam];
        s32 boundwidth = width;
        s32 boundheight = height;
        if (sprtype == 3)
        {
            boundwidth <<= 1;
            boundheight <<= 1;
        }

        if (xpos <= -boundwidth)
            continue;
        bool yc0 = ((ypos + boundheight) > 0) && (ypos < 192);
        bool yc1 = (((ypos & 0xFF) + boundheight) > 0) && ((ypos & 0xFF) < 192);
        if (!(yc0 || yc1))
            continue;

        if (sprmode == 3)
        {
            if ((unit.DispCnt & 0x60) == 0x60)
                continue;
            if ((attrib[2] >> 12) == 0)
                continue;
        }

        u32 tilenum = attrib[2] & 0x3FF;
        int type, tileOffset, tileStride, palOffset;
        if (sprmode == 3)
        {
            // direct-color bitmap
            type = 2;
            if (unit.DispCnt & (1 << 6))
            {
                tileOffset = (int)(tilenum << (7 + ((unit.DispCnt >> 22) & 0x1)));
                tileStride = width * 2;
            }
            else if (unit.DispCnt & (1 << 5))
            {
                tileOffset = (int)(((tilenum & 0x01F) << 4) + ((tilenum & 0x3E0) << 7));
                tileStride = 256 * 2;
            }
            else
            {
                tileOffset = (int)(((tilenum & 0x00F) << 4) + ((tilenum & 0x3F0) << 7));
                tileStride = 128 * 2;
            }
            palOffset = 1 + (attrib[2] >> 12);
        }
        else
        {
            if (unit.DispCnt & (1 << 4))
            {
                tileOffset = (int)(tilenum << (5 + ((unit.DispCnt >> 20) & 0x3)));
                tileStride = (width >> 3) * 32;
                if (attrib[0] & (1 << 13))
                    tileStride <<= 1;
            }
            else
            {
                tileOffset = (int)(tilenum << 5);
                tileStride = 32 * 32;
            }

            if (attrib[0] & (1 << 13))
            {
                type = 1;
                palOffset = (unit.DispCnt & (1u << 31)) ? (int)(1 + (attrib[2] >> 12)) : 0;
            }
            else
            {
                type = 0;
                palOffset = (int)((attrib[2] >> 12) << 4);
            }
        }

        if (width <= 0 || height <= 0 || width > 64 || height > 64)
            continue;

        u64 tileHash = SpriteTileHash(objvram, objvrammask, type,
                                      tileOffset, tileStride, width, height);
        bool hasPal = false;
        u64 palHash = SpritePalHash(stdPal, extPal, type, palOffset, hasPal);
        const char* bppTag = (type == 0) ? "4" : (type == 1) ? "8" : "bmp";

        if (dump)
        {
            bool stable = true;
            if (type == 2)
            {
                u64 k = tileHash ^ ((u64)width << 32) ^ (u64)height;
                stable = PrevBitmapKeys.count(k) != 0;
                CurBitmapKeys.insert(k);
            }
            if (stable)
            {
                PixelScratch.resize((size_t)width * height);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        u32 pixel;
                        if (type == 0)
                        {
                            u32 addr = (u32)(tileOffset + ((x >> 3) * 32) + ((y >> 3) * tileStride)
                                             + ((x & 0x7) >> 1) + ((y & 0x7) << 2));
                            u8 byte = objvram[addr & objvrammask];
                            int col = (x & 1) ? (byte >> 4) : (byte & 0xF);
                            col += palOffset;
                            pixel = Pal555ToRGBA8(stdPal[col], (col & 0xF) != 0);
                        }
                        else if (type == 1)
                        {
                            u32 addr = (u32)(tileOffset + ((x >> 3) * 64) + ((y >> 3) * tileStride)
                                             + (x & 0x7) + ((y & 0x7) << 3));
                            u8 col = objvram[addr & objvrammask];
                            u16 entry = (palOffset == 0)
                                ? stdPal[col]
                                : extPal[(palOffset - 1) * 256 + col];
                            pixel = Pal555ToRGBA8(entry, col != 0);
                        }
                        else
                        {
                            u32 addr = (u32)(tileOffset + (x * 2) + (y * tileStride)) & objvrammask;
                            u16 col = (u16)(objvram[addr] | (objvram[(addr + 1) & objvrammask] << 8));
                            pixel = Bitmap555ToRGBA8(col);
                        }
                        PixelScratch[(size_t)y * width + x] = pixel;
                    }
                }
                pack->DumpSprite((u32)width, (u32)height, tileHash, palHash, hasPal, bppTag,
                                 PixelScratch.data(), WalkBatch, screen, i, xpos, ypos);
            }
        }

        // replacement v1 skips rotscale sprites and window OBJs
        if (load && !(sprtype & 1) && sprmode != 2)
        {
            const HDTexPackImage* img =
                pack->LookupSprite((u32)width, (u32)height, tileHash, palHash, hasPal, bppTag);
            if (img)
            {
                u8 flip = (u8)(((attrib[1] & (1 << 12)) ? 1 : 0)
                               | ((attrib[1] & (1 << 13)) ? 2 : 0));
                EmitSpriteInstance(img, num, flip, xpos, ypos, width, height);
            }
        }
    }
}

void HDPack2D::WalkBGLayers(GPU& gpu, int num, HDTexPack* pack, bool dump, bool load)
{
    const GPU2D::Unit& unit = num ? (const GPU2D::Unit&)gpu.GPU2D_B : gpu.GPU2D_A;
    u32 dispcnt = unit.DispCnt;
    u32 mode = dispcnt & 0x7;
    const char screen = num ? 'B' : 'A';

    u8* bgvram;
    u32 bgvrammask;
    unit.GetBGVRAM(bgvram, bgvrammask);

    for (int layer = 0; layer < 4; layer++)
    {
        if (!(dispcnt & (0x100u << layer)))
            continue;

        // v1 covers text BGs only; the layer type table mirrors
        // DrawScanlineBGMode in the software renderer
        bool text;
        switch (layer)
        {
        case 0: text = (mode != 6) && !(num == 0 && (dispcnt & 0x8)); break;
        case 1: text = (mode != 6); break;
        case 2: text = (mode == 0 || mode == 1 || mode == 3); break;
        default: text = (mode == 0); break;
        }
        if (!text)
            continue;

        u16 bgcnt = unit.BGCnt[layer];
        u32 tilesetaddr, tilemapaddr;
        const u16* pal;
        if (num)
        {
            tilesetaddr = (u32)(bgcnt & 0x003C) << 12;
            tilemapaddr = (u32)(bgcnt & 0x1F00) << 3;
            pal = (const u16*)&gpu.Palette[0x400];
        }
        else
        {
            tilesetaddr = ((dispcnt & 0x07000000) >> 8) + ((u32)(bgcnt & 0x003C) << 12);
            tilemapaddr = ((dispcnt & 0x38000000) >> 11) + ((u32)(bgcnt & 0x1F00) << 3);
            pal = (const u16*)&gpu.Palette[0];
        }

        const bool eightbpp = (bgcnt & 0x0080) != 0;
        const u32 bpp = eightbpp ? 8 : 4;
        const bool extpal = (dispcnt & 0x40000000) != 0;
        const u32 extpalslot = ((layer < 2) && (bgcnt & 0x2000)) ? (u32)(2 + layer) : (u32)layer;
        const u32 widexmask = (bgcnt & 0x4000) ? 0x100 : 0;
        const int mapTilesX = (bgcnt & 0x4000) ? 64 : 32;
        const int mapTilesY = (bgcnt & 0x8000) ? 64 : 32;

        auto mapEntry = [&](int tx, int ty) -> u16
        {
            u32 yoff = (u32)(ty * 8);
            u32 xoff = (u32)(tx * 8);
            u32 addr = tilemapaddr;
            if (bgcnt & 0x8000)
            {
                addr += (yoff & 0x1F8) << 3;
                if (bgcnt & 0x4000)
                    addr += (yoff & 0x100) << 3;
            }
            else
            {
                addr += (yoff & 0xF8) << 3;
            }
            addr += ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3);
            addr &= bgvrammask & ~1u;
            return (u16)(bgvram[addr] | (bgvram[addr + 1] << 8));
        };

        auto tilePalette = [&](u16 curtile, u64& palHash) -> const u16*
        {
            if (!eightbpp)
            {
                const u16* curpal = pal + ((curtile & 0xF000) >> 8);
                palHash = XXH64(curpal, 16 * 2, 0);
                return curpal;
            }
            if (extpal)
            {
                const u16* curpal =
                    const_cast<GPU2D::Unit&>(unit).GetBGExtPal(extpalslot, curtile >> 12);
                palHash = XXH64(curpal, 256 * 2, 0);
                return curpal;
            }
            palHash = XXH64(pal, 256 * 2, 0);
            return pal;
        };

        auto tileHashOf = [&](u16 curtile) -> u64
        {
            u32 tilenum = curtile & 0x03FF;
            u32 addr = tilesetaddr + (tilenum << (eightbpp ? 6 : 5));
            return HashVRAMRange(bgvram, bgvrammask, 0, addr, eightbpp ? 64 : 32);
        };

        if (dump)
        {
            PixelScratch.resize(64);
            for (int ty = 0; ty < mapTilesY; ty++)
            {
                for (int tx = 0; tx < mapTilesX; tx++)
                {
                    u16 curtile = mapEntry(tx, ty);
                    u64 palHash = 0;
                    const u16* curpal = tilePalette(curtile, palHash);
                    u64 tileHash = tileHashOf(curtile);

                    u32 tilenum = curtile & 0x03FF;
                    u32 base = tilesetaddr + (tilenum << (eightbpp ? 6 : 5));
                    for (int y = 0; y < 8; y++)
                    {
                        for (int x = 0; x < 8; x++)
                        {
                            int col;
                            if (eightbpp)
                            {
                                col = bgvram[(base + (u32)(y * 8 + x)) & bgvrammask];
                            }
                            else
                            {
                                u8 byte = bgvram[(base + (u32)(y * 4 + (x >> 1))) & bgvrammask];
                                col = (x & 1) ? (byte >> 4) : (byte & 0xF);
                            }
                            PixelScratch[(size_t)y * 8 + x] =
                                Pal555ToRGBA8(curpal[col], col != 0);
                        }
                    }
                    pack->DumpBGTile(tileHash, palHash, true, bpp, PixelScratch.data(),
                                     WalkBatch, screen, layer, tx, ty);
                }
            }
        }

        if (load)
        {
            u16 xoff = unit.BGXPos[layer];
            u16 yoff = unit.BGYPos[layer];
            const int fineX = xoff & 7;
            const int fineY = yoff & 7;
            const int cols = fineX ? 33 : 32;
            const int rows = fineY ? 25 : 24;

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    if (Instances.size() >= kMaxInstances)
                        return;

                    int tx = (xoff >> 3) + c;
                    int ty = (yoff >> 3) + r;
                    u16 curtile = mapEntry(tx, ty);

                    u64 palHash = 0;
                    tilePalette(curtile, palHash);
                    u64 tileHash = tileHashOf(curtile);

                    const HDTexPackImage* img = pack->LookupBGTile(tileHash, palHash, true, bpp);
                    if (!img)
                        continue;

                    HDPack2DInstance inst;
                    inst.Image = img;
                    inst.Engine = (u8)num;
                    inst.RequireMask = (u8)(1u << layer);
                    inst.RejectMask = 0x90;
                    inst.Flip = (u8)(((curtile & 0x0400) ? 1 : 0) | ((curtile & 0x0800) ? 2 : 0));
                    inst.X = (s16)(c * 8 - fineX);
                    inst.Y = (s16)(r * 8 - fineY);
                    inst.W = 8;
                    inst.H = 8;
                    Instances.push_back(inst);
                }
            }
        }
    }
}

}
