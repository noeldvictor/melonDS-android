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

#ifndef HDTEXPACK_H
#define HDTEXPACK_H

#include "types.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace melonDS
{

// HD texture pack repository: content-hash keyed replacement images plus
// artist-facing dumping. The identity scheme and file layout are shared
// with the desktop tooling; filenames are the database:
//   tex1_<W>x<H>_<texhash16>_<palhash16|none|$>_<fmt>.png
//   obj1_<W>x<H>_<tilehash16>_<palhash16|none|$>_<4|8|bmp>.png
//   bg1_8x8_<tilehash16>_<palhash16|none|$>_<4|8>.png
// All methods are intended for use from the render thread only.

struct HDTexPackImage
{
    u32 Width = 0, Height = 0;   // pixel size of the replacement image
    u32 Scale = 1;               // integer factor vs the native asset
    std::vector<u32> RGBA;       // packed r | g<<8 | b<<16 | a<<24, 8-bit channels
};

class HDTexPack
{
public:
    // packDir: TexturePacks/<GAMECODE> root holding textures/ and sprites/.
    // dumpDir: Dump/<GAMECODE> root receiving textures/ and sprites/ dumps.
    HDTexPack(const std::string& packDir, const std::string& dumpDir,
              bool loadEnabled, bool dumpEnabled);

    bool LoadActive() const { return LoadEnabled && EntryCount > 0; }
    bool DumpActive() const { return DumpEnabled; }
    u32 Scale() const { return PackScale; }
    bool Has2DEntries() const
    {
        return !SpriteEntries.empty() || !SpriteWildcard.empty()
            || !BGEntries.empty() || !BGWildcard.empty();
    }
    bool Has3DEntries() const
    {
        return !TexEntries.empty() || !TexWildcard.empty();
    }

    // 3D textures. hasPal is false for fmt 7 (direct bitmap).
    const HDTexPackImage* LookupTexture(u32 width, u32 height, u64 texHash,
                                        u64 palHash, bool hasPal, u32 fmt) const;
    // rgb6a5 spans width*scale x height*scale texels; scale > 1 dumps a
    // pre-filtered image that loads back as a scaled entry (scale 1 = plain
    // native-res dump).
    void DumpTexture(u32 width, u32 height, u64 texHash,
                     u64 palHash, bool hasPal, u32 fmt, const u32* rgb6a5,
                     u32 scale = 1);

    // 2D OBJ sprites. bppTag: "4", "8" or "bmp".
    const HDTexPackImage* LookupSprite(u32 width, u32 height, u64 tileHash,
                                       u64 palHash, bool hasPal, const char* bppTag) const;
    // rgba8: assembled sprite pixels, 8-bit channels.
    void DumpSprite(u32 width, u32 height, u64 tileHash,
                    u64 palHash, bool hasPal, const char* bppTag, const u32* rgba8,
                    u32 frame, char screen, int oamSlot, int x, int y);

    // 2D BG tiles, always 8x8, pre-flip. bpp: 4 or 8.
    const HDTexPackImage* LookupBGTile(u64 tileHash, u64 palHash, bool hasPal, u32 bpp) const;
    // rgba8: decoded unflipped tile pixels; x/y are tilemap coordinates.
    void DumpBGTile(u64 tileHash, u64 palHash, bool hasPal, u32 bpp, const u32* rgba8,
                    u32 frame, char screen, int layer, int x, int y);

    static u32 RGB6A5ToRGBA8(u32 texel);
    static u32 RGBA8ToRGB6A5(u32 pixel);

private:
    void LoadDir(const std::string& dir, const char* kind);
    bool AddEntry(const std::string& path, const std::string& name, const char* kind);
    const HDTexPackImage* Find(const std::unordered_map<u64, HDTexPackImage>& exact,
                               const std::unordered_map<u64, HDTexPackImage>& wildcard,
                               u64 exactKey, u64 wildcardKey) const;
    void WriteDumpPNG(const char* subdir, const std::string& name,
                      u32 width, u32 height, const u32* rgba8);
    void AppendManifest(const char* subdir, const std::string& line);

    std::string PackDir, DumpDir;
    bool LoadEnabled = false, DumpEnabled = false;
    u32 PackScale = 1;
    u32 EntryCount = 0;

    // keyed by XXH64 over the canonical key fields; wildcard maps ignore the palette hash
    std::unordered_map<u64, HDTexPackImage> TexEntries, TexWildcard;
    std::unordered_map<u64, HDTexPackImage> SpriteEntries, SpriteWildcard;
    std::unordered_map<u64, HDTexPackImage> BGEntries, BGWildcard;

    std::unordered_set<u64> DumpedKeys;
    std::unordered_set<u64> LoggedSpriteInstances;
};

}

#endif
