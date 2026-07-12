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

#include "HDTexPack.h"
#include "Platform.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

#define XXH_STATIC_LINKING_ONLY
#include "xxhash/xxhash.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace melonDS
{

namespace fs = std::filesystem;

namespace
{

struct KeyFields
{
    u32 Width, Height;
    u64 Hash1, Hash2;   // texel/tile hash; palette hash (0 when absent/wildcard)
    u32 Discriminator;  // tex: fmt 1..7; sprite: 4, 8, or 0xB ("bmp")
    u32 HasPal;
};

u64 MapKey(u32 width, u32 height, u64 hash1, u64 hash2, u32 disc, bool hasPal)
{
    KeyFields f = { width, height, hash1, hash2, disc, hasPal ? 1u : 0u };
    return XXH64(&f, sizeof(f), 0x484454455850414BULL); // "HDTEXPAK"
}

// DumpedKeys spans all asset kinds while MapKey does not encode the kind;
// an 8x8 4bpp sprite and BG tile with identical bytes must still dump both
u64 DumpKey(u64 mapKey, const char* kind)
{
    return XXH64(kind, strlen(kind), mapKey);
}

bool ParseHash16(const std::string& s, u64& out, bool& wildcard, bool& none)
{
    wildcard = false; none = false;
    if (s == "$") { wildcard = true; return true; }
    if (s == "none") { none = true; return true; }
    if (s.size() != 16) return false;
    out = 0;
    for (char c : s)
    {
        u32 v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else return false;
        out = (out << 4) | v;
    }
    return true;
}

std::string Hash16(u64 h)
{
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return buf;
}

// split "tex1_64x64_aabb..ff_cc..00_5" stem into underscore-separated parts
std::vector<std::string> SplitStem(const std::string& stem)
{
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= stem.size())
    {
        size_t end = stem.find('_', start);
        if (end == std::string::npos) { parts.push_back(stem.substr(start)); break; }
        parts.push_back(stem.substr(start, end - start));
        start = end + 1;
    }
    return parts;
}

}

u32 HDTexPack::RGB6A5ToRGBA8(u32 texel)
{
    u32 r = texel & 0x3F, g = (texel >> 8) & 0x3F, b = (texel >> 16) & 0x3F, a = (texel >> 24) & 0x1F;
    r = (r << 2) | (r >> 4); g = (g << 2) | (g >> 4); b = (b << 2) | (b >> 4);
    a = (a << 3) | (a >> 2);
    return r | (g << 8) | (b << 16) | (a << 24);
}

u32 HDTexPack::RGBA8ToRGB6A5(u32 pixel)
{
    u32 r = (pixel & 0xFF) >> 2, g = ((pixel >> 8) & 0xFF) >> 2,
        b = ((pixel >> 16) & 0xFF) >> 2, a = ((pixel >> 24) & 0xFF) >> 3;
    return r | (g << 8) | (b << 16) | (a << 24);
}

HDTexPack::HDTexPack(const std::string& packDir, const std::string& dumpDir,
                     bool loadEnabled, bool dumpEnabled)
    : PackDir(packDir), DumpDir(dumpDir), LoadEnabled(loadEnabled), DumpEnabled(dumpEnabled)
{
    if (LoadEnabled)
    {
        LoadDir(PackDir + "/textures", "tex1");
        LoadDir(PackDir + "/sprites", "obj1");
        LoadDir(PackDir + "/bgtiles", "bg1");
        if (EntryCount > 0)
            Platform::Log(Platform::LogLevel::Info,
                          "HDTexPack: loaded %u entries from %s (scale %ux)\n",
                          EntryCount, PackDir.c_str(), PackScale);
    }
    if (DumpEnabled)
    {
        std::error_code ec;
        fs::create_directories(fs::u8path(DumpDir + "/textures"), ec);
        fs::create_directories(fs::u8path(DumpDir + "/sprites"), ec);
        fs::create_directories(fs::u8path(DumpDir + "/bgtiles"), ec);
        Platform::Log(Platform::LogLevel::Info,
                      "HDTexPack: dumping to %s\n", DumpDir.c_str());
    }
}

void HDTexPack::LoadDir(const std::string& dir, const char* kind)
{
    std::error_code ec;
    if (!fs::is_directory(fs::u8path(dir), ec))
        return;

    for (auto it = fs::recursive_directory_iterator(fs::u8path(dir), ec);
         it != fs::recursive_directory_iterator(); it.increment(ec))
    {
        if (ec) break;
        if (!it->is_regular_file(ec)) continue;
        const fs::path& p = it->path();
        if (p.extension() != ".png" && p.extension() != ".PNG") continue;
        std::string name = p.stem().u8string();
        if (name.rfind(kind, 0) != 0) continue;
        if (!AddEntry(p.u8string(), name, kind))
            Platform::Log(Platform::LogLevel::Warn,
                          "HDTexPack: skipping %s (bad name or size)\n", name.c_str());
    }
}

bool HDTexPack::AddEntry(const std::string& path, const std::string& name, const char* kind)
{
    // <kind>_<W>x<H>_<hash16>_<palhash16|none|$>_<disc>
    auto parts = SplitStem(name);
    if (parts.size() != 5 || parts[0] != kind) return false;

    u32 w = 0, h = 0;
    if (sscanf(parts[1].c_str(), "%ux%u", &w, &h) != 2 || !w || !h) return false;

    u64 hash1 = 0, palHash = 0;
    bool wc1 = false, none1 = false, wcPal = false, nonePal = false;
    if (!ParseHash16(parts[2], hash1, wc1, none1) || wc1 || none1) return false;
    if (!ParseHash16(parts[3], palHash, wcPal, nonePal)) return false;

    u32 disc;
    bool isTex = !strcmp(kind, "tex1");
    bool isBG = !strcmp(kind, "bg1");
    if (isTex)
    {
        if (parts[4].size() != 1 || parts[4][0] < '1' || parts[4][0] > '7') return false;
        disc = parts[4][0] - '0';
    }
    else if (isBG)
    {
        if (w != 8 || h != 8) return false;
        if (parts[4] == "4") disc = 4;
        else if (parts[4] == "8") disc = 8;
        else return false;
    }
    else
    {
        if (parts[4] == "4") disc = 4;
        else if (parts[4] == "8") disc = 8;
        else if (parts[4] == "bmp") disc = 0xB;
        else return false;
    }

    int pw = 0, ph = 0, pc = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &pw, &ph, &pc, 4);
    if (!pixels) return false;

    // scale must be a positive integer and identical on both axes
    if (pw <= 0 || ph <= 0 || pw % (int)w || ph % (int)h || pw / (int)w != ph / (int)h)
    {
        stbi_image_free(pixels);
        return false;
    }
    u32 scale = pw / w;

    // renderers store replacements at most at 8x; larger factors would only
    // produce deterministic allocation failures (a legal 1024x1024 texture
    // at 8x is already a 256 MiB image)
    if (scale > 8)
    {
        stbi_image_free(pixels);
        Platform::Log(Platform::LogLevel::Warn,
                      "HDTexPack: %s has scale %ux, maximum is 8x — skipped\n",
                      name.c_str(), scale);
        return false;
    }

    if (EntryCount > 0 && scale != PackScale)
    {
        stbi_image_free(pixels);
        Platform::Log(Platform::LogLevel::Warn,
                      "HDTexPack: %s has scale %ux, pack is %ux — skipped\n",
                      name.c_str(), scale, PackScale);
        return false;
    }
    PackScale = scale;

    HDTexPackImage img;
    img.Width = pw; img.Height = ph; img.Scale = scale;
    img.RGBA.resize((size_t)pw * ph);
    memcpy(img.RGBA.data(), pixels, (size_t)pw * ph * 4);
    stbi_image_free(pixels);

    bool hasPal = !nonePal;
    u64 key = MapKey(w, h, hash1, wcPal ? 0 : palHash, disc, hasPal && !wcPal);
    auto& exact = isTex ? TexEntries : isBG ? BGEntries : SpriteEntries;
    auto& wild = isTex ? TexWildcard : isBG ? BGWildcard : SpriteWildcard;
    (wcPal ? wild : exact)[key] = std::move(img);
    EntryCount++;
    return true;
}

const HDTexPackImage* HDTexPack::Find(const std::unordered_map<u64, HDTexPackImage>& exact,
                                      const std::unordered_map<u64, HDTexPackImage>& wildcard,
                                      u64 exactKey, u64 wildcardKey) const
{
    auto it = exact.find(exactKey);
    if (it != exact.end()) return &it->second;
    auto wit = wildcard.find(wildcardKey);
    if (wit != wildcard.end()) return &wit->second;
    return nullptr;
}

const HDTexPackImage* HDTexPack::LookupTexture(u32 width, u32 height, u64 texHash,
                                               u64 palHash, bool hasPal, u32 fmt) const
{
    if (!LoadActive()) return nullptr;
    return Find(TexEntries, TexWildcard,
                MapKey(width, height, texHash, hasPal ? palHash : 0, fmt, hasPal),
                MapKey(width, height, texHash, 0, fmt, false));
}

const HDTexPackImage* HDTexPack::LookupSprite(u32 width, u32 height, u64 tileHash,
                                              u64 palHash, bool hasPal, const char* bppTag) const
{
    if (!LoadActive()) return nullptr;
    u32 disc = !strcmp(bppTag, "bmp") ? 0xB : (u32)atoi(bppTag);
    return Find(SpriteEntries, SpriteWildcard,
                MapKey(width, height, tileHash, hasPal ? palHash : 0, disc, hasPal),
                MapKey(width, height, tileHash, 0, disc, false));
}

const HDTexPackImage* HDTexPack::LookupBGTile(u64 tileHash, u64 palHash, bool hasPal, u32 bpp) const
{
    if (!LoadActive()) return nullptr;
    return Find(BGEntries, BGWildcard,
                MapKey(8, 8, tileHash, hasPal ? palHash : 0, bpp, hasPal),
                MapKey(8, 8, tileHash, 0, bpp, false));
}

void HDTexPack::WriteDumpPNG(const char* subdir, const std::string& name,
                             u32 width, u32 height, const u32* rgba8)
{
    std::string path = DumpDir + "/" + subdir + "/" + name + ".png";
    std::error_code ec;
    if (fs::exists(fs::u8path(path), ec))
        return;
    stbi_write_png(path.c_str(), (int)width, (int)height, 4, rgba8, (int)width * 4);
}

void HDTexPack::AppendManifest(const char* subdir, const std::string& line)
{
    std::string path = DumpDir + "/" + subdir + "/manifest.jsonl";
    FILE* f = fopen(path.c_str(), "ab");
    if (!f) return;
    fwrite(line.data(), 1, line.size(), f);
    fputc('\n', f);
    fclose(f);
}

void HDTexPack::DumpTexture(u32 width, u32 height, u64 texHash,
                            u64 palHash, bool hasPal, u32 fmt, const u32* rgb6a5,
                            u32 scale)
{
    if (!DumpEnabled) return;
    if (scale < 1) scale = 1;

    u64 key = MapKey(width, height, texHash, hasPal ? palHash : 0, fmt, hasPal);
    if (!DumpedKeys.insert(DumpKey(key, "tex1")).second) return;

    std::string palStr = hasPal ? Hash16(palHash) : "none";
    char nameBuf[96];
    snprintf(nameBuf, sizeof(nameBuf), "tex1_%ux%u_%s_%s_%u",
             width, height, Hash16(texHash).c_str(), palStr.c_str(), fmt);

    const u32 pw = width * scale, ph = height * scale;
    std::vector<u32> rgba((size_t)pw * ph);
    for (size_t i = 0; i < rgba.size(); i++)
        rgba[i] = RGB6A5ToRGBA8(rgb6a5[i]);

    WriteDumpPNG("textures", nameBuf, pw, ph, rgba.data());

    char line[256];
    snprintf(line, sizeof(line),
             "{\"kind\":\"tex1\",\"w\":%u,\"h\":%u,\"texhash\":\"%s\",\"palhash\":%s%s%s,\"fmt\":%u,\"scale\":%u}",
             width, height, Hash16(texHash).c_str(),
             hasPal ? "\"" : "", hasPal ? palStr.c_str() : "null", hasPal ? "\"" : "", fmt, scale);
    AppendManifest("textures", line);
}

void HDTexPack::DumpSprite(u32 width, u32 height, u64 tileHash,
                           u64 palHash, bool hasPal, const char* bppTag, const u32* rgba8,
                           u32 frame, char screen, int oamSlot, int x, int y)
{
    if (!DumpEnabled) return;

    // fully-transparent sprites (e.g. zeroed OAM pointing at empty tiles)
    // produce useless art and drown the manifest — skip them entirely
    bool anyOpaque = false;
    for (size_t i = 0, n = (size_t)width * height; i < n; i++)
        if (rgba8[i] & 0xFF000000u) { anyOpaque = true; break; }
    if (!anyOpaque)
        return;

    u32 disc = !strcmp(bppTag, "bmp") ? 0xB : (u32)atoi(bppTag);
    u64 key = MapKey(width, height, tileHash, hasPal ? palHash : 0, disc, hasPal);
    std::string palStr = hasPal ? Hash16(palHash) : "none";

    if (DumpedKeys.insert(DumpKey(key, "obj1")).second)
    {
        char nameBuf[96];
        snprintf(nameBuf, sizeof(nameBuf), "obj1_%ux%u_%s_%s_%s",
                 width, height, Hash16(tileHash).c_str(), palStr.c_str(), bppTag);
        WriteDumpPNG("sprites", nameBuf, width, height, rgba8);
    }

    // one manifest line per distinct (key, screen, position) — the stitcher
    // clusters with these; repeats across batches add nothing
    u64 inst[4] = { key, (u64)(u8)screen, (u64)(u32)x, (u64)(u32)y };
    if (!LoggedSpriteInstances.insert(XXH64(inst, sizeof(inst), 0)).second)
        return;

    char line[320];
    snprintf(line, sizeof(line),
             "{\"kind\":\"obj1\",\"w\":%u,\"h\":%u,\"tilehash\":\"%s\",\"palhash\":%s%s%s,"
             "\"bpp\":\"%s\",\"frame\":%u,\"screen\":\"%c\",\"oam\":%d,\"x\":%d,\"y\":%d}",
             width, height, Hash16(tileHash).c_str(),
             hasPal ? "\"" : "", hasPal ? palStr.c_str() : "null", hasPal ? "\"" : "",
             bppTag, frame, screen, oamSlot, x, y);
    AppendManifest("sprites", line);
}

void HDTexPack::DumpBGTile(u64 tileHash, u64 palHash, bool hasPal, u32 bpp, const u32* rgba8,
                           u32 frame, char screen, int layer, int x, int y)
{
    if (!DumpEnabled) return;

    bool anyOpaque = false;
    for (size_t i = 0; i < 64; i++)
        if (rgba8[i] & 0xFF000000u) { anyOpaque = true; break; }
    if (!anyOpaque)
        return;

    u64 key = MapKey(8, 8, tileHash, hasPal ? palHash : 0, bpp, hasPal);
    std::string palStr = hasPal ? Hash16(palHash) : "none";

    if (DumpedKeys.insert(DumpKey(key, "bg1")).second)
    {
        char nameBuf[96];
        snprintf(nameBuf, sizeof(nameBuf), "bg1_8x8_%s_%s_%u",
                 Hash16(tileHash).c_str(), palStr.c_str(), bpp);
        WriteDumpPNG("bgtiles", nameBuf, 8, 8, rgba8);
    }

    // one manifest line per distinct (key, screen, layer, tilemap position)
    u64 inst[4] = { key, ((u64)(u8)screen << 8) | (u64)(u8)layer, (u64)(u32)x, (u64)(u32)y };
    if (!LoggedSpriteInstances.insert(XXH64(inst, sizeof(inst), 1)).second)
        return;

    char line[320];
    snprintf(line, sizeof(line),
             "{\"kind\":\"bg1\",\"w\":8,\"h\":8,\"tilehash\":\"%s\",\"palhash\":%s%s%s,"
             "\"bpp\":\"%u\",\"frame\":%u,\"screen\":\"%c\",\"layer\":%d,\"x\":%d,\"y\":%d}",
             Hash16(tileHash).c_str(),
             hasPal ? "\"" : "", hasPal ? palStr.c_str() : "null", hasPal ? "\"" : "",
             bpp, frame, screen, layer, x, y);
    AppendManifest("bgtiles", line);
}

}
