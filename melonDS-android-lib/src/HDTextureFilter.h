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

#ifndef HDTEXTUREFILTER_H
#define HDTEXTUREFILTER_H

#include "types.h"

#include <vector>

namespace melonDS
{

// CPU upscaling filter library operating on RGB6A5 texel buffers
// (r6 | g6 << 8 | b6 << 16 | a5 << 24, one texel per u32).
//
// Filter modes:
//   0  nearest (accurate storage scaling, no filtering)
//   1  linear
//   2  smoothstep
//   3  Scale2x
//   4  HQ2x lite
//   5  2xSaI lite
//   6  SuperEagle lite
//   7  MMPX lite
//   8  Anime4K lite
//   9  Super2xSaI strong
//   10 SuperEagle smooth
//   11 crisp gradient
//   12 crisp edge AA
//   13 ScaleFX (faithful multi-pass whole-image port)
namespace HDTextureFilter
{

u32 ClampScale(int scale);
int ClampMode(int mode);

// Faithful whole-image translation of the libretro scalefx 5-pass pipeline,
// run at its native 3x (9x for scale >= 6) and center-resampled to scale.
void ScaleFXImage(const u32* src, u32 width, u32 height, u32 scale, std::vector<u32>& dst);

// Upscales src (width x height) to (width*scale x height*scale) using the
// requested filter mode. Mode 0 is an exact nearest resample.
void UpscaleTexture(const u32* src, u32 width, u32 height, u32 scale, int mode, std::vector<u32>& dst);

}

}

#endif
