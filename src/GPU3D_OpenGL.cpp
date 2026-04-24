/*
    Copyright 2016-2025 melonDS team

    This file is part of melonDS.

    melonDS is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    melonDS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with melonDS. If not, see http://www.gnu.org/licenses/.
*/

#include "GPU3D_OpenGL.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include "NDS.h"
#include "GPU.h"
#include "GPU3D_AcceleratedFrontend.h"
#include "GPU3D_OpenGL_shaders.h"

namespace melonDS
{

namespace
{
u32 ConvertBgraToRgba8(u32 packedColor)
{
    const u32 rb = packedColor & 0x00FF00FFu;
    const u32 g = packedColor & 0x0000FF00u;
    const u32 a = packedColor & 0xFF000000u;
    return ((rb & 0x000000FFu) << 16) | g | ((rb & 0x00FF0000u) >> 16) | a;
}

u32 PackOpenGlAttrToLogical(u32 packedColor)
{
    const u32 polyIdByte = packedColor & 0xFFu;
    const u32 edgeByte = (packedColor >> 8) & 0xFFu;
    const u32 fogByte = (packedColor >> 16) & 0xFFu;

    const u32 polyId = ((polyIdByte * 63u) + 127u) / 255u;
    u32 attr = (polyId & 0x3Fu) << 24u;
    if (fogByte >= 0x80u)
        attr |= 1u << 15u;
    if (edgeByte >= 0x80u)
    {
        attr |= 0xFu;
        attr |= 0x10u << 8u;
    }
    return attr;
}
}

bool GLRenderer::BuildRenderShader(u32 flags, const std::string& vs, const std::string& fs)
{
    char shadername[32];
    snprintf(shadername, sizeof(shadername), "RenderShader%02X", flags);

    int headerlen = strlen(kShaderHeader);

    std::string vsbuf;
    vsbuf += kShaderHeader;
    vsbuf += kRenderVSCommon;
    vsbuf += vs;

    std::string fsbuf;
    fsbuf += kShaderHeader;
    fsbuf += kRenderFSCommon;
    fsbuf += fs;

    GLuint prog;
    bool ret = OpenGL::CompileVertexFragmentProgram(prog,
        vsbuf, fsbuf,
        shadername,
        {{"vPosition", 0}, {"vColor", 1}, {"vTexcoord", 2}, {"vPolygonAttr", 3}},
        {{"oColor", 0}, {"oAttr", 1}});

    if (!ret) return false;

    GLint uni_id = glGetUniformBlockIndex(prog, "uConfig");
    glUniformBlockBinding(prog, uni_id, 0);

    glUseProgram(prog);

    uni_id = glGetUniformLocation(prog, "TexMem");
    glUniform1i(uni_id, 0);
    uni_id = glGetUniformLocation(prog, "TexPalMem");
    glUniform1i(uni_id, 1);

    RenderShader[flags] = prog;

    return true;
}

void GLRenderer::UseRenderShader(u32 flags)
{
    if (CurShaderID == flags) return;
    glUseProgram(RenderShader[flags]);
    CurShaderID = flags;
}

void SetupDefaultTexParams(GLuint tex)
{
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

GLRenderer::GLRenderer(GLCompositor&& compositor) noexcept :
    Renderer3D(true),
    CurGLCompositor(std::move(compositor))
{
    // GLRenderer::New() will be used to actually initialize the renderer;
    // The various glDelete* functions silently ignore invalid IDs,
    // so we can just let the destructor clean up a half-initialized renderer.
}

std::unique_ptr<GLRenderer> GLRenderer::New() noexcept
{
    assert(glEnable != nullptr);

    std::optional<GLCompositor> compositor =  GLCompositor::New();
    if (!compositor)
        return nullptr;

    // Will be returned if the initialization succeeds,
    // or cleaned up via RAII if it fails.
    std::unique_ptr<GLRenderer> result = std::unique_ptr<GLRenderer>(new GLRenderer(std::move(*compositor)));
    compositor = std::nullopt;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);

    glDepthRangef(0.0f, 1.0f);
    glClearDepthf(1.0f);

    if (!OpenGL::CompileVertexFragmentProgram(result->ClearShaderPlain,
            kClearVS, kClearFS,
            "ClearShader",
            {{"vPosition", 0}},
            {{"oColor", 0}, {"oAttr", 1}}))
        return nullptr;

    result->ClearUniformLoc[0] = glGetUniformLocation(result->ClearShaderPlain, "uColor");
    result->ClearUniformLoc[1] = glGetUniformLocation(result->ClearShaderPlain, "uDepth");
    result->ClearUniformLoc[2] = glGetUniformLocation(result->ClearShaderPlain, "uOpaquePolyID");
    result->ClearUniformLoc[3] = glGetUniformLocation(result->ClearShaderPlain, "uFogFlag");

    memset(result->RenderShader, 0, sizeof(RenderShader));

    if (!result->BuildRenderShader(0, kRenderVS_Z, kRenderFS_ZO))
        return nullptr;

    if (!result->BuildRenderShader(RenderFlag_WBuffer, kRenderVS_W, kRenderFS_WO))
        return nullptr;

    if (!result->BuildRenderShader(RenderFlag_Edge, kRenderVS_Z, kRenderFS_ZE))
        return nullptr;

    if (!result->BuildRenderShader(RenderFlag_Edge | RenderFlag_WBuffer, kRenderVS_W, kRenderFS_WE))
        return nullptr;

    if (!result->BuildRenderShader(RenderFlag_Trans, kRenderVS_Z, kRenderFS_ZT))
        return nullptr;

    if (!result->BuildRenderShader(RenderFlag_Trans | RenderFlag_WBuffer, kRenderVS_W, kRenderFS_WT))
        return nullptr;

    if (!result->BuildRenderShader(RenderFlag_ShadowMask, kRenderVS_Z, kRenderFS_ZSM))
        return nullptr;

    if (!result->BuildRenderShader(RenderFlag_ShadowMask | RenderFlag_WBuffer, kRenderVS_W, kRenderFS_WSM))
        return nullptr;

    if (!OpenGL::CompileVertexFragmentProgram(result->FinalPassEdgeShader,
            kFinalPassVS, kFinalPassEdgeFS,
            "FinalPassEdgeShader",
            {{"vPosition", 0}},
            {{"oColor", 0}}))
        return nullptr;
    if (!OpenGL::CompileVertexFragmentProgram(result->FinalPassFogShader,
            kFinalPassVS, kFinalPassFogFS,
            "FinalPassFogShader",
            {{"vPosition", 0}},
            {{"oColor", 0}}))
        return nullptr;

    GLuint uni_id = glGetUniformBlockIndex(result->FinalPassEdgeShader, "uConfig");
    glUniformBlockBinding(result->FinalPassEdgeShader, uni_id, 0);

    glUseProgram(result->FinalPassEdgeShader);
    uni_id = glGetUniformLocation(result->FinalPassEdgeShader, "DepthBuffer");
    glUniform1i(uni_id, 0);
    uni_id = glGetUniformLocation(result->FinalPassEdgeShader, "AttrBuffer");
    glUniform1i(uni_id, 1);

    uni_id = glGetUniformBlockIndex(result->FinalPassFogShader, "uConfig");
    glUniformBlockBinding(result->FinalPassFogShader, uni_id, 0);

    glUseProgram(result->FinalPassFogShader);
    uni_id = glGetUniformLocation(result->FinalPassFogShader, "DepthBuffer");
    glUniform1i(uni_id, 0);
    uni_id = glGetUniformLocation(result->FinalPassFogShader, "AttrBuffer");
    glUniform1i(uni_id, 1);


    memset(&result->ShaderConfig, 0, sizeof(ShaderConfig));

    glGenBuffers(1, &result->ShaderConfigUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, result->ShaderConfigUBO);
    static_assert((sizeof(ShaderConfig) & 15) == 0);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(ShaderConfig), &result->ShaderConfig, GL_STATIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, result->ShaderConfigUBO);


    float clearvtx[6*2] =
    {
        -1.0, -1.0,
        1.0, 1.0,
        -1.0, 1.0,

        -1.0, -1.0,
        1.0, -1.0,
        1.0, 1.0
    };

    glGenBuffers(1, &result->ClearVertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, result->ClearVertexBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(clearvtx), clearvtx, GL_STATIC_DRAW);

    glGenVertexArrays(1, &result->ClearVertexArrayID);
    glBindVertexArray(result->ClearVertexArrayID);
    glEnableVertexAttribArray(0); // position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)(0));


    glGenBuffers(1, &result->VertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, result->VertexBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexBuffer), nullptr, GL_DYNAMIC_DRAW);

    glGenVertexArrays(1, &result->VertexArrayID);
    glBindVertexArray(result->VertexArrayID);
    glEnableVertexAttribArray(0); // position
    glVertexAttribIPointer(0, 4, GL_UNSIGNED_SHORT, 7*4, (void*)(0));
    glEnableVertexAttribArray(1); // color
    glVertexAttribIPointer(1, 4, GL_UNSIGNED_BYTE, 7*4, (void*)(2*4));
    glEnableVertexAttribArray(2); // texcoords
    glVertexAttribIPointer(2, 2, GL_SHORT, 7*4, (void*)(3*4));
    glEnableVertexAttribArray(3); // attrib
    glVertexAttribIPointer(3, 3, GL_UNSIGNED_INT, 7*4, (void*)(4*4));

    glGenBuffers(1, &result->IndexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, result->IndexBufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(IndexBuffer), nullptr, GL_DYNAMIC_DRAW);

    glGenFramebuffers(1, &result->MainFramebuffer);
    glGenFramebuffers(1, &result->DownscaleFramebuffer);

    // color buffers
    glGenTextures(1, &result->ColorBufferTex);
    SetupDefaultTexParams(result->ColorBufferTex);

    // depth/stencil buffer
    glGenTextures(1, &result->DepthBufferTex);
    SetupDefaultTexParams(result->DepthBufferTex);

    // attribute buffer
    // R: opaque polyID (for edgemarking)
    // G: edge flag
    // B: fog flag
    glGenTextures(1, &result->AttrBufferTex);
    SetupDefaultTexParams(result->AttrBufferTex);

    // downscale framebuffer for display capture (always 256x192)
    glGenTextures(1, &result->DownScaleBufferTex);
    SetupDefaultTexParams(result->DownScaleBufferTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 192, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);

    glGenBuffers(1, &result->PixelbufferID);

    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &result->TexMemID);
    glBindTexture(GL_TEXTURE_2D, result->TexMemID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 1024, 512, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, NULL);

    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &result->TexPalMemID);
    glBindTexture(GL_TEXTURE_2D, result->TexPalMemID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB5_A1, 1024, 48, 0, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, NULL);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return result;
}

GLRenderer::~GLRenderer()
{
    assert(glDeleteTextures != nullptr);

    glDeleteTextures(1, &TexMemID);
    glDeleteTextures(1, &TexPalMemID);

    glDeleteFramebuffers(1, &MainFramebuffer);
    glDeleteFramebuffers(1, &DownscaleFramebuffer);
    glDeleteTextures(1, &ColorBufferTex);
    glDeleteTextures(1, &DepthBufferTex);
    glDeleteTextures(1, &AttrBufferTex);
    glDeleteTextures(1, &DownScaleBufferTex);

    glDeleteVertexArrays(1, &VertexArrayID);
    glDeleteBuffers(1, &VertexBufferID);
    glDeleteVertexArrays(1, &ClearVertexArrayID);
    glDeleteBuffers(1, &ClearVertexBufferID);

    glDeleteBuffers(1, &ShaderConfigUBO);

    for (int i = 0; i < 16; i++)
    {
        if (!RenderShader[i]) continue;
        glDeleteProgram(RenderShader[i]);
    }
}

void GLRenderer::Reset(GPU& gpu)
{
    // This is where the compositor's Reset() method would be called,
    // except there's no such method right now.
}

void GLRenderer::SetBetterPolygons(bool betterpolygons) noexcept
{
    SetRenderSettings(betterpolygons, ScaleFactor);
}

void GLRenderer::SetScaleFactor(int scale) noexcept
{
    SetRenderSettings(BetterPolygons, scale);
}

void GLRenderer::SetCoverageFixSettings(
    bool enabled,
    float coveragePx,
    float depthBias,
    bool applyRepeat,
    bool applyClamp,
    bool debug3dClearMagenta) noexcept
{
    CoverageFixEnabled = enabled;
    CoverageFixPx = std::clamp(coveragePx, 0.0f, 2.0f);
    CoverageFixDepthBias = std::clamp(depthBias, 0.0f, 0.01f);
    CoverageFixApplyRepeat = applyRepeat;
    CoverageFixApplyClamp = applyClamp;
    Debug3dClearMagenta = debug3dClearMagenta;

    ShaderConfig.uCoverageFixDepthBias = CoverageFixDepthBias;
}


void GLRenderer::SetRenderSettings(bool betterpolygons, int scale) noexcept
{
    if (betterpolygons == BetterPolygons && scale == ScaleFactor)
        return;

    CurGLCompositor.SetScaleFactor(scale);
    ScaleFactor = scale;
    BetterPolygons = betterpolygons;

    ScreenW = 256 * scale;
    ScreenH = 192 * scale;

    glBindTexture(GL_TEXTURE_2D, ColorBufferTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ScreenW, ScreenH, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindTexture(GL_TEXTURE_2D, DepthBufferTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, ScreenW, ScreenH, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
    glBindTexture(GL_TEXTURE_2D, AttrBufferTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ScreenW, ScreenH, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glBindFramebuffer(GL_FRAMEBUFFER, DownscaleFramebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, DownScaleBufferTex, 0);

    GLenum fbassign[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};

    glBindFramebuffer(GL_FRAMEBUFFER, MainFramebuffer);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, ColorBufferTex, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, DepthBufferTex, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, AttrBufferTex, 0);
    glDrawBuffers(2, fbassign);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, PixelbufferID);
    glBufferData(GL_PIXEL_PACK_BUFFER, 256*192*4, NULL, GL_DYNAMIC_READ);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    //glLineWidth(scale);
    //glLineWidth(1.5);
}


void GLRenderer::SetupPolygon(GLRenderer::RendererPolygon* rp, Polygon* polygon) const
{
    rp->PolyData = polygon;
    rp->Meta = BuildAcceleratedPolygonMeta(*polygon);
}

u32* GLRenderer::SetupVertex(const Polygon* poly, int vid, const Vertex* vtx, u32 vtxattr, u32* vptr) const
{
    const bool useHiresCoordinates = ScaleFactor > 1;
    const u32 xFixed = static_cast<u32>(ResolveAcceleratedVertexFixedX(*vtx, ScaleFactor, useHiresCoordinates));
    const u32 yFixed = static_cast<u32>(ResolveAcceleratedVertexFixedY(*vtx, ScaleFactor, useHiresCoordinates));

    return SetupVertex(poly, vid, vtx, vtxattr, xFixed, yFixed, vptr);
}

u32* GLRenderer::SetupVertex(const Polygon* poly, int vid, const Vertex* vtx, u32 vtxattr, u32 x, u32 y, u32* vptr) const
{
    if (x > 0xFFFF) x = 0xFFFF;
    if (y > 0xFFFF) y = 0xFFFF;

    u32 z = poly->FinalZ[vid];
    u32 w = poly->FinalW[vid];

    u32 alpha = (poly->Attr >> 16) & 0x1F;

    // Z should always fit within 16 bits, so it's okay to do this
    u32 zshift = 0;
    while (z > 0xFFFF) { z >>= 1; zshift++; }

    // correct nearly-vertical edges that would look vertical on the DS
    /*{
        int vtopid = vid - 1;
        if (vtopid < 0) vtopid = poly->NumVertices-1;
        Vertex* vtop = poly->Vertices[vtopid];
        if (vtop->FinalPosition[1] >= vtx->FinalPosition[1])
        {
            vtopid = vid + 1;
            if (vtopid >= poly->NumVertices) vtopid = 0;
            vtop = poly->Vertices[vtopid];
        }
        if ((vtop->FinalPosition[1] < vtx->FinalPosition[1]) &&
            (vtx->FinalPosition[0] == vtop->FinalPosition[0]-1))
        {
            if (ScaleFactor > 1)
                x = (vtop->HiresPosition[0] * ScaleFactor) >> 4;
            else
                x = vtop->FinalPosition[0];
        }
    }*/

    *vptr++ = x | (y << 16);
    *vptr++ = z | (w << 16);

    *vptr++ =  (vtx->FinalColor[0] >> 1) |
              ((vtx->FinalColor[1] >> 1) << 8) |
              ((vtx->FinalColor[2] >> 1) << 16) |
              (alpha << 24);

    *vptr++ = (u16)vtx->TexCoords[0] | ((u16)vtx->TexCoords[1] << 16);

    // Split TexParam into 2 because some GPUs don't have 32 bit ints. TexPalette only uses 13 bits
    *vptr++ = vtxattr | (zshift << 16);
    *vptr++ = poly->TexParam & 0xFFFF;
    *vptr++ = (poly->TexParam >> 16 ) | (poly->TexPalette << 16);

    return vptr;
}

void GLRenderer::BuildPolygons(const AcceleratedScene& scene)
{
    NumVertices = static_cast<u32>(scene.Vertices.size());
    NumIndices = static_cast<u32>(scene.Indices.size());
    NumEdgeIndices = static_cast<u32>(scene.EdgeIndices.size());

    u32* vptr = &VertexBuffer[0];
    for (u32 drawIndex = 0; drawIndex < scene.Draws.size(); drawIndex++)
    {
        const AcceleratedSceneDraw& draw = scene.Draws[drawIndex];
        RendererPolygon* rp = &PolygonList[drawIndex];
        rp->PolyData = const_cast<Polygon*>(draw.SourcePolygon);
        rp->Meta = draw.Meta;
        rp->NumIndices = draw.IndexCount;
        rp->IndicesOffset = draw.FirstIndex;
        rp->PrimType = draw.PrimitiveType == AcceleratedPrimitiveType::Lines ? GL_LINES : GL_TRIANGLES;
        rp->NumEdgeIndices = draw.EdgeIndexCount;
        rp->EdgeIndicesOffset = EdgeIndicesOffset + draw.FirstEdgeIndex;

        const Polygon* poly = draw.SourcePolygon;
        for (u32 vertexIndex = draw.FirstVertex; vertexIndex < draw.FirstVertex + draw.VertexCount; vertexIndex++)
        {
            if (vertexIndex >= scene.Vertices.size() || poly == nullptr)
                continue;

            const AcceleratedSceneVertex& vertex = scene.Vertices[vertexIndex];
            *vptr++ = vertex.XFixed | (vertex.YFixed << 16u);
            *vptr++ = vertex.GlZWPacked;
            *vptr++ = vertex.GlColorPacked;
            *vptr++ = static_cast<u16>(vertex.TexCoordS) | (static_cast<u16>(vertex.TexCoordT) << 16u);
            *vptr++ = vertex.VertexAttr;
            *vptr++ = poly->TexParam & 0xFFFFu;
            *vptr++ = (poly->TexParam >> 16u) | (poly->TexPalette << 16u);
        }
    }

    std::copy(scene.Indices.begin(), scene.Indices.end(), IndexBuffer);
    std::copy(scene.EdgeIndices.begin(), scene.EdgeIndices.end(), IndexBuffer + EdgeIndicesOffset);
}

int GLRenderer::RenderSinglePolygon(int i) const
{
    const RendererPolygon* rp = &PolygonList[i];

    glDrawElements(rp->PrimType, rp->NumIndices, GL_UNSIGNED_SHORT, (void*)(uintptr_t)(rp->IndicesOffset * 2));

    return 1;
}

int GLRenderer::RenderPolygonBatch(int i) const
{
    const RendererPolygon* rp = &PolygonList[i];
    GLuint primtype = rp->PrimType;
    u32 key = rp->Meta.RenderKey;
    int numpolys = 0;
    u32 numindices = 0;

    for (int iend = i; iend < NumFinalPolys; iend++)
    {
        const RendererPolygon* cur_rp = &PolygonList[iend];
        if (cur_rp->PrimType != primtype) break;
        if (cur_rp->Meta.RenderKey != key) break;

        numpolys++;
        numindices += cur_rp->NumIndices;
    }

    glDrawElements(primtype, numindices, GL_UNSIGNED_SHORT, (void*)(uintptr_t)(rp->IndicesOffset * 2));
    return numpolys;
}

int GLRenderer::RenderPolygonEdgeBatch(int i) const
{
    const RendererPolygon* rp = &PolygonList[i];
    u32 key = rp->Meta.RenderKey;
    int numpolys = 0;
    u32 numindices = 0;

    for (int iend = i; iend < NumFinalPolys; iend++)
    {
        const RendererPolygon* cur_rp = &PolygonList[iend];
        if (cur_rp->Meta.RenderKey != key) break;

        numpolys++;
        numindices += cur_rp->NumEdgeIndices;
    }

    glDrawElements(GL_LINES, numindices, GL_UNSIGNED_SHORT, (void*)(uintptr_t)(rp->EdgeIndicesOffset * 2));
    return numpolys;
}

void GLRenderer::RenderSceneChunk(const GPU3D& gpu3d, int y, int h)
{
    u32 flags = 0;
    if (gpu3d.RenderPolygonRAM[0]->WBuffer) flags |= RenderFlag_WBuffer;

    if (h != 192) glScissor(0, y<<ScaleFactor, 256<<ScaleFactor, h<<ScaleFactor);

    GLboolean fogenable = (gpu3d.RenderDispCnt & (1<<7)) ? GL_TRUE : GL_FALSE;

    // TODO: proper 'equal' depth test!
    // (has margin of +-0x200 in Z-buffer mode, +-0xFF in W-buffer mode)
    // for now we're using GL_LEQUAL to make it work to some extent

    // pass 1: opaque pixels

    UseRenderShader(flags);
    glLineWidth(1.0);

    glColorMaski(1, GL_TRUE, GL_TRUE, fogenable, GL_FALSE);

    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    glBindVertexArray(VertexArrayID);

    for (int i = 0; i < NumFinalPolys; )
    {
        RendererPolygon* rp = &PolygonList[i];
        const AcceleratedPolygonMeta& polygonMeta = rp->Meta;

        if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagShadowMask)) { i++; continue; }
        if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagTranslucent)) { i++; continue; }

        if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagDepthEqual))
            glDepthFunc(GL_LEQUAL);
        else
            glDepthFunc(GL_LESS);

        glStencilFunc(GL_ALWAYS, polygonMeta.PolyId, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glStencilMask(0xFF);

        i += RenderPolygonBatch(i);
    }

    // if edge marking is enabled, mark all opaque edges
    // TODO BETTER EDGE MARKING!!! THIS SUCKS
    /*if (RenderDispCnt & (1<<5))
    {
        UseRenderShader(flags | RenderFlag_Edge);
        glLineWidth(1.5);

        glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glColorMaski(1, GL_FALSE, GL_TRUE, GL_FALSE, GL_FALSE);

        glDepthFunc(GL_ALWAYS);
        glDepthMask(GL_FALSE);

        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilMask(0);

        for (int i = 0; i < NumFinalPolys; )
        {
            RendererPolygon* rp = &PolygonList[i];

            if (rp->PolyData->IsShadowMask) { i++; continue; }

            i += RenderPolygonEdgeBatch(i);
        }

        glDepthMask(GL_TRUE);
    }*/

    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);

    if (gpu3d.RenderDispCnt & (1<<3))
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
    else
        glBlendFuncSeparate(GL_ONE, GL_ZERO, GL_ONE, GL_ONE);

    glLineWidth(1.0);

    if (NumOpaqueFinalPolys > -1)
    {
        // pass 2: if needed, render translucent pixels that are against background pixels
        // when background alpha is zero, those need to be rendered with blending disabled

        if ((gpu3d.RenderClearAttr1 & 0x001F0000) == 0)
        {
            glDisable(GL_BLEND);

            for (int i = 0; i < NumFinalPolys; )
            {
                RendererPolygon* rp = &PolygonList[i];
                const AcceleratedPolygonMeta& polygonMeta = rp->Meta;

                if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagShadowMask))
                {
                    // draw actual shadow mask

                    UseRenderShader(flags | RenderFlag_ShadowMask);

                    glDisable(GL_BLEND);
                    glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                    glColorMaski(1, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                    glDepthMask(GL_FALSE);

                    glDepthFunc(GL_LESS);
                    glStencilFunc(GL_EQUAL, 0xFF, 0xFF);
                    glStencilOp(GL_KEEP, GL_INVERT, GL_KEEP);
                    glStencilMask(0x01);

                    i += RenderPolygonBatch(i);
                }
                else if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagTranslucent))
                {
                    bool needopaque = HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagNeedOpaquePass);
                    u32 polyattr = polygonMeta.PolyAttr;
                    u32 polyid = polygonMeta.PolyId;

                    if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagDepthEqual))
                        glDepthFunc(GL_LEQUAL);
                    else
                        glDepthFunc(GL_LESS);

                    if (needopaque)
                    {
                        UseRenderShader(flags);

                        glDisable(GL_BLEND);
                        glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                        glColorMaski(1, GL_TRUE, GL_TRUE, fogenable, GL_FALSE);

                        glStencilFunc(GL_ALWAYS, polyid, 0xFF);
                        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
                        glStencilMask(0xFF);

                        glDepthMask(GL_TRUE);

                        RenderSinglePolygon(i);
                    }

                    UseRenderShader(flags | RenderFlag_Trans);

                    const GLboolean transfog = HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagFogWrite)
                        ? fogenable
                        : GL_FALSE;

                    if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagShadow))
                    {
                        // shadow against clear-plane will only pass if its polyID matches that of the clear plane
                        u32 clrpolyid = (gpu3d.RenderClearAttr1 >> 24) & 0x3F;
                        if (polyid != clrpolyid) { i++; continue; }

                        glEnable(GL_BLEND);
                        glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                        glColorMaski(1, GL_FALSE, GL_FALSE, transfog, GL_FALSE);

                        glStencilFunc(GL_EQUAL, 0xFE, 0xFF);
                        glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
                        glStencilMask(~(0x40|polyid)); // heheh

                        if (polyattr & (1<<11)) glDepthMask(GL_TRUE);
                        else                    glDepthMask(GL_FALSE);

                        i += needopaque ? RenderSinglePolygon(i) : RenderPolygonBatch(i);
                    }
                    else
                    {
                        glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                        glColorMaski(1, GL_FALSE, GL_FALSE, transfog, GL_FALSE);

                        glStencilFunc(GL_EQUAL, 0xFF, 0xFE);
                        glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
                        glStencilMask(~(0x40|polyid)); // heheh

                        if (polyattr & (1<<11)) glDepthMask(GL_TRUE);
                        else                    glDepthMask(GL_FALSE);

                        i += needopaque ? RenderSinglePolygon(i) : RenderPolygonBatch(i);
                    }
                }
                else
                    i++;
            }

            glEnable(GL_BLEND);
            glStencilMask(0xFF);
        }

        // pass 3: translucent pixels

        for (int i = 0; i < NumFinalPolys; )
        {
            RendererPolygon* rp = &PolygonList[i];
            const AcceleratedPolygonMeta& polygonMeta = rp->Meta;

            if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagShadowMask))
            {
                // clear shadow bits in stencil buffer

                glStencilMask(0x80);
                glClear(GL_STENCIL_BUFFER_BIT);

                // draw actual shadow mask

                UseRenderShader(flags | RenderFlag_ShadowMask);

                glDisable(GL_BLEND);
                glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                glColorMaski(1, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                glDepthMask(GL_FALSE);

                glDepthFunc(GL_LESS);
                glStencilFunc(GL_ALWAYS, 0x80, 0x80);
                glStencilOp(GL_KEEP, GL_REPLACE, GL_KEEP);

                i += RenderPolygonBatch(i);
            }
            else if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagTranslucent))
            {
                bool needopaque = HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagNeedOpaquePass);
                u32 polyattr = polygonMeta.PolyAttr;
                u32 polyid = polygonMeta.PolyId;

                if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagDepthEqual))
                    glDepthFunc(GL_LEQUAL);
                else
                    glDepthFunc(GL_LESS);

                if (needopaque)
                {
                    UseRenderShader(flags);

                    glDisable(GL_BLEND);
                    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                    glColorMaski(1, GL_TRUE, GL_TRUE, fogenable, GL_FALSE);

                    glStencilFunc(GL_ALWAYS, polyid, 0xFF);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
                    glStencilMask(0xFF);

                    glDepthMask(GL_TRUE);

                    RenderSinglePolygon(i);
                }

                UseRenderShader(flags | RenderFlag_Trans);

                const GLboolean transfog = HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagFogWrite)
                    ? fogenable
                    : GL_FALSE;

                if (HasAcceleratedPolygonFlag(polygonMeta, AcceleratedPolygonFlagShadow))
                {
                    glDisable(GL_BLEND);
                    glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                    glColorMaski(1, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
                    glDepthMask(GL_FALSE);
                    glStencilFunc(GL_EQUAL, polyid, 0x3F);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO);
                    glStencilMask(0x80);

                    RenderSinglePolygon(i);

                    glEnable(GL_BLEND);
                    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                    glColorMaski(1, GL_FALSE, GL_FALSE, transfog, GL_FALSE);

                    glStencilFunc(GL_EQUAL, 0xC0|polyid, 0x80);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
                    glStencilMask(0x7F);

                    if (polyattr & (1<<11)) glDepthMask(GL_TRUE);
                    else                    glDepthMask(GL_FALSE);

                    i += RenderSinglePolygon(i);
                }
                else
                {
                    glEnable(GL_BLEND);
                    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                    glColorMaski(1, GL_FALSE, GL_FALSE, transfog, GL_FALSE);

                    glStencilFunc(GL_NOTEQUAL, 0x40|polyid, 0x7F);
                    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
                    glStencilMask(0x7F);

                    if (polyattr & (1<<11)) glDepthMask(GL_TRUE);
                    else                    glDepthMask(GL_FALSE);

                    i += needopaque ? RenderSinglePolygon(i) : RenderPolygonBatch(i);
                }
            }
            else
                i++;
        }
    }

    if (gpu3d.RenderDispCnt & 0x00A0) // fog/edge enabled
    {
        glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glColorMaski(1, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);

        glDepthFunc(GL_ALWAYS);
        glDepthMask(GL_FALSE);
        glStencilFunc(GL_ALWAYS, 0, 0);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilMask(0);
        // While depth and stencil writing operations are disabled by the commands above, the fact that the same texture is used as both input and output results in undefined
        // behaviour, which manifests as visual artifacts on some devices. Depth/stencil texture is attached again after fog/edge rendering
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 0, 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, DepthBufferTex);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, AttrBufferTex);

        glBindBuffer(GL_ARRAY_BUFFER, ClearVertexBufferID);
        glBindVertexArray(ClearVertexArrayID);

        if (gpu3d.RenderDispCnt & (1<<5))
        {
            // edge marking
            // TODO: depth/polyid values at screen edges

            glUseProgram(FinalPassEdgeShader);

            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);

            glDrawArrays(GL_TRIANGLES, 0, 2*3);
        }

        if (gpu3d.RenderDispCnt & (1<<7))
        {
            // fog

            glUseProgram(FinalPassFogShader);

            if (gpu3d.RenderDispCnt & (1<<6))
                glBlendFuncSeparate(GL_ZERO, GL_ONE, GL_CONSTANT_COLOR, GL_ONE_MINUS_SRC_ALPHA);
            else
                glBlendFuncSeparate(GL_CONSTANT_COLOR, GL_ONE_MINUS_SRC_ALPHA, GL_CONSTANT_COLOR, GL_ONE_MINUS_SRC_ALPHA);

            {
                u32 c = gpu3d.RenderFogColor;
                u32 r = c & 0x1F;
                u32 g = (c >> 5) & 0x1F;
                u32 b = (c >> 10) & 0x1F;
                u32 a = (c >> 16) & 0x1F;

                glBlendColor((float)b/31.0, (float)g/31.0, (float)r/31.0, (float)a/31.0);
            }

            glDrawArrays(GL_TRIANGLES, 0, 2*3);
        }

        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, DepthBufferTex, 0);
    }
}


void GLRenderer::RenderFrame(GPU& gpu)
{
    CurShaderID = -1;

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, MainFramebuffer);

    ShaderConfig.uScreenSize[0] = ScreenW;
    ShaderConfig.uScreenSize[1] = ScreenH;
    ShaderConfig.uDispCnt = gpu.GPU3D.RenderDispCnt;
    ShaderConfig.uCoverageFixDepthBias = CoverageFixDepthBias;

    for (int i = 0; i < 32; i++)
    {
        u16 c = gpu.GPU3D.RenderToonTable[i];
        u32 r = c & 0x1F;
        u32 g = (c >> 5) & 0x1F;
        u32 b = (c >> 10) & 0x1F;

        ShaderConfig.uToonColors[i][0] = (float)r / 31.0;
        ShaderConfig.uToonColors[i][1] = (float)g / 31.0;
        ShaderConfig.uToonColors[i][2] = (float)b / 31.0;
    }

    for (int i = 0; i < 8; i++)
    {
        u16 c = gpu.GPU3D.RenderEdgeTable[i];
        u32 r = c & 0x1F;
        u32 g = (c >> 5) & 0x1F;
        u32 b = (c >> 10) & 0x1F;

        ShaderConfig.uEdgeColors[i][0] = (float)r / 31.0;
        ShaderConfig.uEdgeColors[i][1] = (float)g / 31.0;
        ShaderConfig.uEdgeColors[i][2] = (float)b / 31.0;
    }

    {
        u32 c = gpu.GPU3D.RenderFogColor;
        u32 r = c & 0x1F;
        u32 g = (c >> 5) & 0x1F;
        u32 b = (c >> 10) & 0x1F;
        u32 a = (c >> 16) & 0x1F;

        ShaderConfig.uFogColor[0] = (float)r / 31.0;
        ShaderConfig.uFogColor[1] = (float)g / 31.0;
        ShaderConfig.uFogColor[2] = (float)b / 31.0;
        ShaderConfig.uFogColor[3] = (float)a / 31.0;
    }

    for (int i = 0; i < 34; i++)
    {
        u8 d = gpu.GPU3D.RenderFogDensityTable[i];
        ShaderConfig.uFogDensity[i][0] = (float)d / 127.0;
    }

    ShaderConfig.uFogOffset = gpu.GPU3D.RenderFogOffset;
    ShaderConfig.uFogShift = gpu.GPU3D.RenderFogShift;

    glBindBuffer(GL_UNIFORM_BUFFER, ShaderConfigUBO);
    void* unibuf = glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(ShaderConfig), GL_MAP_WRITE_BIT);
    if (unibuf) memcpy(unibuf, &ShaderConfig, sizeof(ShaderConfig));
    glUnmapBuffer(GL_UNIFORM_BUFFER);

    // SUCKY!!!!!!!!!!!!!!!!!!
    // TODO: detect when VRAM blocks are modified!
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, TexMemID);
    for (int i = 0; i < 4; i++)
    {
        u32 mask = gpu.VRAMMap_Texture[i];
        u8* vram;
        if (!mask) continue;
        else if (mask & (1<<0)) vram = gpu.VRAM_A;
        else if (mask & (1<<1)) vram = gpu.VRAM_B;
        else if (mask & (1<<2)) vram = gpu.VRAM_C;
        else if (mask & (1<<3)) vram = gpu.VRAM_D;

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, i*128, 1024, 128, GL_RED_INTEGER, GL_UNSIGNED_BYTE, vram);
    }

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, TexPalMemID);

    u16* tempBuffer = (u16*) malloc(1024 * 8 * 2);
    for (int i = 0; i < 6; i++)
    {
        // 6 x 16K chunks
        u32 mask = gpu.VRAMMap_TexPal[i];
        u8* vram;
        if (!mask) continue;
        else if (mask & (1<<4)) vram = &gpu.VRAM_E[(i&3)*0x4000];
        else if (mask & (1<<5)) vram = gpu.VRAM_F;
        else if (mask & (1<<6)) vram = gpu.VRAM_G;

        memcpy(tempBuffer, vram, 1024 * 8 * 2);
        for (int j = 0; j < 1024 * 8; j++)
        {
            u16 value = tempBuffer[j];

            u8 a = (value >> 15) & 0x1;
            u8 b = (value >> 10) & 0x1F;
            u8 g = (value >> 5) & 0x1F;
            u8 r = (value >> 0) & 0x1F;

            tempBuffer[j] = (r << 11) | (g << 6) | (b << 1) | a;
        }

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, i*8, 1024, 8, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, tempBuffer);
    }
    free(tempBuffer);

    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);

    glViewport(0, 0, ScreenW, ScreenH);

    glDisable(GL_BLEND);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glStencilMask(0xFF);

    // clear buffers
    // TODO: clear bitmap
    // TODO: check whether 'clear polygon ID' affects translucent polyID
    // (for example when alpha is 1..30)
    {
        glUseProgram(ClearShaderPlain);
        glDepthFunc(GL_ALWAYS);

        u32 r = gpu.GPU3D.RenderClearAttr1 & 0x1F;
        u32 g = (gpu.GPU3D.RenderClearAttr1 >> 5) & 0x1F;
        u32 b = (gpu.GPU3D.RenderClearAttr1 >> 10) & 0x1F;
        u32 fog = (gpu.GPU3D.RenderClearAttr1 >> 15) & 0x1;
        u32 a = (gpu.GPU3D.RenderClearAttr1 >> 16) & 0x1F;
        u32 polyid = (gpu.GPU3D.RenderClearAttr1 >> 24) & 0x3F;
        u32 z = ((gpu.GPU3D.RenderClearAttr2 & 0x7FFF) * 0x200) + 0x1FF;

        if (Debug3dClearMagenta)
        {
            r = 31;
            g = 0;
            b = 31;
            a = 31;
        }

        glStencilFunc(GL_ALWAYS, 0xFF, 0xFF);
        glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);

        /*if (r) r = r*2 + 1;
        if (g) g = g*2 + 1;
        if (b) b = b*2 + 1;*/

        glUniform4ui(ClearUniformLoc[0], r, g, b, a);
        glUniform1ui(ClearUniformLoc[1], z);
        glUniform1ui(ClearUniformLoc[2], polyid);
        glUniform1ui(ClearUniformLoc[3], fog);

        glBindBuffer(GL_ARRAY_BUFFER, ClearVertexBufferID);
        glBindVertexArray(ClearVertexArrayID);
        glDrawArrays(GL_TRIANGLES, 0, 2*3);
    }

    if (gpu.GPU3D.RenderNumPolygons)
    {
        u32 flags = 0;
        if (gpu.GPU3D.RenderPolygonRAM[0]->WBuffer) flags |= RenderFlag_WBuffer;

        AcceleratedSceneBuildConfig sceneBuildConfig{};
        sceneBuildConfig.Scale = ScaleFactor;
        sceneBuildConfig.BetterPolygons = BetterPolygons;
        sceneBuildConfig.UseHiresCoordinates = ScaleFactor > 1;
        sceneBuildConfig.MaxFixedX = (ScreenW * 16) - 1;
        sceneBuildConfig.MaxFixedY = (ScreenH * 16) - 1;
        sceneBuildConfig.CoverageFix.Enabled = CoverageFixEnabled;
        sceneBuildConfig.CoverageFix.UserPx = CoverageFixPx;
        sceneBuildConfig.CoverageFix.ApplyRepeat = CoverageFixApplyRepeat;
        sceneBuildConfig.CoverageFix.ApplyClamp = CoverageFixApplyClamp;

        BuildAcceleratedScene(gpu.GPU3D, sceneBuildConfig, SharedScene);
        NumFinalPolys = static_cast<int>(SharedScene.Draws.size());
        NumOpaqueFinalPolys = SharedScene.FirstTranslucentDraw == std::numeric_limits<u32>::max()
            ? -1
            : static_cast<int>(SharedScene.FirstTranslucentDraw);

        BuildPolygons(SharedScene);
        glBindBuffer(GL_ARRAY_BUFFER, VertexBufferID);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NumVertices*7*4, VertexBuffer);

        // bind to access the index buffer
        glBindVertexArray(VertexArrayID);
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, NumIndices * 2, IndexBuffer);
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, EdgeIndicesOffset * 2, NumEdgeIndices * 2, IndexBuffer + EdgeIndicesOffset);

        RenderSceneChunk(gpu.GPU3D, 0, 192);
    }
}

void GLRenderer::Stop(const GPU& gpu)
{
    CurGLCompositor.Stop(gpu);
}

void GLRenderer::PrepareCaptureFrame()
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, MainFramebuffer);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, DownscaleFramebuffer);
    GLenum bufferAttachment = GL_COLOR_ATTACHMENT0;
    glDrawBuffers(1, &bufferAttachment);
    glBlitFramebuffer(0, 0, ScreenW, ScreenH, 0, 0, 256, 192, GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, PixelbufferID);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, DownscaleFramebuffer);
    glReadPixels(0, 0, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

void GLRenderer::Blit(const GPU& gpu)
{
    CurGLCompositor.RenderFrame(gpu, *this);
}

void GLRenderer::SetOutputTexture(int buffer, u32 texture)
{
    CurGLCompositor.SetOutputTexture(buffer, (GLuint) texture);
}

void GLRenderer::BindOutputTexture(int buffer)
{
    CurGLCompositor.BindOutputTexture(buffer);
}

u32* GLRenderer::GetLine(int line)
{
    int stride = 256;

    if (line == 0)
    {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, PixelbufferID);
        u8* data = (u8*)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, 4 * stride * 192, GL_MAP_READ_BIT);
        if (data) memcpy(&Framebuffer[stride*0], data, 4*stride*192);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }

    u64* ptr = (u64*)&Framebuffer[stride * line];
    for (int i = 0; i < stride; i+=2)
    {
        // Data is in BGRA format but we need to convert it to RGBA
        u64 redBits = (*ptr & 0x000000FC000000FC) << 16;
        u64 blueBits = (*ptr & 0x00FC000000FC0000) >> 16;

        u64 rgb = (*ptr & 0x0000FC000000FC00) | redBits | blueBits;
        u64 a = *ptr & 0xF8000000F8000000;

        *ptr++ = (rgb >> 2) | (a >> 3);
    }

    return &Framebuffer[stride * line];
}

void GLRenderer::SetupAccelFrame()
{
    glBindTexture(GL_TEXTURE_2D, ColorBufferTex);
}

std::vector<u32> GLRenderer::CaptureColorTargetForDebug()
{
    if (ScreenW <= 0 || ScreenH <= 0)
        return {};

    std::vector<u32> pixels(static_cast<size_t>(ScreenW) * static_cast<size_t>(ScreenH));
    glBindFramebuffer(GL_READ_FRAMEBUFFER, MainFramebuffer);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, ScreenW, ScreenH, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    for (u32& pixel : pixels)
        pixel = ConvertBgraToRgba8(pixel);
    return pixels;
}

std::vector<u32> GLRenderer::CaptureTopDepthForDebug()
{
    if (ScreenW <= 0 || ScreenH <= 0)
        return {};

    std::vector<u32> depth(static_cast<size_t>(ScreenW) * static_cast<size_t>(ScreenH));
    glBindFramebuffer(GL_READ_FRAMEBUFFER, MainFramebuffer);
    glReadBuffer(GL_NONE);
    glReadPixels(0, 0, ScreenW, ScreenH, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, depth.data());
    for (u32& value : depth)
        value >>= 8;
    return depth;
}

std::vector<u32> GLRenderer::CaptureTopAttrForDebug()
{
    if (ScreenW <= 0 || ScreenH <= 0)
        return {};

    std::vector<u32> rawPixels(static_cast<size_t>(ScreenW) * static_cast<size_t>(ScreenH));
    glBindFramebuffer(GL_READ_FRAMEBUFFER, MainFramebuffer);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, ScreenW, ScreenH, GL_RGBA, GL_UNSIGNED_BYTE, rawPixels.data());

    std::vector<u32> attr(rawPixels.size(), 0u);
    for (size_t i = 0; i < rawPixels.size(); i++)
        attr[i] = PackOpenGlAttrToLogical(ConvertBgraToRgba8(rawPixels[i]));
    return attr;
}

std::vector<u32> GLRenderer::CaptureTopCoverageForDebug()
{
    const std::vector<u32> topAttr = CaptureTopAttrForDebug();
    if (topAttr.empty())
        return {};

    std::vector<u32> coverage(topAttr.size(), 0u);
    for (size_t i = 0; i < topAttr.size(); i++)
        coverage[i] = (topAttr[i] >> 8u) & 0x1Fu;
    return coverage;
}

}
