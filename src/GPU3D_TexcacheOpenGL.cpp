#include "GPU3D_TexcacheOpenGL.h"

#include "HDTextureFilter.h"

namespace melonDS
{

bool TexcacheOpenGLLoader::SetHDTextureFilter(int scale, int mode)
{
    const u32 clampedScale = HDTextureFilter::ClampScale(scale);
    const int clampedMode = HDTextureFilter::ClampMode(mode);
    if (HDTextureScale == clampedScale && HDTextureFilterMode == clampedMode)
        return false;

    HDTextureScale = clampedScale;
    HDTextureFilterMode = clampedMode;
    UploadBuffer.clear();
    return true;
}

void TexcacheOpenGLLoader::SetTexPackScale(u32 scale)
{
    if (scale < 1) scale = 1;
    if (scale > 8) scale = 8;
    if (TexPackScale == scale)
        return;
    TexPackScale = scale;
    UploadBuffer.clear();
}

GLuint TexcacheOpenGLLoader::GenerateTexture(u32 width, u32 height, u32 layers)
{
    const u32 storageScale = GetStorageScale();
    GLuint texarray;
    glGenTextures(1, &texarray);
    glBindTexture(GL_TEXTURE_2D_ARRAY, texarray);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8UI, width * storageScale, height * storageScale, layers);
    return texarray;
}

void TexcacheOpenGLLoader::UploadTexture(GLuint handle, u32 width, u32 height, u32 layer, void* data)
{
    const u32 storageScale = GetStorageScale();
    const void* uploadData = data;
    if (storageScale > 1)
    {
        // mode 0 upscales nearest so pack-forced storage scaling stays accurate
        HDTextureFilter::UpscaleTexture(static_cast<const u32*>(data), width, height, storageScale,
                                        HDTextureFilterMode, UploadBuffer);
        uploadData = UploadBuffer.data();
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, handle);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
        0, 0, 0, layer,
        width * storageScale, height * storageScale, 1,
        GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, uploadData);
}

void TexcacheOpenGLLoader::UploadReplacement(GLuint handle, u32 width, u32 height, u32 layer, const HDTexPackImage& img)
{
    const u32 storageScale = GetStorageScale();
    const u32 dstW = width * storageScale, dstH = height * storageScale;

    UploadBuffer.resize((size_t)dstW * dstH);
    if (img.Width == dstW && img.Height == dstH)
    {
        for (size_t i = 0; i < UploadBuffer.size(); i++)
            UploadBuffer[i] = HDTexPack::RGBA8ToRGB6A5(img.RGBA[i]);
    }
    else
    {
        // pack scale differs from storage scale: nearest-resample to fit
        for (u32 y = 0; y < dstH; y++)
        {
            const u32 sy = (u32)((u64)y * img.Height / dstH);
            for (u32 x = 0; x < dstW; x++)
            {
                const u32 sx = (u32)((u64)x * img.Width / dstW);
                UploadBuffer[x + y * (size_t)dstW] =
                    HDTexPack::RGBA8ToRGB6A5(img.RGBA[sx + sy * (size_t)img.Width]);
            }
        }
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, handle);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
        0, 0, 0, layer,
        dstW, dstH, 1,
        GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, UploadBuffer.data());
}

void TexcacheOpenGLLoader::DeleteTexture(GLuint handle)
{
    glDeleteTextures(1, &handle);
}

}
