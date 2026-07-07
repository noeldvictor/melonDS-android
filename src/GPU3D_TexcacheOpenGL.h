#ifndef GPU3D_TEXCACHEOPENGL
#define GPU3D_TEXCACHEOPENGL

#include "GPU3D_Texcache.h"
#include "OpenGLSupport.h"

#include <vector>

namespace melonDS
{

template <typename, typename>
class Texcache;

class TexcacheOpenGLLoader
{
public:
    bool SetHDTextureFilter(int scale, int mode);
    [[nodiscard]] u32 GetHDTextureScale() const { return HDTextureScale; }
    [[nodiscard]] int GetHDTextureFilterMode() const { return HDTextureFilterMode; }
    [[nodiscard]] u32 GetStorageScale() const
    {
        u32 filterScale = HDTextureFilterMode == 0 ? 1 : HDTextureScale;
        return filterScale > TexPackScale ? filterScale : TexPackScale;
    }

    void SetTexPackScale(u32 scale);
    [[nodiscard]] u32 GetTexPackScale() const { return TexPackScale; }

    GLuint GenerateTexture(u32 width, u32 height, u32 layers);
    void UploadTexture(GLuint handle, u32 width, u32 height, u32 layer, void* data);
    void UploadReplacement(GLuint handle, u32 width, u32 height, u32 layer, const HDTexPackImage& img);
    void DeleteTexture(GLuint handle);

private:
    u32 HDTextureScale = 1;
    int HDTextureFilterMode = 0;
    u32 TexPackScale = 1;
    std::vector<u32> UploadBuffer;
};

using TexcacheOpenGL = Texcache<TexcacheOpenGLLoader, GLuint>;

}

#endif
