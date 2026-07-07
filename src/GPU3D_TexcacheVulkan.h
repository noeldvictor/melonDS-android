#ifndef GPU3D_TEXCACHEVULKAN
#define GPU3D_TEXCACHEVULKAN

#include <memory>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#include "GPU3D_Texcache.h"

namespace melonDS
{

class TexcacheVulkanLoader
{
public:
    using TextureHandle = u64;

    TexcacheVulkanLoader();
    ~TexcacheVulkanLoader();

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

    TextureHandle GenerateTexture(u32 width, u32 height, u32 layers);
    void UploadTexture(TextureHandle handle, u32 width, u32 height, u32 layer, void* data);
    void UploadReplacement(TextureHandle handle, u32 width, u32 height, u32 layer, const HDTexPackImage& img);
    void DeleteTexture(TextureHandle handle);
    bool GetTextureDescriptor(TextureHandle handle, VkDescriptorImageInfo* outImageInfo) const;
    bool IsTextureLayerOpaque(TextureHandle handle, u32 layer) const;

private:
    struct TextureArray
    {
        u32 Width = 0;
        u32 Height = 0;
        u32 Layers = 0;
        u32 Scale = 1;

        VkImage Image = VK_NULL_HANDLE;
        VkDeviceMemory Memory = VK_NULL_HANDLE;
        VkImageView ArrayView = VK_NULL_HANDLE;
        VkSampler Sampler = VK_NULL_HANDLE;

        VkBuffer StagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory StagingMemory = VK_NULL_HANDLE;
        VkDeviceSize StagingSize = 0;
        std::vector<u8> LayerOpaque;
    };

    struct SharedState
    {
        TextureHandle NextHandle = 1;
        std::unordered_map<TextureHandle, TextureArray> TextureArrays;

        bool ContextAcquired = false;
        VkDevice Device = VK_NULL_HANDLE;
        VkQueue Queue = VK_NULL_HANDLE;
        u32 QueueFamilyIndex = 0;
        VkCommandPool CommandPool = VK_NULL_HANDLE;
        VkCommandBuffer CommandBuffer = VK_NULL_HANDLE;
        VkFence UploadFence = VK_NULL_HANDLE;
    };

    bool EnsureVulkanState();
    void CleanupVulkanState();
    void DestroyTextureArray(TextureArray& textureArray);
    void UploadLayer(TextureArray& textureArray, u32 layer, const u32* texels);

private:
    std::shared_ptr<SharedState> State;
    u32 HDTextureScale = 1;
    int HDTextureFilterMode = 0;
    u32 TexPackScale = 1;
    std::vector<u32> UploadBuffer;
};

using TexcacheVulkan = Texcache<TexcacheVulkanLoader, TexcacheVulkanLoader::TextureHandle>;

}

#endif
