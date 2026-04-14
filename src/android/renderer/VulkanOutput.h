#ifndef VULKANOUTPUT_H
#define VULKANOUTPUT_H

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vulkan/vulkan.h>

#include "renderer/FrameQueue.h"
#include "types.h"
#include "VulkanPerfStats.h"

namespace melonDS
{
class GPU;
class VulkanRenderer3D;
}

namespace MelonDSAndroid
{

struct VulkanCompositionInputs
{
    VkImage sourceImage{VK_NULL_HANDLE};
    VkImageView sourceImageView{VK_NULL_HANDLE};
    VkBuffer topPackedBuffer{VK_NULL_HANDLE};
    VkBuffer bottomPackedBuffer{VK_NULL_HANDLE};
    VkDeviceSize packedBufferSize{};
    u32 packedStride{};
    u32 scale{};
    u32 rendererWidth{};
    u32 rendererHeight{};
    bool needsReadback{};
    bool multiSurface{};
    bool validationMode{};
};

class VulkanOutput
{
public:
    VulkanOutput() = default;
    ~VulkanOutput();

    VulkanOutput(const VulkanOutput&) = delete;
    VulkanOutput& operator=(const VulkanOutput&) = delete;

    bool init();
    void shutdown();
    [[nodiscard]] bool isInitialized() const { return initialized; }

    bool ensureFrameResources(Frame* frame, u32 width, u32 height);
    bool captureRenderer3dSnapshot(Frame* frame, const melonDS::VulkanRenderer3D& renderer3D);
    bool prepareFrameForPresentation(Frame* frame, const melonDS::GPU& gpu, int frontBuffer, const melonDS::VulkanRenderer3D& renderer3D);
    bool composeAndSubmitFrame(Frame* frame, const VulkanCompositionInputs& inputs);
    bool buildCompositionInputs(
        const Frame* frame,
        const melonDS::VulkanRenderer3D& renderer3D,
        int scale,
        bool needsReadback,
        bool multiSurface,
        bool validationMode,
        VulkanCompositionInputs& outInputs) const;
    bool validateFrameSubmission(Frame* frame, u64 waitTimeoutNs = UINT64_MAX);
    bool validateCompositorSubmission(Frame* frame, const melonDS::VulkanRenderer3D& renderer3D, int scale, u64 waitTimeoutNs = UINT64_MAX);
    bool validateRuntimePath(u32 width, u32 height, const melonDS::VulkanRenderer3D& renderer3D, int scale);
    bool isFrameReady(const Frame* frame) const;
    bool waitForFrame(const Frame* frame, u64 timeoutNs);
    bool readFramePixels(const Frame* frame, u32* destinationPixels, size_t destinationPixelCount, u64 waitTimeoutNs = UINT64_MAX);
    bool readPreparedRenderer3dPixels(
        const Frame* frame,
        u32* destinationPixels,
        size_t destinationPixelCount,
        u32& outWidth,
        u32& outHeight,
        u64 waitTimeoutNs = UINT64_MAX);
    bool getPreparedRenderer3dDimensions(const Frame* frame, u32& outWidth, u32& outHeight) const;
    bool getPreparedPackedBuffers(
        const Frame* frame,
        const u32*& outTopPacked,
        const u32*& outBottomPacked,
        u32& outPackedStride,
        u32& outPackedHeight,
        bool& outScreenSwap) const;
    [[nodiscard]] VkImage getFrameImage(const Frame* frame) const;
    [[nodiscard]] VkImageView getFrameImageView(const Frame* frame) const;

private:
    struct CompositorPushConstants
    {
        u32 outputWidth;
        u32 outputHeight;
        u32 scale;
        u32 rendererWidth;
        u32 rendererHeight;
        u32 packedStride;
    };

    struct FrameResource
    {
        VkImage image{VK_NULL_HANDLE};
        VkImageView imageView{VK_NULL_HANDLE};
        VkDeviceMemory imageMemory{VK_NULL_HANDLE};

        VkBuffer stagingBuffer{VK_NULL_HANDLE};
        VkDeviceMemory stagingMemory{VK_NULL_HANDLE};
        VkDeviceSize stagingSize{};

        VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
        VkFence submitFence{VK_NULL_HANDLE};
        VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
        VkQueryPool timestampQueryPool{VK_NULL_HANDLE};
        VkBuffer topPackedBuffer{VK_NULL_HANDLE};
        VkDeviceMemory topPackedMemory{VK_NULL_HANDLE};
        void* topPackedMapped{};
        VkBuffer bottomPackedBuffer{VK_NULL_HANDLE};
        VkDeviceMemory bottomPackedMemory{VK_NULL_HANDLE};
        void* bottomPackedMapped{};
        VkDeviceSize packedBufferSize{};
        VkImage renderer3dSnapshot{VK_NULL_HANDLE};
        VkImageView renderer3dSnapshotView{VK_NULL_HANDLE};
        VkDeviceMemory renderer3dSnapshotMemory{VK_NULL_HANDLE};
        u32 snapshotWidth{};
        u32 snapshotHeight{};

        u64 submissionValue{};
        u32 width{};
        u32 height{};
        bool screenSwap{};
        bool hasContent{};
        bool hasPreparedInputs{};
        bool hasRenderer3dSnapshot{};
        bool descriptorSetReady{};
        bool timestampPending{};
        VkImageView cachedRendererImageView{VK_NULL_HANDLE};
    };

private:
    bool createSyncObjects();
    bool createCommandObjects();
    bool createCompositorResources();
    bool createTimestampQueryPool(VkQueryPool& queryPool);
    void destroyTimestampQueryPool(VkQueryPool& queryPool);
    void destroyCompositorResources();
    bool createFrameResource(Frame* frame, u32 width, u32 height);
    void destroyFrameResource(Frame* frame);
    void destroyFrameResources();
    u32 findMemoryType(u32 typeBits, VkMemoryPropertyFlags properties) const;

    bool beginFrameCommand(FrameResource& resource, u64 waitTimeoutNs = UINT64_MAX);
    bool submitFrameCommand(Frame* frame, FrameResource& resource, bool signalTimeline);
    bool updateCompositorPackedBuffers(FrameResource& resource, const melonDS::GPU& gpu, int frontBuffer);
    bool ensureRenderer3dSnapshot(FrameResource& resource, u32 width, u32 height);
    void destroyRenderer3dSnapshot(FrameResource& resource);
    bool recordRenderer3dSnapshotCopy(FrameResource& resource, const melonDS::VulkanRenderer3D& renderer3D);
    bool recordDirectPresentationPrep(Frame* frame, FrameResource& resource, const melonDS::VulkanRenderer3D& renderer3D);
    bool dispatchCompositor(Frame* frame, FrameResource& resource, const VulkanCompositionInputs& inputs);
    void consumeFrameGpuTiming(FrameResource& resource);
    void logPerformanceIfNeeded();
    bool readResourceImagePixels(
        FrameResource& resource,
        const Frame* frame,
        VkImage image,
        u32 width,
        u32 height,
        u32* destinationPixels,
        size_t destinationPixelCount,
        u64 waitTimeoutNs);

private:
    bool initialized{};
    bool contextAcquired{};

    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    u32 queueFamilyIndex{};

    VkCommandPool commandPool{VK_NULL_HANDLE};

    VkSemaphore timelineSemaphore{VK_NULL_HANDLE};
    u64 timelineValue{};
    bool useTimelineSemaphores{};

    PFN_vkWaitSemaphoresKHR waitSemaphores{};
    PFN_vkGetSemaphoreCounterValueKHR getSemaphoreCounterValue{};
    PFN_vkResetQueryPoolEXT resetQueryPool{};
    float timestampPeriodNs{};
    bool timestampQueriesSupported{};

    VkDescriptorSetLayout compositorDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool compositorDescriptorPool{VK_NULL_HANDLE};
    VkPipelineLayout compositorPipelineLayout{VK_NULL_HANDLE};
    VkPipeline compositorPipeline{VK_NULL_HANDLE};

    std::unordered_map<Frame*, FrameResource> resources;
    PerfSampleWindow<120> packedUploadCpuWindow;
    PerfSampleWindow<120> composeCpuWindow;
    PerfSampleWindow<120> waitCpuWindow;
    PerfSampleWindow<120> compositorGpuWindow;
};

}

#endif // VULKANOUTPUT_H
