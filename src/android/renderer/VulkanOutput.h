#ifndef VULKANOUTPUT_H
#define VULKANOUTPUT_H

#include <cstddef>
#include <cstdint>
#include <array>
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

struct SoftPackedScreenStats
{
    std::array<u32, 4> DisplayModeCounts{};
    std::array<u32, 8> CompModeCounts{};
    int MinXOffset = 0;
    int MaxXOffset = 0;
    bool HasOffsets = false;
    u32 CaptureBackedComp4Pixels = 0;
    u32 CaptureBackedComp4Lines = 0;
    u32 RegularCaptureUses3dLines = 0;
    u32 VramCaptureUses3dLines = 0;
};

struct SoftPackedFrameSnapshot
{
    static constexpr size_t kScreenWidth = 256u;
    static constexpr size_t kScreenHeight = 192u;
    static constexpr size_t kPixelCount = kScreenWidth * kScreenHeight;
    static constexpr size_t kLineCount = kScreenHeight;

    u64 frameId = 0;
    int frontBufferLatched = -1;
    bool screenSwapLatched = false;
    bool valid = false;
    bool hasCapture3dSource = false;
    std::array<u32, kPixelCount> packedTopPlane0{};
    std::array<u32, kPixelCount> packedTopPlane1{};
    std::array<u32, kPixelCount> packedTopControl{};
    std::array<u32, kLineCount> packedTopLineMeta{};
    std::array<u32, kPixelCount> packedBottomPlane0{};
    std::array<u32, kPixelCount> packedBottomPlane1{};
    std::array<u32, kPixelCount> packedBottomControl{};
    std::array<u32, kLineCount> packedBottomLineMeta{};
    std::array<u32, kPixelCount> capture3dSourceDsFrame{};
    std::array<u8, kLineCount> captureLineUses3dMask{};
    std::array<u8, kLineCount> captureFallbackLines{};
    std::array<u32, kPixelCount> comp4TopPlaceholder{};
    std::array<u32, kPixelCount> comp4BottomPlaceholder{};
    SoftPackedScreenStats topScreenStats{};
    SoftPackedScreenStats bottomScreenStats{};

    void clear()
    {
        frameId = 0;
        frontBufferLatched = -1;
        screenSwapLatched = false;
        valid = false;
        hasCapture3dSource = false;
        packedTopPlane0.fill(0);
        packedTopPlane1.fill(0);
        packedTopControl.fill(0);
        packedTopLineMeta.fill(0);
        packedBottomPlane0.fill(0);
        packedBottomPlane1.fill(0);
        packedBottomControl.fill(0);
        packedBottomLineMeta.fill(0);
        capture3dSourceDsFrame.fill(0);
        captureLineUses3dMask.fill(0);
        captureFallbackLines.fill(0);
        comp4TopPlaceholder.fill(0);
        comp4BottomPlaceholder.fill(0);
        topScreenStats = {};
        bottomScreenStats = {};
    }
};

struct PreparedSoftPackedFrameDebugView
{
    u64 frameId = 0;
    int frontBufferLatched = -1;
    bool screenSwapLatched = false;
    const u32* capture3dSourceDsFrame = nullptr;
    const u8* captureLineUses3dMask = nullptr;
    const u8* captureFallbackLines = nullptr;
    const u32* comp4TopPlaceholder = nullptr;
    const u32* comp4BottomPlaceholder = nullptr;
    SoftPackedScreenStats topScreenStats{};
    SoftPackedScreenStats bottomScreenStats{};
    bool valid = false;
};

struct VulkanCompositionInputs
{
    VkImage sourceImage{VK_NULL_HANDLE};
    VkImageView sourceImageView{VK_NULL_HANDLE};
    VkImage previousTopSourceImage{VK_NULL_HANDLE};
    VkImageView previousTopSourceImageView{VK_NULL_HANDLE};
    VkImage previousBottomSourceImage{VK_NULL_HANDLE};
    VkImageView previousBottomSourceImageView{VK_NULL_HANDLE};
    VkBuffer topPackedBuffer{VK_NULL_HANDLE};
    VkBuffer bottomPackedBuffer{VK_NULL_HANDLE};
    VkBuffer capture3dBuffer{VK_NULL_HANDLE};
    VkDeviceSize packedBufferSize{};
    VkDeviceSize capture3dBufferSize{};
    u32 packedStride{};
    u32 screenSwap{};
    u32 scale{};
    u32 rendererWidth{};
    u32 rendererHeight{};
    bool previousTopSourceValid{};
    bool previousBottomSourceValid{};
    bool capture3dSourceValid{};
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
    void invalidateTemporalHistory();
    bool captureRenderer3dSnapshot(Frame* frame, const melonDS::VulkanRenderer3D& renderer3D);
    bool prepareFrameForPresentation(
        Frame* frame,
        const melonDS::GPU& gpu,
        int frontBuffer,
        bool frameScreenSwap,
        SoftPackedFrameSnapshot& softPackedSnapshot,
        melonDS::VulkanRenderer3D& renderer3D);
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
    bool isFrameReferencedAsPendingPreviousSource(const Frame* frame) const;
    bool readFramePixels(const Frame* frame, u32* destinationPixels, size_t destinationPixelCount, u64 waitTimeoutNs = UINT64_MAX);
    bool readPreparedRenderer3dPixels(
        const Frame* frame,
        u32* destinationPixels,
        size_t destinationPixelCount,
        u32& outWidth,
        u32& outHeight,
        u64 waitTimeoutNs = UINT64_MAX);
    bool getPreparedRenderer3dCaptureFrame(
        const Frame* frame,
        const u32*& outPixels,
        u32& outWidth,
        u32& outHeight) const;
    bool getPreparedRenderer3dDimensions(const Frame* frame, u32& outWidth, u32& outHeight) const;
    bool getPreparedPackedBuffers(
        const Frame* frame,
        const u32*& outTopPacked,
        const u32*& outBottomPacked,
        u32& outPackedStride,
        u32& outPackedHeight,
        bool& outScreenSwap) const;
    bool getPreparedSoftPackedFrameDebugView(
        const Frame* frame,
        PreparedSoftPackedFrameDebugView& outView) const;
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
        u32 screenSwap;
        u32 previousTopSourceValid;
        u32 previousBottomSourceValid;
        u32 captureSourceValid;
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
        VkBuffer capture3dBuffer{VK_NULL_HANDLE};
        VkDeviceMemory capture3dMemory{VK_NULL_HANDLE};
        void* capture3dMapped{};
        VkDeviceSize packedBufferSize{};
        VkImage renderer3dSnapshot{VK_NULL_HANDLE};
        VkImageView renderer3dSnapshotView{VK_NULL_HANDLE};
        VkDeviceMemory renderer3dSnapshotMemory{VK_NULL_HANDLE};
        u32 snapshotWidth{};
        u32 snapshotHeight{};
        VkImage previousTopRendererSourceImage{VK_NULL_HANDLE};
        VkImageView previousTopRendererSourceImageView{VK_NULL_HANDLE};
        bool previousTopRendererSourceValid{};
        Frame* previousTopSourceFrame{};
        bool previousTopSourcePending{};
        VkImage previousBottomRendererSourceImage{VK_NULL_HANDLE};
        VkImageView previousBottomRendererSourceImageView{VK_NULL_HANDLE};
        bool previousBottomRendererSourceValid{};
        Frame* previousBottomSourceFrame{};
        bool previousBottomSourcePending{};
        u64 softPackedFrameId{};
        int frontBufferLatched{-1};
        bool hasSoftPackedDebugData{};
        SoftPackedScreenStats topScreenStats{};
        SoftPackedScreenStats bottomScreenStats{};
        std::array<u32, SoftPackedFrameSnapshot::kPixelCount> capture3dSourceDsFrame{};
        std::array<u8, SoftPackedFrameSnapshot::kLineCount> captureLineUses3dMask{};
        std::array<u8, SoftPackedFrameSnapshot::kLineCount> captureFallbackLines{};
        std::array<u32, SoftPackedFrameSnapshot::kPixelCount> comp4TopPlaceholder{};
        std::array<u32, SoftPackedFrameSnapshot::kPixelCount> comp4BottomPlaceholder{};

        u64 submissionValue{};
        u32 width{};
        u32 height{};
        bool screenSwap{};
        bool hasContent{};
        bool hasPreparedInputs{};
        bool hasRenderer3dSnapshot{};
        bool hasPreparedCapture3dSource{};
        bool snapshotFromPreRun{};
        bool snapshotFromInitializedTarget{};
        bool snapshotFromGraphicsBackend{};
        bool descriptorSetReady{};
        bool timestampPending{};
        VkImageView cachedRendererImageView{VK_NULL_HANDLE};
        VkImageView cachedPreviousTopRendererImageView{VK_NULL_HANDLE};
        VkImageView cachedPreviousBottomRendererImageView{VK_NULL_HANDLE};
        std::array<u32, 256 * 192> preparedCapture3dSource{};
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
    bool updateCompositorPackedBuffers(Frame* frame, FrameResource& resource, const SoftPackedFrameSnapshot& softPackedSnapshot);
    bool updatePreparedCapture3dSource(
        FrameResource& resource,
        SoftPackedFrameSnapshot& softPackedSnapshot,
        const FrameResource* previousResource,
        bool currentBackendIsGraphics,
        bool currentFrameNeedsCapture3dSource,
        melonDS::VulkanRenderer3D& renderer3D);
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
    Frame* lastPreparedFrame{nullptr};
    Frame* lastTopRendererSourceFrame{nullptr};
    Frame* lastBottomRendererSourceFrame{nullptr};
    std::array<u32, SoftPackedFrameSnapshot::kPixelCount> lastValidCapture3dSource{};
    std::array<u8, SoftPackedFrameSnapshot::kLineCount> lastValidCapture3dSourceLines{};
    std::array<u32, SoftPackedFrameSnapshot::kPixelCount> lastValidTopComp4Placeholder{};
    std::array<u8, SoftPackedFrameSnapshot::kLineCount> lastValidTopComp4PlaceholderLines{};
    std::array<u32, SoftPackedFrameSnapshot::kPixelCount> lastValidBottomComp4Placeholder{};
    std::array<u8, SoftPackedFrameSnapshot::kLineCount> lastValidBottomComp4PlaceholderLines{};
    u32 packedDebugLogsRemaining{};
    PerfSampleWindow<120> packedUploadCpuWindow;
    PerfSampleWindow<120> composeCpuWindow;
    PerfSampleWindow<120> waitCpuWindow;
    PerfSampleWindow<120> compositorGpuWindow;
};

}

#endif // VULKANOUTPUT_H
