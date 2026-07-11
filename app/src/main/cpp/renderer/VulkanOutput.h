#ifndef VULKANOUTPUT_H
#define VULKANOUTPUT_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <array>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

#include "renderer/FrameQueue.h"
#include "renderer/VulkanFilterMode.h"
#include "GPU2D_HDPack.h"
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
    u32 ForceLive3dCompMode7Lines = 0;
    u32 StructuredSlotPixels = 0;
    u32 StructuredAbovePixels = 0;
    u32 Structured2DOnlyPixels = 0;
    u32 Plane0UsefulPixels = 0;
    u32 Plane0VisiblePixels = 0;
    u32 Plane0OpaqueBlackPixels = 0;
    u32 Plane1UsefulPixels = 0;
    u32 Plane1VisiblePixels = 0;
    u32 Plane1OpaqueBlackPixels = 0;
    u32 StructuredAboveVisiblePixels = 0;
    u32 StructuredAboveBlackPixels = 0;
    u32 Structured2DOnlyVisiblePixels = 0;
    u32 ProtectedBlackPixels = 0;
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
    bool captureBackedClass4Only = false;
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
    std::vector<melonDS::HDPack2DInstance> replacementInstances;

    void clear()
    {
        replacementInstances.clear();
        frameId = 0;
        frontBufferLatched = -1;
        screenSwapLatched = false;
        valid = false;
        hasCapture3dSource = false;
        captureBackedClass4Only = false;
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
    bool captureBackedClass4Only = false;
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
    VulkanFilterMode filtering{VulkanFilterMode::Nearest};
    bool previousTopSourceValid{};
    bool previousBottomSourceValid{};
    bool capture3dSourceValid{};
    bool capture3dSourceScreenSwapValid{};
    bool capture3dSourceScreenSwap{};
    bool liveSourceScreenSwap{};
    bool class4VramStructuredPair{};
    bool class4NoAboveVramStructuredPair{};
    bool class4PreservePackedVramValid{};
    bool class4PreservePackedVramScreenSwap{};
    bool topStructuredHandoffNoCurrent3d{};
    bool bottomStructuredHandoffNoCurrent3d{};
    bool topStructuredHandoffSuppress3d{};
    bool bottomStructuredHandoffSuppress3d{};
    bool needsReadback{};
    bool multiSurface{};
    bool validationMode{};
    // per-producer 2D filtering only happens in the compositor, so the
    // presenter must not take its direct path while a filter mode is active
    bool planeFilterRequested{};
};

struct VulkanOutputTemporalStats
{
    u64 FramesPrepared = 0;
    u64 FramesWithCapture3dSource = 0;
    u64 TopNeedsHighres = 0;
    u64 BottomNeedsHighres = 0;
    u64 TopPreviousSourceValid = 0;
    u64 BottomPreviousSourceValid = 0;
    u64 TopMissingHighresSource = 0;
    u64 BottomMissingHighresSource = 0;
    u64 TopStructuredSlot = 0;
    u64 BottomStructuredSlot = 0;
    u64 TopStructuredMissingAccumulator = 0;
    u64 BottomStructuredMissingAccumulator = 0;
    u64 TopAccumulatorAvailable = 0;
    u64 BottomAccumulatorAvailable = 0;
    u64 TopRegularCapture = 0;
    u64 BottomRegularCapture = 0;
    u64 TopVramCapture = 0;
    u64 BottomVramCapture = 0;
    u64 TopForceLiveCompMode7 = 0;
    u64 BottomForceLiveCompMode7 = 0;
    u64 TopCaptureBackedComp4 = 0;
    u64 BottomCaptureBackedComp4 = 0;
    u64 PackedTopOwner = 0;
    u64 PackedBottomOwner = 0;
    u64 LiveTopOwner = 0;
    u64 LiveBottomOwner = 0;
    u64 LiveOwnerOverride = 0;
    u64 SnapshotFrames = 0;
    u64 SnapshotTopOwner = 0;
    u64 SnapshotBottomOwner = 0;
    u64 SnapshotOwnerDiffersFromLive = 0;
    u64 TopPlane0UsefulPixels = 0;
    u64 TopPlane0VisiblePixels = 0;
    u64 TopPlane0OpaqueBlackPixels = 0;
    u64 TopPlane1UsefulPixels = 0;
    u64 TopPlane1VisiblePixels = 0;
    u64 TopPlane1OpaqueBlackPixels = 0;
    u64 TopStructuredAboveVisiblePixels = 0;
    u64 TopStructuredAboveBlackPixels = 0;
    u64 TopStructured2DOnlyVisiblePixels = 0;
    u64 TopProtectedBlackPixels = 0;
    u64 BottomPlane0UsefulPixels = 0;
    u64 BottomPlane0VisiblePixels = 0;
    u64 BottomPlane0OpaqueBlackPixels = 0;
    u64 BottomPlane1UsefulPixels = 0;
    u64 BottomPlane1VisiblePixels = 0;
    u64 BottomPlane1OpaqueBlackPixels = 0;
    u64 BottomStructuredAboveVisiblePixels = 0;
    u64 BottomStructuredAboveBlackPixels = 0;
    u64 BottomStructured2DOnlyVisiblePixels = 0;
    u64 BottomProtectedBlackPixels = 0;
};

class VulkanOutput
{
public:
    VulkanOutput();
    ~VulkanOutput();

    VulkanOutput(const VulkanOutput&) = delete;
    VulkanOutput& operator=(const VulkanOutput&) = delete;

    bool init();
    void shutdown();
    [[nodiscard]] bool isInitialized() const { return initialized; }

    bool ensureFrameResources(Frame* frame, u32 width, u32 height);
    // Per-producer 2D filter modes, applied through small per-mode pre-pass
    // pipelines that are created lazily when a mode is first used. Written
    // by the emulation thread, read by the presentation thread, so the modes
    // are atomics and every compose snapshots them once per frame.
    void setPacked2DFilterModes(u32 objMode, u32 bgMode)
    {
        objFilterMode.store(objMode, std::memory_order_release);
        bgFilterMode.store(bgMode, std::memory_order_release);
    }
    // Enables the HD 2D replacement overlay (sprites/BG tiles from a texture
    // pack). Resets the replacement atlas cache: the pack that backs the
    // cached images may have been rebuilt, so stale slots must not survive.
    void setReplacement2DActive(bool active);
    // Frees the large filter/overlay images (filtered planes, ScaleFX
    // intermediates, replacement atlas) when the features that need them are
    // disabled; they are recreated lazily on next use.
    void releaseUnusedHDResources();
    // Bounded wait for every in-flight frame submission; used before the
    // texture pack backing replacement instances is destroyed.
    void flushInFlightFrames();
    void invalidateTemporalHistory();
    void clearStructuredCaptureHistory();
    void releaseTemporalFrameReferences();
    bool captureRenderer3dSnapshot(Frame* frame, const melonDS::VulkanRenderer3D& renderer3D, bool snapshotScreenSwap);
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
        VulkanFilterMode filtering,
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
    VulkanOutputTemporalStats takeTemporalStatsSnapshotAndReset();

private:
    static constexpr size_t kPackedScreenWordCount =
        SoftPackedFrameSnapshot::kLineCount
        * ((SoftPackedFrameSnapshot::kScreenWidth * 3u) + 1u);

    struct CompositorPushConstants
    {
        u32 outputWidth;
        u32 outputHeight;
        u32 scale;
        u32 rendererWidth;
        u32 rendererHeight;
        u32 packedStride;
        u32 screenSwap;
        u32 filtering;
        u32 previousTopSourceValid;
        u32 previousBottomSourceValid;
        u32 captureSourceValid;
        u32 captureSourceScreenSwapValid;
        u32 captureSourceScreenSwap;
        u32 liveSourceScreenSwap;
        u32 class4VramStructuredPair;
        u32 class4NoAboveVramStructuredPair;
        u32 class4PreservePackedVramValid;
        u32 class4PreservePackedVramScreenSwap;
        u32 topStructuredHandoffNoCurrent3d;
        u32 bottomStructuredHandoffNoCurrent3d;
        u32 topStructuredHandoffSuppress3d;
        u32 bottomStructuredHandoffSuppress3d;
        u32 planeFilterActive;
    };

    struct AccumulatePushConstants
    {
        u32 scale;
        u32 packedStride;
        u32 topLcd;
    };

    struct PlaneFilterPushConstants
    {
        u32 scale;
        u32 packedStride;
        u32 applyObj;
        u32 applyBg;
        u32 writeOthers;
        u32 debugTint;
    };

    struct PlaneOverlayPushConstants
    {
        u32 scale;
        u32 instanceIndex;
        u32 debugTint;
    };

    // std430 mirror of the overlay shader's OverlayInstance struct
    struct PlaneOverlayGpuInstance
    {
        s32 destX, destY;
        u32 nativeW, nativeH;
        u32 atlasX, atlasY;
        u32 masks;  // bits 0-7 require, bits 8-15 reject
        u32 flags;  // bit 0 flipH, bit 1 flipV
    };

    struct OverlayAtlasSlot
    {
        u32 x{}, y{}, w{}, h{};
        bool uploaded{};
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
        bool hasPackedUpload{};
        int frontBufferLatched{-1};
        bool captureBackedClass4Only{};
        bool class4NoAboveVramStructuredPair{};
        bool class4PreservePackedVramValid{};
        bool class4PreservePackedVramScreenSwap{};
        bool topStructuredHandoffNoCurrent3d{};
        bool bottomStructuredHandoffNoCurrent3d{};
        bool topStructuredHandoffSuppress3d{};
        bool bottomStructuredHandoffSuppress3d{};
        bool topPackedCarryFromPrevious{};
        bool bottomPackedCarryFromPrevious{};
        bool topPureAlternatingVramCapture{};
        bool bottomPureAlternatingVramCapture{};
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
        bool screenSwapToggledFromPrevious{};
        bool hasContent{};
        bool hasPreparedInputs{};
        bool replayTopComposedFromPrevious{};
        bool replayBottomComposedFromPrevious{};
        bool replayTopComposedFromLatest{};
        Frame* previousTopComposedFrame{};
        Frame* previousBottomComposedFrame{};
        bool hasRenderer3dSnapshot{};
        bool renderer3dSnapshotScreenSwap{};
        bool hasPreparedCapture3dSource{};
        bool snapshotFromPreRun{};
        bool snapshotFromInitializedTarget{};
        bool snapshotFromGraphicsBackend{};
        bool descriptorSetReady{};
        bool timestampPending{};
        VkImageView cachedRendererImageView{VK_NULL_HANDLE};
        VkImageView cachedPreviousTopRendererImageView{VK_NULL_HANDLE};
        VkImageView cachedPreviousBottomRendererImageView{VK_NULL_HANDLE};
        VkImageView cachedTopFilteredPlaneView{VK_NULL_HANDLE};
        VkImageView cachedBottomFilteredPlaneView{VK_NULL_HANDLE};
        std::array<VkDescriptorSet, 2> planeFilterDescriptorSets{};
        bool planeFilterDescriptorsReady{};
        std::array<VkDescriptorSet, 2> scalefxDescriptorSets{};
        bool scalefxDescriptorsReady{};
        VkImageView cachedScaleFXTopPlaneView{VK_NULL_HANDLE};
        VkImageView cachedScaleFXBottomPlaneView{VK_NULL_HANDLE};
        std::vector<melonDS::HDPack2DInstance> replacementInstances;
        VkBuffer overlayInstanceBuffer{VK_NULL_HANDLE};
        VkDeviceMemory overlayInstanceMemory{VK_NULL_HANDLE};
        void* overlayInstanceMapped{};
        VkBuffer overlayStagingBuffer{VK_NULL_HANDLE};
        VkDeviceMemory overlayStagingMemory{VK_NULL_HANDLE};
        void* overlayStagingMapped{};
        VkImageView cachedOverlayTopPlaneView{VK_NULL_HANDLE};
        VkImageView cachedOverlayBottomPlaneView{VK_NULL_HANDLE};
        std::array<VkDescriptorSet, 2> overlayDescriptorSets{};
        bool overlayDescriptorsReady{};
        std::array<u32, 256 * 192> preparedCapture3dSource{};
    };

private:
    bool createSyncObjects();
    bool createCommandObjects();
    bool createCompositorResources();
    bool createCompositorPipeline();
    bool ensurePlaneFilterResources(u32 scale);
    void destroyPlaneFilterResources();
    VkPipeline getPlaneFilterPipeline(u32 mode);
    bool recordPlaneFilterPasses(FrameResource& resource, const VulkanCompositionInputs& inputs,
                                 u32 objMode, u32 bgMode, bool debugTint);
    bool ensureScaleFXResources(FrameResource& resource);
    void destroyScaleFXResources();
    VkPipeline getScaleFXPipeline(u32 pass);
    void recordScaleFXChain(FrameResource& resource, u32 screen,
                            u32 applyObj, u32 applyBg, u32 writeOthers,
                            bool debugTint,
                            const VulkanCompositionInputs& inputs);
    bool ensurePlaneOverlayResources(FrameResource& resource);
    void destroyPlaneOverlayResources();
    VkPipeline getPlaneOverlayPipeline();
    bool acquireOverlayAtlasSlot(const melonDS::HDTexPackImage* image, u32 scale,
                                 u32 nativeW, u32 nativeH, FrameResource& resource,
                                 VkDeviceSize& stagingUsed,
                                 std::vector<VkBufferImageCopy>& pendingCopies,
                                 OverlayAtlasSlot& outSlot);
    void recordPlaneOverlayPasses(FrameResource& resource, const VulkanCompositionInputs& inputs);
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
    bool recordRenderer3dSnapshotCopy(FrameResource& resource, const melonDS::VulkanRenderer3D& renderer3D, bool snapshotScreenSwap);

    bool createAccumulateResources();
    void destroyAccumulateResources();
    bool ensureAccumulatedHighresImages(u32 width, u32 height);
    void destroyAccumulatedHighresImage(VkImage& image, VkImageView& view, VkDeviceMemory& memory, bool& valid, bool& layoutReady);
    bool recordAccumulateMerge(FrameResource& resource, bool topLcd, bool replaceExisting);
    bool recordDirectPresentationPrep(
        Frame* frame,
        FrameResource& resource,
        const melonDS::VulkanRenderer3D& renderer3D,
        bool snapshotScreenSwap,
        bool accumulateTopHighres,
        bool accumulateBottomHighres,
        bool replaceAccumulatedHighres);
    bool dispatchCompositor(Frame* frame, FrameResource& resource, const VulkanCompositionInputs& inputs);
    void recordTemporalStats(
        const SoftPackedFrameSnapshot& softPackedSnapshot,
        const FrameResource& resource,
        bool topNeedsAccumulatedHighres,
        bool bottomNeedsAccumulatedHighres,
        bool topAccumulatorAvailable,
        bool bottomAccumulatorAvailable,
        bool packedScreenSwap,
        bool liveSourceScreenSwap,
        bool hasRenderer3dSnapshot,
        bool renderer3dSnapshotScreenSwap);
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

    VkImage accumulatedTopHighresImage{VK_NULL_HANDLE};
    VkImageView accumulatedTopHighresView{VK_NULL_HANDLE};
    VkDeviceMemory accumulatedTopHighresMemory{VK_NULL_HANDLE};
    bool accumulatedTopHighresValid{false};
    bool accumulatedTopHighresLayoutReady{false};
    VkImage accumulatedBottomHighresImage{VK_NULL_HANDLE};
    VkImageView accumulatedBottomHighresView{VK_NULL_HANDLE};
    VkDeviceMemory accumulatedBottomHighresMemory{VK_NULL_HANDLE};
    bool accumulatedBottomHighresValid{false};
    bool accumulatedBottomHighresLayoutReady{false};
    u32 accumulatedHighresWidth{0};
    u32 accumulatedHighresHeight{0};

    VkDescriptorSetLayout accumulateDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool accumulateDescriptorPool{VK_NULL_HANDLE};
    VkPipelineLayout accumulatePipelineLayout{VK_NULL_HANDLE};
    VkPipeline accumulatePipeline{VK_NULL_HANDLE};
    VkDescriptorSet accumulateTopDescriptorSet{VK_NULL_HANDLE};
    VkDescriptorSet accumulateBottomDescriptorSet{VK_NULL_HANDLE};
    bool accumulateTopDescriptorReady{false};
    bool accumulateBottomDescriptorReady{false};
    VkImageView cachedAccumulateTopSourceView{VK_NULL_HANDLE};
    VkImageView cachedAccumulateBottomSourceView{VK_NULL_HANDLE};

    std::atomic<u32> objFilterMode{0};
    std::atomic<u32> bgFilterMode{0};
    u64 lastEmptyPackedComposeLogNs{0};
    u64 lastPlaneFilterUnavailableLogNs{0};
    u64 statsWindowStartNs{0};
    u32 statsComposes{0};
    u32 statsSkips{0};
    u32 statsFallbacks{0};
    u32 statsPlaneFilters{0};
    u32 statsOverlayInstances{0};
    u32 statsPrevTopFromLatch{0};
    u32 statsPrevTopFromAccum{0};
    u32 statsPrevBottomFromLatch{0};
    u32 statsPrevBottomFromAccum{0};
    u32 statsPrevLatchOwnershipRejects{0};
    u32 statsPlaneFilterCacheHits{0};
    u32 statsPlaneFilterCacheMisses{0};
    bool planeFilterCacheValid{false};
    u32 planeFilterCacheScale{0};
    u32 planeFilterCacheObjMode{0};
    u32 planeFilterCacheBgMode{0};
    u64 planeFilterCacheTopHash{0};
    u64 planeFilterCacheBottomHash{0};
    bool planeFilterCacheTint{false};

    static constexpr u32 kPlaneFilterModeCount = 14;
    VkDescriptorSetLayout planeFilterDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool planeFilterDescriptorPool{VK_NULL_HANDLE};
    VkPipelineLayout planeFilterPipelineLayout{VK_NULL_HANDLE};
    std::array<VkPipeline, kPlaneFilterModeCount> planeFilterPipelines{};
    std::array<bool, kPlaneFilterModeCount> planeFilterPipelineFailed{};
    VkImage topFilteredPlaneImage{VK_NULL_HANDLE};
    VkImageView topFilteredPlaneView{VK_NULL_HANDLE};
    VkDeviceMemory topFilteredPlaneMemory{VK_NULL_HANDLE};
    VkImage bottomFilteredPlaneImage{VK_NULL_HANDLE};
    VkImageView bottomFilteredPlaneView{VK_NULL_HANDLE};
    VkDeviceMemory bottomFilteredPlaneMemory{VK_NULL_HANDLE};
    u32 filteredPlaneWidth{0};
    u32 filteredPlaneHeight{0};
    bool filteredPlaneLayoutReady{false};
    VkImage placeholderPlaneImage{VK_NULL_HANDLE};
    VkImageView placeholderPlaneView{VK_NULL_HANDLE};
    VkDeviceMemory placeholderPlaneMemory{VK_NULL_HANDLE};
    bool placeholderPlaneLayoutReady{false};

    // faithful multi-pass ScaleFX chain for plane filter mode 13; the
    // analysis intermediates live at native plane resolution (256x192 per
    // plane half, both halves stacked) and are shared by both screens
    static constexpr u32 kScaleFXPassCount = 5;
    static constexpr u32 kScaleFXImageWidth = 256;
    static constexpr u32 kScaleFXImageHeight = 192 * 2;
    VkDescriptorSetLayout scalefxDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool scalefxDescriptorPool{VK_NULL_HANDLE};
    VkPipelineLayout scalefxPipelineLayout{VK_NULL_HANDLE};
    std::array<VkPipeline, kScaleFXPassCount> scalefxPipelines{};
    bool scalefxPipelinesFailed{false};
    VkImage scalefxMetricImage{VK_NULL_HANDLE};
    VkImageView scalefxMetricView{VK_NULL_HANDLE};
    VkDeviceMemory scalefxMetricMemory{VK_NULL_HANDLE};
    VkImage scalefxStrengthImage{VK_NULL_HANDLE};
    VkImageView scalefxStrengthView{VK_NULL_HANDLE};
    VkDeviceMemory scalefxStrengthMemory{VK_NULL_HANDLE};
    VkImage scalefxFlagsImage{VK_NULL_HANDLE};
    VkImageView scalefxFlagsView{VK_NULL_HANDLE};
    VkDeviceMemory scalefxFlagsMemory{VK_NULL_HANDLE};
    VkImage scalefxCandidateImage{VK_NULL_HANDLE};
    VkImageView scalefxCandidateView{VK_NULL_HANDLE};
    VkDeviceMemory scalefxCandidateMemory{VK_NULL_HANDLE};
    bool scalefxImagesLayoutReady{false};

    // HD 2D replacement overlay: instances land on the filtered plane images
    // through a small compute pass sampling a shared replacement atlas
    static constexpr u32 kOverlayAtlasSize = 2048;
    static constexpr size_t kOverlayMaxInstances = 4096;
    static constexpr VkDeviceSize kOverlayStagingSize = 1024 * 1024;
    bool replacement2DActive{false};
    VkDescriptorSetLayout overlayDescriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool overlayDescriptorPool{VK_NULL_HANDLE};
    VkPipelineLayout overlayPipelineLayout{VK_NULL_HANDLE};
    VkPipeline overlayPipeline{VK_NULL_HANDLE};
    bool overlayPipelineFailed{false};
    VkImage overlayAtlasImage{VK_NULL_HANDLE};
    VkImageView overlayAtlasView{VK_NULL_HANDLE};
    VkDeviceMemory overlayAtlasMemory{VK_NULL_HANDLE};
    bool overlayAtlasLayoutReady{false};
    std::unordered_map<const void*, OverlayAtlasSlot> overlayAtlasSlots;
    u32 overlayAtlasShelfX{0};
    u32 overlayAtlasShelfY{0};
    u32 overlayAtlasShelfHeight{0};
    u32 overlayAtlasScale{0};
    bool overlayAtlasFull{false};
    u64 lastOverlayAtlasFullLogNs{0};

    std::unordered_map<Frame*, FrameResource> resources;
    std::mutex commandPoolLock;
    // guards FrameResource::replacementInstances: written by the emulation
    // thread (prepare) and cleared by flushInFlightFrames while the
    // presentation thread reads them during compose
    mutable std::mutex replacementInstanceLock;
    Frame* lastPreparedFrame{nullptr};
    Frame* lastTopRendererSourceFrame{nullptr};
    Frame* lastBottomRendererSourceFrame{nullptr};
    Frame* lastTopComposedFrame{nullptr};
    Frame* lastBottomComposedFrame{nullptr};
    std::vector<u32> lastValidTopPacked;
    std::vector<u32> lastValidBottomPacked;
    bool lastValidTopPackedAvailable{false};
    bool lastValidBottomPackedAvailable{false};
    bool lastPackedScreenSwapValid{false};
    bool lastPackedScreenSwap{false};
    u32 framesSinceTopLive3D{1024};
    u32 framesSinceBottomLive3D{1024};
    bool class4AsymmetricCadenceActive{};
    u32 class4AsymmetricCadencePhase{};
    bool class4BottomAboveHashValid{};
    u64 class4BottomAboveHash{};
    u32 class4BottomAboveStableFrames{};
    bool class4BottomAboveMotionActive{};
    bool class4NoAboveVramStructuredActive{};
    std::array<u32, SoftPackedFrameSnapshot::kPixelCount> lastValidCapture3dSource{};
    std::array<u8, SoftPackedFrameSnapshot::kLineCount> lastValidCapture3dSourceLines{};
    std::array<u32, SoftPackedFrameSnapshot::kPixelCount> lastValidTopComp4Placeholder{};
    std::array<u8, SoftPackedFrameSnapshot::kLineCount> lastValidTopComp4PlaceholderLines{};
    std::array<u32, SoftPackedFrameSnapshot::kPixelCount> lastValidBottomComp4Placeholder{};
    std::array<u8, SoftPackedFrameSnapshot::kLineCount> lastValidBottomComp4PlaceholderLines{};
    u32 packedDebugLogsRemaining{};
    u32 class4PairDebugLogsRemaining{};
    u32 regularComp7PackedOwnerDebugLogsRemaining{};
    u32 structuredComp7HandoffDebugLogsRemaining{};
    u32 ownershipIntroDebugLogsRemaining{};
    bool regularComp7PackedOwnerDebugActive{};
    std::mutex temporalStatsLock;
    VulkanOutputTemporalStats temporalStats{};
    PerfSampleWindow<120> packedUploadCpuWindow;
    PerfSampleWindow<120> composeCpuWindow;
    PerfSampleWindow<120> waitCpuWindow;
    PerfSampleWindow<120> compositorGpuWindow;
    u64 waitFailureInvalidFrame = 0;
    u64 waitFailureTimelineZero = 0;
    u64 waitFailureResourceMissing = 0;
    u64 waitFailureFiniteTimeout = 0;
    u64 waitFailureInfinite = 0;
};

}

#endif // VULKANOUTPUT_H
