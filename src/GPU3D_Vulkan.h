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

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#include "GPU3D.h"
#include "GPU3D_TexcacheVulkan.h"
#include "VulkanPerfStats.h"

namespace melonDS
{
class GPU;

class VulkanRenderer3D : public Renderer3D
{
public:
    static std::unique_ptr<VulkanRenderer3D> New() noexcept;

    VulkanRenderer3D() noexcept;
    ~VulkanRenderer3D() override;

    void Reset(GPU& gpu) override;
    void VCount144(GPU& gpu) override;
    void RenderFrame(GPU& gpu) override;
    void RestartFrame(GPU& gpu) override;
    u32* GetLine(int line) override;

    void SetupAccelFrame() override;
    void PrepareCaptureFrame() override;
    void Blit(const GPU& gpu) override;
    void Stop(const GPU& gpu) override;

    void SetRenderSettings(
        bool threaded,
        bool betterPolygons,
        int scale,
        bool conservativeCoverageEnabled,
        float conservativeCoveragePx,
        float conservativeCoverageDepthBias,
        bool conservativeCoverageApplyRepeat,
        bool conservativeCoverageApplyClamp,
        bool debug3dClearMagenta,
        GPU& gpu) noexcept;

    void SetThreaded(bool threaded, GPU& gpu) noexcept;
    [[nodiscard]] bool IsThreaded() const noexcept;

    [[nodiscard]] int GetScaleFactor() const noexcept { return ScaleFactor; }
    [[nodiscard]] bool UsesBetterPolygons() const noexcept { return BetterPolygons; }
    [[nodiscard]] bool IsCoverageFixEnabled() const noexcept { return CoverageFixEnabled; }
    [[nodiscard]] float GetCoverageFixPx() const noexcept { return CoverageFixPx; }
    [[nodiscard]] float GetCoverageFixDepthBias() const noexcept { return CoverageFixDepthBias; }
    [[nodiscard]] bool IsCoverageFixRepeatEnabled() const noexcept { return CoverageFixApplyRepeat; }
    [[nodiscard]] bool IsCoverageFixClampEnabled() const noexcept { return CoverageFixApplyClamp; }
    [[nodiscard]] float GetPassiveCoverageFixRepeatPx() const noexcept { return PassiveCoverageFixRepeatPx; }
    [[nodiscard]] bool IsDebug3dClearMagentaEnabled() const noexcept { return Debug3dClearMagenta; }
    [[nodiscard]] size_t GetAsyncRenderContextCount() const noexcept { return DefaultAsyncRenderContextCount; }
    [[nodiscard]] bool WaitsForReadbackSourceOnly() const noexcept { return true; }
    [[nodiscard]] bool EnsureVulkanReadyForValidation();
    [[nodiscard]] bool HasColorTarget() const noexcept { return ColorImage != VK_NULL_HANDLE && ColorImageView != VK_NULL_HANDLE; }
    [[nodiscard]] VkImage GetColorTargetImage() const noexcept { return ColorImage; }
    [[nodiscard]] VkImageView GetColorTargetImageView() const noexcept { return ColorImageView; }
    [[nodiscard]] u32 GetColorTargetWidth() const noexcept { return ColorImageWidth; }
    [[nodiscard]] u32 GetColorTargetHeight() const noexcept { return ColorImageHeight; }
    [[nodiscard]] std::vector<u32> CaptureColorTargetForDebug();
    [[nodiscard]] std::vector<u32> CaptureTopDepthForDebug();
    [[nodiscard]] std::vector<u32> CaptureTopAttrForDebug();
    [[nodiscard]] std::vector<u32> CaptureTopCoverageForDebug();
    void requestPostFastForwardDrain();

private:
    static constexpr u32 MaxTextureDescriptors = 128;
    static constexpr u32 MaxActiveTextureDescriptors = MaxTextureDescriptors - 1;
    static constexpr u32 FallbackTextureDescriptorIndex = MaxTextureDescriptors - 1;
    static constexpr u32 ToonTableEntryCount = 32;

    enum class RasterDispatchPath : u8
    {
        DirectTiles = 0,
        LegacyWorklist = 1,
    };

    enum class RasterExecutionPath : u8
    {
        CpuDirectTiles = 0,
        DirectTiles = 1,
        LegacyWorklist = 2,
    };

    struct PreviousRasterDispatchMetrics
    {
        bool Valid = false;
        u32 CoveragePercent = 0;
        u32 ActiveDispatchPercent = 0;
        bool HadContextMiss = false;
        u64 FenceWaitNs = 0;
        u32 RasterPassCount = 1;
    };

    enum class TextureSamplingPath : u8
    {
        BaseSingleDescriptor = 0,
        CompatDynamicUniform = 1,
        NonUniform = 2,
    };

    struct DescriptorSetCache
    {
        bool Ready = false;
        VkImageView ColorImageView = VK_NULL_HANDLE;
        VkBuffer TriangleBuffer = VK_NULL_HANDLE;
        VkImageView FallbackTextureView = VK_NULL_HANDLE;
        VkSampler FallbackTextureSampler = VK_NULL_HANDLE;
        VkBuffer ResultBuffer = VK_NULL_HANDLE;
        VkBuffer BinMaskBuffer = VK_NULL_HANDLE;
        VkBuffer GroupListBuffer = VK_NULL_HANDLE;
        VkBuffer ToonBuffer = VK_NULL_HANDLE;
        VkBuffer SpanSetupBuffer = VK_NULL_HANDLE;
        VkBuffer WorkOffsetBuffer = VK_NULL_HANDLE;
        VkBuffer CaptureLineBuffer = VK_NULL_HANDLE;
        std::array<VkDescriptorImageInfo, MaxTextureDescriptors> TextureInfos{};
    };

    struct RenderContext
    {
        VkCommandPool CommandPool = VK_NULL_HANDLE;
        VkCommandBuffer CommandBuffer = VK_NULL_HANDLE;
        VkFence FrameFence = VK_NULL_HANDLE;
        VkDescriptorSet DescriptorSet = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, MaxTextureDescriptors> SingleTextureDescriptorSets{};
        VkBuffer TriangleBuffer = VK_NULL_HANDLE;
        VkDeviceMemory TriangleMemory = VK_NULL_HANDLE;
        VkDeviceSize TriangleBufferSize = 0;
        void* TriangleMapped = nullptr;
        VkBuffer BinMaskBuffer = VK_NULL_HANDLE;
        VkDeviceMemory BinMaskMemory = VK_NULL_HANDLE;
        VkDeviceSize BinMaskBufferSize = 0;
        void* BinMaskMapped = nullptr;
        VkBuffer GroupListBuffer = VK_NULL_HANDLE;
        VkDeviceMemory GroupListMemory = VK_NULL_HANDLE;
        VkDeviceSize GroupListBufferSize = 0;
        void* GroupListMapped = nullptr;
        VkBuffer SpanSetupBuffer = VK_NULL_HANDLE;
        VkDeviceMemory SpanSetupMemory = VK_NULL_HANDLE;
        VkDeviceSize SpanSetupBufferSize = 0;
        void* SpanSetupMapped = nullptr;
        VkBuffer WorkOffsetBuffer = VK_NULL_HANDLE;
        VkDeviceMemory WorkOffsetMemory = VK_NULL_HANDLE;
        VkDeviceSize WorkOffsetBufferSize = 0;
        void* WorkOffsetMapped = nullptr;
        VkBuffer ToonBuffer = VK_NULL_HANDLE;
        VkDeviceMemory ToonMemory = VK_NULL_HANDLE;
        VkDeviceSize ToonBufferSize = 0;
        void* ToonMapped = nullptr;
        VkBuffer CaptureLineBuffer = VK_NULL_HANDLE;
        VkDeviceMemory CaptureLineMemory = VK_NULL_HANDLE;
        VkDeviceSize CaptureLineBufferSize = 0;
        void* CaptureLineMapped = nullptr;
        VkQueryPool TimestampQueryPool = VK_NULL_HANDLE;
        bool TimestampPending = false;
        DescriptorSetCache DescriptorCache{};
        std::array<DescriptorSetCache, MaxTextureDescriptors> SingleTextureDescriptorCaches{};
    };

    struct RasterPushConstants
    {
        u32 width;
        u32 height;
        u32 clearColor;
        u32 clearDepth;
        u32 triangleCount;
        u32 dispCnt;
        u32 alphaRef;
        u32 fogColor;
        u32 fogOffset;
        u32 fogShift;
        u32 clearAttr;
        u32 fogDensityPacked[9];
        u32 edgeColorPacked[8];
        u32 variantKey;
        u32 passIndex;
        u32 triangleBase;
        u32 depthBlendMode;
    };
    static_assert(sizeof(RasterPushConstants) == 128u, "RasterPushConstants must fit maxPushConstantsSize=128");

    struct TriangleGpu
    {
        float x0;
        float y0;
        float z0;
        float w0;
        float x1;
        float y1;
        float z1;
        float w1;
        float x2;
        float y2;
        float z2;
        float w2;
        float u0;
        float v0;
        float u1;
        float v1;
        float u2;
        float v2;
        u32 yBounds;
        u32 texLayer;
        u32 color0Rgba8;
        u32 color1Rgba8;
        u32 color2Rgba8;
        u32 flags;
        u32 texArrayIndex;
        u32 texWidth;
        u32 texHeight;
        u32 texParam;
        u32 polyAttr;
        u32 variantKey;
    };

    struct SpanSetupGpu
    {
        float minX;
        float minY;
        float maxX;
        float maxY;
        u32 yMin;
        u32 yMax;
        u32 variantKey;
        u32 valid;
        float edgeInv0;
        float edgeInv1;
        float edgeInv2;
    };

    bool ensureInitialized();
    void destroyVulkan();

    bool createCommandObjects();
    bool createCommandObjects(VkCommandPool& commandPool, VkCommandBuffer& commandBuffer);
    bool createSyncObjects();
    bool createFence(VkFence& fence);
    bool createTimestampQueryPool(VkQueryPool& queryPool);
    bool createDescriptorObjects();
    bool createComputePipeline();
    bool createPipelineCache(TextureSamplingPath samplingPath);
    void savePipelineCache();
    std::string buildPipelineCacheFileName(TextureSamplingPath samplingPath) const;

    bool ensureRenderTarget(u32 width, u32 height);
    void destroyRenderTarget();
    bool ensureTriangleBuffer(RenderContext* context, size_t triangleCount);
    void destroyTriangleBuffer(RenderContext* context);
    bool ensureCpuSpanSetupBuffer(RenderContext& context, size_t triangleCount);
    void destroyCpuSpanSetupBuffer(RenderContext& context);
    bool ensureCpuBinBuffers(RenderContext& context, size_t triangleCount, u32 width, u32 height);
    void destroyCpuBinBuffers(RenderContext& context);
    bool ensureCpuWorkOffsetBuffer(RenderContext& context, u32 width, u32 height, size_t triangleCount);
    void destroyCpuWorkOffsetBuffer(RenderContext& context);
    bool ensureResultBuffer(u32 width, u32 height);
    void destroyResultBuffer();
    bool ensureBinMaskBuffer(size_t triangleCount, u32 width, u32 height);
    void destroyBinMaskBuffer();
    bool ensureGroupListBuffer(size_t triangleCount, u32 width, u32 height);
    void destroyGroupListBuffer();
    bool ensureSpanSetupBuffer(size_t triangleCount);
    void destroySpanSetupBuffer();
    bool ensureWorkOffsetBuffer(u32 width, u32 height, size_t triangleCount);
    void destroyWorkOffsetBuffer();
    bool ensureToonBuffer(RenderContext* context);
    void destroyToonBuffer(RenderContext* context);
    bool updateToonBuffer(RenderContext* context, const u16* toonTable);
    bool ensureCaptureLineBuffer(RenderContext* context);
    void destroyCaptureLineBuffer(RenderContext* context);
    void resetCaptureLineState();
    bool finalizeCaptureLineFrame(bool blocking = true);
    bool finalizeCaptureReadback(bool blocking = true);
    bool createFallbackTexture();
    void destroyFallbackTexture();

    bool createReadbackBuffer(u32 width, u32 height);
    void destroyReadbackBuffer();
    bool ensureCaptureReadbackImage();
    void destroyCaptureReadbackImage();
    bool createResultReadbackBuffer();
    void destroyResultReadbackBuffer();

    void updateDescriptorSet(RenderContext* context, u32 singleTextureDescriptorIndex = FallbackTextureDescriptorIndex);
    static bool descriptorImageInfoEquals(const VkDescriptorImageInfo& lhs, const VkDescriptorImageInfo& rhs);
    VkDescriptorSet getDescriptorSet(RenderContext* context, u32 singleTextureDescriptorIndex) const;
    DescriptorSetCache& getDescriptorSetCache(RenderContext* context, u32 singleTextureDescriptorIndex);
    void invalidateDescriptorSetCache(RenderContext* context);
    void invalidateAllDescriptorSetCaches();
    [[nodiscard]] bool usesSingleDescriptorTexturePath() const noexcept;
    [[nodiscard]] u32 getTextureBindingDescriptorCount() const noexcept;
    [[nodiscard]] TextureSamplingPath resolveTextureSamplingPath() const noexcept;
    [[nodiscard]] static const char* textureSamplingPathName(TextureSamplingPath path) noexcept;
    u32 findMemoryType(u32 typeBits, VkMemoryPropertyFlags properties) const;
    bool tryAcquireRenderContext(RenderContext& context, bool countMisses = true);
    bool waitForRenderContext(RenderContext& context);
    RenderContext* tryAcquireReadyRenderContext() noexcept;
    bool waitForAllRenderContexts();
    bool waitForReadbackSource();
    bool waitForDeviceIdle(const char* reason);
    RenderContext& acquireNextRenderContext() noexcept;
    void consumeGpuTiming(RenderContext* context);
    void logPerformanceIfNeeded();
    bool useCpuTileBinning() const noexcept;
    RasterExecutionPath resolveRasterExecutionPath(bool isAdrenoDevice, RenderContext* context) noexcept;
    void recordRasterDispatchMetrics(
        bool isAdrenoDevice,
        RasterExecutionPath executionPath,
        u64 activeTileCount,
        u64 totalTileCount,
        u32 activeDispatchPercent,
        u32 rasterPassCount) noexcept;
    void resetRasterDispatchDecisionState() noexcept;
    bool prepareCpuTileBins(RenderContext& context, const RasterPushConstants& pushConstants);

    void WarmTextureCache(GPU& gpu);
    void buildTriangleList(GPU& gpu);

    bool dispatchRasterAndReadback(
        RenderContext* context,
        u32 rgbaColor,
        u32 clearDepth,
        u32 dispCnt,
        u32 alphaRef,
        u32 fogColor,
        u32 fogOffset,
        u32 fogShift,
        u32 clearAttr,
        const u8* fogDensityTable,
        const u16* edgeColorTable,
        const u16* toonTable,
        bool readbackToCpu,
        bool captureReadbackPath = false);
    bool readbackColorTargetToCpu(bool capturePath = false);
    bool readbackResultBufferToCpu();
    bool copyReadyCaptureLineToLineCache();
    void convertReadbackToLineCache();
    u32 buildClearColorRgba8(const GPU& gpu) const;
    void clearLineCache();

private:
    TexcacheVulkan Texcache;

    int ScaleFactor = 1;
    bool BetterPolygons = true;
    bool CoverageFixEnabled = false;
    float CoverageFixPx = 0.0f;
    float CoverageFixDepthBias = 0.0f;
    bool CoverageFixApplyRepeat = true;
    bool CoverageFixApplyClamp = false;
    float PassiveCoverageFixRepeatPx = 2.2f;
    bool Debug3dClearMagenta = false;
    bool Threaded = false;

    bool Initialized = false;
    bool InitFailed = false;
    bool HasCpuFrame = false;
    bool FrameIdentical = false;
    bool ContextAcquired = false;

    VkInstance Instance = VK_NULL_HANDLE;
    VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
    VkDevice Device = VK_NULL_HANDLE;
    VkQueue Queue = VK_NULL_HANDLE;
    u32 QueueFamilyIndex = 0;

    VkCommandPool CommandPool = VK_NULL_HANDLE;
    VkCommandBuffer CommandBuffer = VK_NULL_HANDLE;
    VkFence FrameFence = VK_NULL_HANDLE;
    VkQueryPool TimestampQueryPool = VK_NULL_HANDLE;
    bool TimestampPending = false;

    VkDescriptorSetLayout DescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet DescriptorSet = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MaxTextureDescriptors> SingleTextureDescriptorSets{};
    DescriptorSetCache DescriptorCache{};
    std::array<DescriptorSetCache, MaxTextureDescriptors> SingleTextureDescriptorCaches{};
    TextureSamplingPath ActiveTextureSamplingPath = TextureSamplingPath::CompatDynamicUniform;
    VkPipelineLayout PipelineLayout = VK_NULL_HANDLE;
    VkPipelineCache ComputePipelineCache = VK_NULL_HANDLE;
    std::string ComputePipelineCacheFile;
    VkPipeline InterpPipeline = VK_NULL_HANDLE;
    VkPipeline BinPipeline = VK_NULL_HANDLE;
    VkPipeline WorkOffsetsPipeline = VK_NULL_HANDLE;
    VkPipeline SortPipeline = VK_NULL_HANDLE;
    VkPipeline DepthBlendPipeline = VK_NULL_HANDLE;
    static constexpr u32 RasterWModeCount = 3;
    static constexpr u32 RasterShadeModeCount = 6;
    static constexpr u32 RasterTextureModeCount = 3;
    static constexpr u32 RasterTranslucencyModeCount = 3;
    static constexpr u32 RasterPipelineVariantCount =
        RasterWModeCount * RasterShadeModeCount * RasterTextureModeCount * RasterTranslucencyModeCount;
    std::array<VkPipeline, RasterPipelineVariantCount> RasterPipelines{};
    static constexpr u32 FinalPipelineVariantCount = 8;
    std::array<VkPipeline, FinalPipelineVariantCount> FinalPipelines{};
    VkPipeline CaptureLineExportPipeline = VK_NULL_HANDLE;
    static constexpr u32 ResultLayerCount = 8;
    static constexpr size_t DefaultAsyncRenderContextCount = 6;
    static constexpr size_t MaxAsyncRenderContextCount = DefaultAsyncRenderContextCount;
    static constexpr u32 TimestampQueryCount = 9;
    std::array<RenderContext, MaxAsyncRenderContextCount> RenderContexts{};
    size_t NextRenderContextIndex = 0;
    RenderContext* LastSubmittedRenderContext = nullptr;
    RasterDispatchPath ActiveRasterDispatchPath = RasterDispatchPath::DirectTiles;
    bool CpuTileBinningEnabled = false;
    bool DenseBypassActive = true;
    u32 ConsecutiveDenseFrames = 0;
    u32 ConsecutiveSparseFrames = 0;
    PreviousRasterDispatchMetrics PreviousRasterDispatchFrame{};
    u64 CurrentFrameContextWaitNs = 0;
    bool CurrentFrameHadContextMiss = false;

    VkImage ColorImage = VK_NULL_HANDLE;
    VkDeviceMemory ColorImageMemory = VK_NULL_HANDLE;
    VkImageView ColorImageView = VK_NULL_HANDLE;
    u32 ColorImageWidth = 0;
    u32 ColorImageHeight = 0;
    bool ColorImageInitialized = false;

    VkBuffer ReadbackBuffer = VK_NULL_HANDLE;
    VkDeviceMemory ReadbackMemory = VK_NULL_HANDLE;
    VkDeviceSize ReadbackSize = 0;
    void* ReadbackMapped = nullptr;
    u32 RawReadbackWidth = 0;
    u32 RawReadbackHeight = 0;
    VkImage CaptureReadbackImage = VK_NULL_HANDLE;
    VkDeviceMemory CaptureReadbackMemory = VK_NULL_HANDLE;
    bool CaptureReadbackImageInitialized = false;
    VkBuffer ResultReadbackBuffer = VK_NULL_HANDLE;
    VkDeviceMemory ResultReadbackMemory = VK_NULL_HANDLE;
    VkDeviceSize ResultReadbackSize = 0;
    void* ResultReadbackMapped = nullptr;

    VkBuffer TriangleBuffer = VK_NULL_HANDLE;
    VkDeviceMemory TriangleMemory = VK_NULL_HANDLE;
    VkDeviceSize TriangleBufferSize = 0;
    void* TriangleMapped = nullptr;

    VkBuffer ResultBuffer = VK_NULL_HANDLE;
    VkDeviceMemory ResultMemory = VK_NULL_HANDLE;
    VkDeviceSize ResultBufferSize = 0;

    VkBuffer BinMaskBuffer = VK_NULL_HANDLE;
    VkDeviceMemory BinMaskMemory = VK_NULL_HANDLE;
    VkDeviceSize BinMaskBufferSize = 0;

    VkBuffer GroupListBuffer = VK_NULL_HANDLE;
    VkDeviceMemory GroupListMemory = VK_NULL_HANDLE;
    VkDeviceSize GroupListBufferSize = 0;

    VkBuffer SpanSetupBuffer = VK_NULL_HANDLE;
    VkDeviceMemory SpanSetupMemory = VK_NULL_HANDLE;
    VkDeviceSize SpanSetupBufferSize = 0;

    VkBuffer WorkOffsetBuffer = VK_NULL_HANDLE;
    VkDeviceMemory WorkOffsetMemory = VK_NULL_HANDLE;
    VkDeviceSize WorkOffsetBufferSize = 0;

    VkBuffer ToonBuffer = VK_NULL_HANDLE;
    VkDeviceMemory ToonMemory = VK_NULL_HANDLE;
    VkDeviceSize ToonBufferSize = 0;
    void* ToonMapped = nullptr;
    VkBuffer CaptureLineBuffer = VK_NULL_HANDLE;
    VkDeviceMemory CaptureLineMemory = VK_NULL_HANDLE;
    VkDeviceSize CaptureLineBufferSize = 0;
    void* CaptureLineMapped = nullptr;

    VkImage FallbackTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory FallbackTextureMemory = VK_NULL_HANDLE;
    VkImageView FallbackTextureView = VK_NULL_HANDLE;
    VkSampler FallbackTextureSampler = VK_NULL_HANDLE;
    VkBuffer FallbackTextureStagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory FallbackTextureStagingMemory = VK_NULL_HANDLE;

    std::array<VkDescriptorImageInfo, MaxTextureDescriptors> ActiveTextureDescriptors{};
    u32 ActiveTextureDescriptorCount = 0;

    std::vector<TriangleGpu> Triangles;
    std::vector<u32> RawReadbackRgba;
    std::vector<u32> RawResultReadback;
    std::array<u32, 256 * 192> LineCache{};
    PFN_vkResetQueryPoolEXT ResetQueryPool = nullptr;
    float TimestampPeriodNs = 0.0f;
    bool TimestampQueriesSupported = false;
    PerfSampleWindow<120> RenderCpuWindow;
    PerfSampleWindow<120> FenceWaitCpuWindow;
    PerfSampleWindow<120> GpuWindow;
    PerfSampleWindow<120> TriangleCountWindow;
    PerfSampleWindow<120> PassCountWindow;
    PerfSampleWindow<120> InterpCpuWindow;
    PerfSampleWindow<120> BinCpuWindow;
    PerfSampleWindow<120> WorkOffsetsCpuWindow;
    PerfSampleWindow<120> SortCpuWindow;
    PerfSampleWindow<120> RasterCpuWindow;
    PerfSampleWindow<120> DepthBlendCpuWindow;
    PerfSampleWindow<120> FinalCpuWindow;
    PerfSampleWindow<120> CaptureLineExportCpuWindow;
    PerfSampleWindow<120> CpuActiveTileCountWindow;
    PerfSampleWindow<120> CpuTileCountWindow;
    PerfSampleWindow<120> CpuActiveGroupCountWindow;
    PerfSampleWindow<120> CpuActiveDispatchWindow;
    PerfSampleWindow<120> InterpGpuWindow;
    PerfSampleWindow<120> BinGpuWindow;
    PerfSampleWindow<120> WorkOffsetsGpuWindow;
    PerfSampleWindow<120> SortGpuWindow;
    PerfSampleWindow<120> RasterGpuWindow;
    PerfSampleWindow<120> DepthBlendGpuWindow;
    PerfSampleWindow<120> FinalGpuWindow;
    PerfSampleWindow<120> CaptureLineExportGpuWindow;
    PerfSampleWindow<120> EarlySubmitCpuWindow;
    PerfSampleWindow<120> EarlySubmitContextWaitCpuWindow;
    u64 ContextMissCount = 0;
    u64 LateFrameCount = 0;
    u64 DroppedFrameCount = 0;
    u64 CpuDirectTilesPathCount = 0;
    u64 DirectTilesPathCount = 0;
    u64 LegacyWorklistPathCount = 0;
    u64 AutoDecisionCpuCount = 0;
    u64 AutoDecisionDirectCount = 0;
    u64 ForcedCpuDecisionCount = 0;
    u64 ForcedDirectDecisionCount = 0;
    u64 ForcedLegacyDecisionCount = 0;
    u64 DenseBypassEnterCount = 0;
    u64 DenseBypassExitCount = 0;
    u64 ReadbackColorRequestCount = 0;
    u64 ReadbackResultRequestCount = 0;
    u64 CapturePrepareRequestCount = 0;
    std::array<u64, 4> CaptureModeCounts{};
    std::array<u64, 4> CaptureSizeModeCounts{};
    u64 CaptureSource3dCount = 0;
    u64 CaptureEnabledCount = 0;
    u64 CaptureLineExportCount = 0;
    u64 RasterSpecializedShadeModeCount = 0;
    u64 RasterSpecializedTextureModeCount = 0;
    u64 RasterSpecializedTranslucencyModeCount = 0;
    u64 RasterSpecializedAllModesCount = 0;
    u64 EarlySubmitAttemptCount = 0;
    u64 EarlySubmitHitCount = 0;
    u64 EarlySubmitMissCount = 0;
    u64 EarlySubmitSkipVCount215Count = 0;
    bool SkipRenderAtVCount215 = false;
    bool InEarlySubmitAttempt = false;
    u64 CurrentEarlySubmitContextWaitNs = 0;
    bool CaptureReadbackPending = false;
    RenderContext* PendingCaptureReadbackContext = nullptr;
    bool CaptureLinePending = false;
    bool CaptureLineReady = false;
    RenderContext* PendingCaptureLineContext = nullptr;
    const u32* ReadyCaptureLineData = nullptr;
    u32 PostFastForwardDrainFrames = 0;
};
}
