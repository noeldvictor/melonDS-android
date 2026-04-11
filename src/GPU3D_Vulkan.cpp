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

#include "GPU3D_Vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <vector>

#include "GPU.h"
#include "GPU3D_Vulkan_BinCombinedShaderData.h"
#include "GPU3D_Vulkan_CalculateWorkOffsetsShaderData.h"
#include "GPU3D_Vulkan_CaptureLineExportShaderData.h"
#include "GPU3D_Vulkan_DepthBlendShaderData.h"
#include "GPU3D_Vulkan_FinalPassShaderData.h"
#include "GPU3D_Vulkan_InterpSpansShaderData.h"
#include "GPU3D_Vulkan_SortWorkShaderData.h"
#include "GPU3D_Vulkan_TriRasterCompatShaderData.h"
#include "GPU3D_Vulkan_TriRasterShaderData.h"
#include "Platform.h"
#include "VulkanContext.h"
#include "version.h"

namespace MelonDSAndroid
{
bool isFastForwardActive();
bool areRendererDebugToolsEnabled();
melonDS::u32 getVulkanDiagnosticFlags();
}

namespace melonDS
{
using Platform::Log;
using Platform::LogLevel;

namespace
{
constexpr u32 kPipelineCacheFileVersion = 1;
constexpr u32 kVulkanDiagnosticDisablePassiveRepeatCoverageExpand = 1u << 0u;
constexpr float kTriangleAreaEpsilon = 0.000001f;
constexpr float kTileOverlapEpsilon = 0.00001f;
constexpr u32 kWorkSortDispatchX = 0u;
constexpr u32 kWorkSortDispatchY = 1u;
constexpr u32 kWorkSortDispatchZ = 2u;
constexpr u32 kWorkRasterDispatchX = 3u;
constexpr u32 kWorkRasterDispatchY = 4u;
constexpr u32 kWorkRasterDispatchZ = 5u;
constexpr u32 kWorkActiveTileCount = 6u;
constexpr u32 kWorkActiveGroupCount = 7u;
constexpr u32 kWorkTileOffsetsBase = 8u;
constexpr u32 kCpuActiveTileDispatchMaxCoveragePercent = 90u;

u64 fnv1a64(const char* value)
{
    constexpr u64 kOffsetBasis = 14695981039346656037ull;
    constexpr u64 kPrime = 1099511628211ull;

    u64 hash = kOffsetBasis;
    if (value == nullptr)
        return hash;

    for (const unsigned char* cursor = reinterpret_cast<const unsigned char*>(value); *cursor != 0; cursor++)
    {
        hash ^= static_cast<u64>(*cursor);
        hash *= kPrime;
    }

    return hash;
}

inline float edgeFunction2D(float ax, float ay, float bx, float by, float px, float py)
{
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

struct EdgeEquation2D
{
    float a;
    float b;
    float c;
};

inline EdgeEquation2D makeEdgeEquation2D(float ax, float ay, float bx, float by)
{
    return {
        by - ay,
        ax - bx,
        (bx * ay) - (ax * by),
    };
}

inline float evaluateEdge2D(const EdgeEquation2D& edge, float px, float py)
{
    return edge.a * px + edge.b * py + edge.c;
}

inline bool edgeSeparatesTile(float e0, float e1, float e2, float e3, bool positiveArea)
{
    if (positiveArea)
        return e0 < -kTileOverlapEpsilon && e1 < -kTileOverlapEpsilon && e2 < -kTileOverlapEpsilon && e3 < -kTileOverlapEpsilon;
    return e0 > kTileOverlapEpsilon && e1 > kTileOverlapEpsilon && e2 > kTileOverlapEpsilon && e3 > kTileOverlapEpsilon;
}

VkBufferCreateInfo makeBufferCreateInfo(VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = size;
    createInfo.usage = usage;
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    return createInfo;
}

VkMemoryAllocateInfo makeMemoryAllocateInfo(VkDeviceSize allocationSize, u32 memoryTypeIndex)
{
    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = allocationSize;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;
    return allocateInfo;
}

VkWriteDescriptorSet makeImageDescriptorWrite(
    VkDescriptorSet descriptorSet,
    u32 binding,
    const VkDescriptorImageInfo* imageInfo,
    u32 descriptorCount,
    VkDescriptorType descriptorType)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.descriptorCount = descriptorCount;
    write.descriptorType = descriptorType;
    write.pImageInfo = imageInfo;
    return write;
}

VkWriteDescriptorSet makeBufferDescriptorWrite(
    VkDescriptorSet descriptorSet,
    u32 binding,
    const VkDescriptorBufferInfo* bufferInfo,
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = descriptorType;
    write.pBufferInfo = bufferInfo;
    return write;
}

template <typename FindMemoryTypeFn>
bool createBufferAllocation(
    VkDevice device,
    FindMemoryTypeFn&& findMemoryType,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags preferredProperties,
    VkMemoryPropertyFlags fallbackProperties,
    VkBuffer& buffer,
    VkDeviceMemory& memory,
    void** mappedMemory = nullptr)
{
    VkBufferCreateInfo bufferCreateInfo = makeBufferCreateInfo(size, usage);
    if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
        return false;

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

    u32 memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, preferredProperties);
    if (memoryTypeIndex == UINT32_MAX && fallbackProperties != preferredProperties)
        memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, fallbackProperties);
    if (memoryTypeIndex == UINT32_MAX)
    {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateInfo memoryAllocateInfo = makeMemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex);
    if (vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory) != VK_SUCCESS)
    {
        vkDestroyBuffer(device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS)
    {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        memory = VK_NULL_HANDLE;
        buffer = VK_NULL_HANDLE;
        return false;
    }

    if (mappedMemory != nullptr && vkMapMemory(device, memory, 0, size, 0, mappedMemory) != VK_SUCCESS)
    {
        vkFreeMemory(device, memory, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        memory = VK_NULL_HANDLE;
        buffer = VK_NULL_HANDLE;
        *mappedMemory = nullptr;
        return false;
    }

    return true;
}
}

std::unique_ptr<VulkanRenderer3D> VulkanRenderer3D::New() noexcept
{
    return std::make_unique<VulkanRenderer3D>();
}

VulkanRenderer3D::VulkanRenderer3D() noexcept
    : Renderer3D(true)
    , Texcache(TexcacheVulkanLoader())
{
    clearLineCache();
}

VulkanRenderer3D::~VulkanRenderer3D()
{
    destroyVulkan();
}

void VulkanRenderer3D::Reset(GPU& gpu)
{
    (void)gpu;
    Texcache.Reset();
    HasCpuFrame = false;
    FrameIdentical = false;
    LastSubmittedRenderContext = nullptr;
    SkipRenderAtVCount215 = false;
    InEarlySubmitAttempt = false;
    CurrentEarlySubmitContextWaitNs = 0;
    CaptureReadbackPending = false;
    PendingCaptureReadbackContext = nullptr;
    resetCaptureLineState();
    clearLineCache();
    CaptureLineExportCount = 0;
    EarlySubmitAttemptCount = 0;
    EarlySubmitHitCount = 0;
    EarlySubmitMissCount = 0;
    EarlySubmitSkipVCount215Count = 0;
}

void VulkanRenderer3D::VCount144(GPU& gpu)
{
    SkipRenderAtVCount215 = false;

    if (!Threaded)
        return;

    const u32 captureCnt = gpu.GPU2D_A.CaptureCnt;
    const bool captureEnabled = (captureCnt & (1u << 31u)) != 0u;
    const u32 captureMode = (captureCnt >> 29u) & 0x3u;
    const bool captureNeedsSourceA = captureEnabled && (captureMode != 1u);
    if (!captureNeedsSourceA)
        return;

    if (!ensureInitialized())
        return;

    if (VulkanContext::Get().GetDeviceProfile().IsAdreno)
        return;

    EarlySubmitAttemptCount++;
    InEarlySubmitAttempt = true;
    CurrentEarlySubmitContextWaitNs = 0;
    const u64 earlySubmitStartNs = PerfNowNs();
    RenderFrame(gpu);
    EarlySubmitCpuWindow.Add(PerfNowNs() - earlySubmitStartNs);
    EarlySubmitContextWaitCpuWindow.Add(CurrentEarlySubmitContextWaitNs);
    InEarlySubmitAttempt = false;
    CurrentEarlySubmitContextWaitNs = 0;
    SkipRenderAtVCount215 = Initialized && ColorImageInitialized;
    if (SkipRenderAtVCount215)
        EarlySubmitHitCount++;
    else
        EarlySubmitMissCount++;
}

void VulkanRenderer3D::RenderFrame(GPU& gpu)
{
    if (SkipRenderAtVCount215 && gpu.VCount == 215u)
    {
        SkipRenderAtVCount215 = false;
        EarlySubmitSkipVCount215Count++;
        return;
    }

    const u64 renderStartNs = PerfNowNs();
    auto renderPerfScope = MakeScopeExit([&]() {
        RenderCpuWindow.Add(PerfNowNs() - renderStartNs);
        logPerformanceIfNeeded();
    });

    const bool textureCacheChanged = Texcache.Update(gpu);
    WarmTextureCache(gpu);

    const u32 scale = static_cast<u32>(std::max(1, ScaleFactor));
    const u32 targetWidth = 256u * scale;
    const u32 targetHeight = 192u * scale;
    FrameIdentical = !textureCacheChanged && gpu.GPU3D.RenderFrameIdentical;
    const bool canReuseIdenticalFrame = FrameIdentical
        && Initialized
        && ColorImageInitialized
        && HasColorTarget()
        && ColorImageWidth == targetWidth
        && ColorImageHeight == targetHeight;
    if (canReuseIdenticalFrame)
        return;

    if (!ensureInitialized())
    {
        HasCpuFrame = false;
        return;
    }

    const VulkanDeviceProfile& deviceProfile = VulkanContext::Get().GetDeviceProfile();
    const u32 captureCnt = gpu.GPU2D_A.CaptureCnt;
    const bool captureEnabled = (captureCnt & (1u << 31u)) != 0u;
    const u32 captureMode = (captureCnt >> 29u) & 0x3u;
    const u32 captureSizeMode = (captureCnt >> 20u) & 0x3u;
    const bool captureSource3d = (captureCnt & (1u << 24u)) != 0u;
    const bool preservePreparedCpuCapture = HasCpuFrame && captureEnabled && (captureMode != 1u);
    const bool captureNeedsCpuReadback = captureEnabled && (captureMode != 1u) && deviceProfile.IsAdreno;
    // Adreno stays on the stable readback capture path here because the
    // VCount144 export path can cross top/bottom when both DS screens depend
    // on 3D in the same frame.
    const bool captureNeedsGpuCaptureLine = captureEnabled && (captureMode != 1u) && !deviceProfile.IsAdreno;

    CpuTileBinningEnabled = Threaded && deviceProfile.IsAdreno;

    if (!ensureRenderTarget(targetWidth, targetHeight))
    {
        HasCpuFrame = false;
        return;
    }

    if (!ensureResultBuffer(targetWidth, targetHeight))
    {
        HasCpuFrame = false;
        return;
    }

    RenderContext* renderContext = nullptr;
    if (Threaded)
    {
        renderContext = tryAcquireReadyRenderContext();
        bool renderContextReady = renderContext != nullptr;
        if (!renderContextReady)
        {
            renderContext = &acquireNextRenderContext();
            const bool useNonBlockingAcquire = MelonDSAndroid::isFastForwardActive() || PostFastForwardDrainFrames > 0;
            if (useNonBlockingAcquire)
            {
                renderContextReady = renderContext != nullptr && tryAcquireRenderContext(*renderContext);
            }
            else
            {
                const u64 contextWaitStartNs = PerfNowNs();
                renderContextReady = renderContext != nullptr && waitForRenderContext(*renderContext);
                if (InEarlySubmitAttempt)
                    CurrentEarlySubmitContextWaitNs += (PerfNowNs() - contextWaitStartNs);
            }
        }
        if (PostFastForwardDrainFrames > 0)
            PostFastForwardDrainFrames--;
        if (!renderContextReady)
        {
            HasCpuFrame = false;
            return;
        }
    }

    buildTriangleList(gpu);
    if (!ensureTriangleBuffer(renderContext, Triangles.size()))
    {
        HasCpuFrame = false;
        return;
    }

    const bool useContextCpuTileBinning = renderContext != nullptr && useCpuTileBinning();
    if (!useContextCpuTileBinning && !ensureBinMaskBuffer(Triangles.size(), targetWidth, targetHeight))
    {
        HasCpuFrame = false;
        return;
    }

    if (!useContextCpuTileBinning && !ensureGroupListBuffer(Triangles.size(), targetWidth, targetHeight))
    {
        HasCpuFrame = false;
        return;
    }

    if (useContextCpuTileBinning && !ensureCpuBinBuffers(*renderContext, Triangles.size(), targetWidth, targetHeight))
    {
        HasCpuFrame = false;
        return;
    }

    if (useContextCpuTileBinning && !ensureCpuSpanSetupBuffer(*renderContext, Triangles.size()))
    {
        HasCpuFrame = false;
        return;
    }

    if (!useContextCpuTileBinning && !ensureSpanSetupBuffer(Triangles.size()))
    {
        HasCpuFrame = false;
        return;
    }

    if (useContextCpuTileBinning && !ensureCpuWorkOffsetBuffer(*renderContext, targetWidth, targetHeight, Triangles.size()))
    {
        HasCpuFrame = false;
        return;
    }

    if (!useContextCpuTileBinning && !ensureWorkOffsetBuffer(targetWidth, targetHeight, Triangles.size()))
    {
        HasCpuFrame = false;
        return;
    }

    if (!ensureToonBuffer(renderContext))
    {
        HasCpuFrame = false;
        return;
    }

    if (!ensureCaptureLineBuffer(renderContext))
    {
        HasCpuFrame = false;
        return;
    }

    updateDescriptorSet(renderContext);

    if (captureEnabled)
    {
        CaptureEnabledCount++;
        CaptureModeCounts[captureMode]++;
        CaptureSizeModeCounts[captureSizeMode]++;
        if (captureSource3d)
            CaptureSource3dCount++;
    }

    const u32 clearColor = Debug3dClearMagenta ? 0xFFFF00FFu : buildClearColorRgba8(gpu);
    const u32 clearDepth = ((gpu.GPU3D.RenderClearAttr2 & 0x7FFFu) * 0x200u) + 0x1FFu;
    if (!dispatchRasterAndReadback(
            renderContext,
            clearColor,
            clearDepth,
            gpu.GPU3D.RenderDispCnt,
            gpu.GPU3D.RenderAlphaRef,
            gpu.GPU3D.RenderFogColor,
            gpu.GPU3D.RenderFogOffset,
            gpu.GPU3D.RenderFogShift,
            gpu.GPU3D.RenderClearAttr1 & 0x3F008000u,
            gpu.GPU3D.RenderFogDensityTable,
            gpu.GPU3D.RenderEdgeTable,
            gpu.GPU3D.RenderToonTable,
            captureNeedsCpuReadback,
            captureNeedsGpuCaptureLine))
    {
        HasCpuFrame = false;
        return;
    }

    if (!captureNeedsCpuReadback)
        HasCpuFrame = preservePreparedCpuCapture;

    if (!captureNeedsGpuCaptureLine)
    {
        if (!captureNeedsCpuReadback)
        {
            CaptureReadbackPending = false;
            PendingCaptureReadbackContext = nullptr;
        }
        resetCaptureLineState();
    }
}

void VulkanRenderer3D::RestartFrame(GPU& gpu)
{
    (void)gpu;
}

u32* VulkanRenderer3D::GetLine(int line)
{
    if (CaptureLinePending)
    {
        if (!finalizeCaptureLineFrame())
            resetCaptureLineState();
    }
    else if (CaptureLineReady && ReadyCaptureLineData == nullptr)
    {
        resetCaptureLineState();
    }

    if (!HasCpuFrame && CaptureReadbackPending && finalizeCaptureReadback(true))
        convertReadbackToLineCache();

    if (!HasCpuFrame)
    {
        if (CaptureLineReady && ReadyCaptureLineData != nullptr)
        {
            // DS-sized capture lines are written directly by CaptureLineExport.
            HasCpuFrame = true;
        }
        else if (readbackColorTargetToCpu(true))
            convertReadbackToLineCache();
        else
            clearLineCache();
    }

    if (line < 0)
        line = 0;
    else if (line > 191)
        line = 191;

    if (CaptureLineReady && ReadyCaptureLineData != nullptr)
        return const_cast<u32*>(&ReadyCaptureLineData[static_cast<size_t>(line) * 256u]);

    return &LineCache[static_cast<size_t>(line) * 256u];
}

void VulkanRenderer3D::SetupAccelFrame()
{
}

void VulkanRenderer3D::PrepareCaptureFrame()
{
    CapturePrepareRequestCount++;

    if (!HasCpuFrame && CaptureReadbackPending)
    {
        if (finalizeCaptureReadback(false))
        {
            convertReadbackToLineCache();
            return;
        }
        if (CaptureReadbackPending)
            return;
    }

    if (CaptureLinePending || CaptureLineReady)
    {
        if (finalizeCaptureLineFrame(false))
            return;
        if (CaptureLinePending)
            return;
    }

    if (HasCpuFrame && RawReadbackWidth > 0 && RawReadbackHeight > 0 && !RawReadbackRgba.empty())
    {
        convertReadbackToLineCache();
        return;
    }

    if (!readbackColorTargetToCpu(true))
    {
        HasCpuFrame = false;
        clearLineCache();
        return;
    }

    convertReadbackToLineCache();
}

void VulkanRenderer3D::Blit(const GPU& gpu)
{
    (void)gpu;
}

void VulkanRenderer3D::Stop(const GPU& gpu)
{
    (void)gpu;
    Texcache.Reset();
    destroyVulkan();
    InitFailed = false;
    HasCpuFrame = false;
    SkipRenderAtVCount215 = false;
    InEarlySubmitAttempt = false;
    CurrentEarlySubmitContextWaitNs = 0;
    resetCaptureLineState();
    clearLineCache();
    CaptureLineExportCount = 0;
    EarlySubmitAttemptCount = 0;
    EarlySubmitHitCount = 0;
    EarlySubmitMissCount = 0;
    EarlySubmitSkipVCount215Count = 0;
}

void VulkanRenderer3D::SetRenderSettings(
    bool threaded,
    bool betterPolygons,
    int scale,
    bool conservativeCoverageEnabled,
    float conservativeCoveragePx,
    float conservativeCoverageDepthBias,
    bool conservativeCoverageApplyRepeat,
    bool conservativeCoverageApplyClamp,
    bool debug3dClearMagenta,
    GPU& gpu) noexcept
{
    SetThreaded(threaded, gpu);

    const int oldScale = ScaleFactor;
    BetterPolygons = betterPolygons;
    ScaleFactor = std::max(1, scale);
    CoverageFixEnabled = conservativeCoverageEnabled;
    CoverageFixPx = std::clamp(conservativeCoveragePx, 0.0f, 2.0f);
    CoverageFixDepthBias = std::clamp(conservativeCoverageDepthBias, 0.0f, 0.01f);
    CoverageFixApplyRepeat = conservativeCoverageApplyRepeat;
    CoverageFixApplyClamp = conservativeCoverageApplyClamp;
    Debug3dClearMagenta = debug3dClearMagenta;

    if (Initialized && oldScale != ScaleFactor)
    {
        (void)waitForDeviceIdle("scale change");
        destroyTriangleBuffer(nullptr);
        for (RenderContext& renderContext : RenderContexts)
        {
            destroyCpuSpanSetupBuffer(renderContext);
            destroyCpuBinBuffers(renderContext);
            destroyCpuWorkOffsetBuffer(renderContext);
            destroyTriangleBuffer(&renderContext);
        }
        destroyBinMaskBuffer();
        destroyGroupListBuffer();
        destroySpanSetupBuffer();
        destroyWorkOffsetBuffer();
        destroyToonBuffer(nullptr);
        for (RenderContext& renderContext : RenderContexts)
            destroyToonBuffer(&renderContext);
        destroyResultBuffer();
        destroyRenderTarget();
        destroyReadbackBuffer();
        FrameIdentical = false;
        HasCpuFrame = false;
        resetCaptureLineState();
        clearLineCache();
    }
}

void VulkanRenderer3D::SetThreaded(bool threaded, GPU& gpu) noexcept
{
    (void)gpu;
    const bool enableThreaded = threaded;
    if (Threaded == enableThreaded)
        return;

    if (Initialized)
        (void)waitForDeviceIdle("threaded mode change");

    Threaded = enableThreaded;
    NextRenderContextIndex = 0;
    LastSubmittedRenderContext = nullptr;
    CaptureReadbackPending = false;
    PendingCaptureReadbackContext = nullptr;
    HasCpuFrame = false;
    resetCaptureLineState();
    clearLineCache();
    if (MelonDSAndroid::areRendererDebugToolsEnabled())
        Log(LogLevel::Warn, "VulkanRenderer3D: threaded rendering %s", Threaded ? "enabled" : "disabled");
}

bool VulkanRenderer3D::IsThreaded() const noexcept
{
    return Threaded;
}

std::vector<u32> VulkanRenderer3D::CaptureColorTargetForDebug()
{
    if (!ensureInitialized() || ColorImage == VK_NULL_HANDLE || ColorImageWidth == 0 || ColorImageHeight == 0)
        return {};

    if (!readbackColorTargetToCpu(false))
        return {};

    return RawReadbackRgba;
}

std::vector<u32> VulkanRenderer3D::CaptureTopDepthForDebug()
{
    if (!ensureInitialized() || ResultBuffer == VK_NULL_HANDLE || ColorImageWidth == 0 || ColorImageHeight == 0)
        return {};

    if (!readbackResultBufferToCpu())
        return {};

    const size_t pixelCount = static_cast<size_t>(ColorImageWidth) * static_cast<size_t>(ColorImageHeight);
    if (RawResultReadback.size() < pixelCount * ResultLayerCount)
        return {};

    const auto depthBegin = RawResultReadback.begin() + static_cast<std::ptrdiff_t>(pixelCount * 2u);
    return std::vector<u32>(depthBegin, depthBegin + static_cast<std::ptrdiff_t>(pixelCount));
}

std::vector<u32> VulkanRenderer3D::CaptureTopAttrForDebug()
{
    if (!ensureInitialized() || ResultBuffer == VK_NULL_HANDLE || ColorImageWidth == 0 || ColorImageHeight == 0)
        return {};

    if (!readbackResultBufferToCpu())
        return {};

    const size_t pixelCount = static_cast<size_t>(ColorImageWidth) * static_cast<size_t>(ColorImageHeight);
    if (RawResultReadback.size() < pixelCount * ResultLayerCount)
        return {};

    const auto attrBegin = RawResultReadback.begin() + static_cast<std::ptrdiff_t>(pixelCount * 4u);
    return std::vector<u32>(attrBegin, attrBegin + static_cast<std::ptrdiff_t>(pixelCount));
}

std::vector<u32> VulkanRenderer3D::CaptureTopCoverageForDebug()
{
    const std::vector<u32> topAttr = CaptureTopAttrForDebug();
    if (topAttr.empty())
        return {};

    std::vector<u32> coverage(topAttr.size(), 0u);
    for (size_t i = 0; i < topAttr.size(); i++)
        coverage[i] = (topAttr[i] >> 8u) & 0x1Fu;
    return coverage;
}

bool VulkanRenderer3D::EnsureVulkanReadyForValidation()
{
    if (!ensureInitialized())
        return false;

    constexpr u32 kValidationWidth = 256;
    constexpr u32 kValidationHeight = 192;

    if (!ensureRenderTarget(kValidationWidth, kValidationHeight))
        return false;
    if (!ensureResultBuffer(kValidationWidth, kValidationHeight))
        return false;
    if (!ensureTriangleBuffer(nullptr, 1))
        return false;
    if (!ensureBinMaskBuffer(0, kValidationWidth, kValidationHeight))
        return false;
    if (!ensureGroupListBuffer(0, kValidationWidth, kValidationHeight))
        return false;
    if (!ensureSpanSetupBuffer(1))
        return false;
    if (!ensureWorkOffsetBuffer(kValidationWidth, kValidationHeight, 1))
        return false;
    if (!ensureToonBuffer(nullptr))
        return false;
    if (!ensureCaptureLineBuffer(nullptr))
        return false;

    Triangles.clear();
    ActiveTextureDescriptorCount = 0;
    ActiveTextureDescriptors.fill(VkDescriptorImageInfo{});
    updateDescriptorSet(nullptr);

    std::array<u8, 34> fogDensity{};
    std::array<u16, 8> edgeColors{};
    std::array<u16, 32> toonTable{};

    return dispatchRasterAndReadback(
        nullptr,
        0xFF000000u,
        0x00FFFFFFu,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        fogDensity.data(),
        edgeColors.data(),
        toonTable.data(),
        false
    );
}

bool VulkanRenderer3D::ensureInitialized()
{
    if (Initialized)
        return true;

    if (InitFailed)
        return false;

    if (!VulkanContext::Get().Acquire())
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to acquire shared Vulkan context");
        InitFailed = true;
        return false;
    }

    ContextAcquired = true;
    Instance = VulkanContext::Get().GetInstance();
    PhysicalDevice = VulkanContext::Get().GetPhysicalDevice();
    Device = VulkanContext::Get().GetDevice();
    Queue = VulkanContext::Get().GetQueue();
    QueueFamilyIndex = VulkanContext::Get().GetQueueFamilyIndex();
    ResetQueryPool = VulkanContext::Get().GetResetQueryPool();
    TimestampPeriodNs = VulkanContext::Get().GetTimestampPeriod();
    TimestampQueriesSupported = VulkanContext::Get().SupportsTimestamps();

    if (!createCommandObjects() || !createSyncObjects() || !createDescriptorObjects() || !createComputePipeline())
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to initialize Vulkan resources");
        destroyVulkan();
        InitFailed = true;
        return false;
    }

    Initialized = true;
    return true;
}

void VulkanRenderer3D::destroyVulkan()
{
    if (Device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(Device);

    CaptureReadbackPending = false;
    PendingCaptureReadbackContext = nullptr;
    resetCaptureLineState();
    destroyReadbackBuffer();
    destroyCaptureReadbackImage();
    destroyResultReadbackBuffer();
    destroyTriangleBuffer(nullptr);
    destroyBinMaskBuffer();
    destroyGroupListBuffer();
    destroySpanSetupBuffer();
    destroyWorkOffsetBuffer();
    destroyToonBuffer(nullptr);
    destroyCaptureLineBuffer(nullptr);
    destroyResultBuffer();
    destroyFallbackTexture();
    destroyRenderTarget();

    if (InterpPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(Device, InterpPipeline, nullptr);
        InterpPipeline = VK_NULL_HANDLE;
    }

    if (BinPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(Device, BinPipeline, nullptr);
        BinPipeline = VK_NULL_HANDLE;
    }

    if (WorkOffsetsPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(Device, WorkOffsetsPipeline, nullptr);
        WorkOffsetsPipeline = VK_NULL_HANDLE;
    }

    if (SortPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(Device, SortPipeline, nullptr);
        SortPipeline = VK_NULL_HANDLE;
    }

    if (DepthBlendPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(Device, DepthBlendPipeline, nullptr);
        DepthBlendPipeline = VK_NULL_HANDLE;
    }

    for (VkPipeline& rasterPipeline : RasterPipelines)
    {
        if (rasterPipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(Device, rasterPipeline, nullptr);
            rasterPipeline = VK_NULL_HANDLE;
        }
    }

    for (VkPipeline& finalPipeline : FinalPipelines)
    {
        if (finalPipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(Device, finalPipeline, nullptr);
            finalPipeline = VK_NULL_HANDLE;
        }
    }

    if (CaptureLineExportPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(Device, CaptureLineExportPipeline, nullptr);
        CaptureLineExportPipeline = VK_NULL_HANDLE;
    }

    savePipelineCache();
    if (ComputePipelineCache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(Device, ComputePipelineCache, nullptr);
        ComputePipelineCache = VK_NULL_HANDLE;
    }
    ComputePipelineCacheFile.clear();

    if (PipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(Device, PipelineLayout, nullptr);
        PipelineLayout = VK_NULL_HANDLE;
    }

    if (DescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(Device, DescriptorPool, nullptr);
        DescriptorPool = VK_NULL_HANDLE;
    }

    DescriptorSet = VK_NULL_HANDLE;
    invalidateAllDescriptorSetCaches();
    for (RenderContext& renderContext : RenderContexts)
        renderContext.DescriptorSet = VK_NULL_HANDLE;

    if (DescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(Device, DescriptorSetLayout, nullptr);
        DescriptorSetLayout = VK_NULL_HANDLE;
    }

    for (RenderContext& renderContext : RenderContexts)
    {
        destroyCpuSpanSetupBuffer(renderContext);
        destroyCpuBinBuffers(renderContext);
        destroyCpuWorkOffsetBuffer(renderContext);
        destroyTriangleBuffer(&renderContext);
        destroyToonBuffer(&renderContext);
        destroyCaptureLineBuffer(&renderContext);
        if (renderContext.TimestampQueryPool != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(Device, renderContext.TimestampQueryPool, nullptr);
            renderContext.TimestampQueryPool = VK_NULL_HANDLE;
        }
        if (renderContext.FrameFence != VK_NULL_HANDLE)
        {
            vkDestroyFence(Device, renderContext.FrameFence, nullptr);
            renderContext.FrameFence = VK_NULL_HANDLE;
        }
        if (renderContext.CommandPool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(Device, renderContext.CommandPool, nullptr);
            renderContext.CommandPool = VK_NULL_HANDLE;
        }
        renderContext.CommandBuffer = VK_NULL_HANDLE;
    }

    if (FrameFence != VK_NULL_HANDLE)
    {
        vkDestroyFence(Device, FrameFence, nullptr);
        FrameFence = VK_NULL_HANDLE;
    }
    if (TimestampQueryPool != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(Device, TimestampQueryPool, nullptr);
        TimestampQueryPool = VK_NULL_HANDLE;
    }

    if (CommandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(Device, CommandPool, nullptr);
        CommandPool = VK_NULL_HANDLE;
    }

    CommandBuffer = VK_NULL_HANDLE;
    NextRenderContextIndex = 0;

    if (ContextAcquired)
    {
        VulkanContext::Get().Release();
        ContextAcquired = false;
    }

    Instance = VK_NULL_HANDLE;
    PhysicalDevice = VK_NULL_HANDLE;
    Device = VK_NULL_HANDLE;
    Queue = VK_NULL_HANDLE;
    QueueFamilyIndex = 0;
    ResetQueryPool = nullptr;
    TimestampPeriodNs = 0.0f;
    TimestampQueriesSupported = false;
    Initialized = false;
    ColorImageInitialized = false;
    ActiveTextureDescriptorCount = 0;
    ActiveTextureDescriptors.fill(VkDescriptorImageInfo{});
}

bool VulkanRenderer3D::createCommandObjects()
{
    if (!createCommandObjects(CommandPool, CommandBuffer))
        return false;

    for (RenderContext& renderContext : RenderContexts)
    {
        if (!createCommandObjects(renderContext.CommandPool, renderContext.CommandBuffer))
            return false;
    }

    return true;
}

bool VulkanRenderer3D::createCommandObjects(VkCommandPool& commandPool, VkCommandBuffer& commandBuffer)
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = QueueFamilyIndex;

    if (vkCreateCommandPool(Device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create command pool");
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(Device, &cmdAllocInfo, &commandBuffer) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate command buffer");
        return false;
    }

    return true;
}

bool VulkanRenderer3D::createSyncObjects()
{
    if (!createFence(FrameFence))
        return false;
    if (!createTimestampQueryPool(TimestampQueryPool))
        return false;

    for (RenderContext& renderContext : RenderContexts)
    {
        if (!createFence(renderContext.FrameFence))
            return false;
        if (!createTimestampQueryPool(renderContext.TimestampQueryPool))
            return false;
    }

    return true;
}

bool VulkanRenderer3D::createFence(VkFence& fence)
{
    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(Device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create frame fence");
        return false;
    }

    return true;
}

bool VulkanRenderer3D::createTimestampQueryPool(VkQueryPool& queryPool)
{
    if (!TimestampQueriesSupported)
        return true;

    VkQueryPoolCreateInfo queryPoolCreateInfo{};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = TimestampQueryCount;

    if (vkCreateQueryPool(Device, &queryPoolCreateInfo, nullptr, &queryPool) != VK_SUCCESS)
    {
        Log(LogLevel::Warn, "VulkanRenderer3D: failed to create timestamp query pool");
        queryPool = VK_NULL_HANDLE;
    }

    return true;
}

bool VulkanRenderer3D::waitForRenderContext(RenderContext& context)
{
    if (Device == VK_NULL_HANDLE || context.FrameFence == VK_NULL_HANDLE)
        return false;

    const VkResult fenceStatus = vkGetFenceStatus(Device, context.FrameFence);
    if (fenceStatus == VK_NOT_READY)
        ContextMissCount++;

    const u64 waitStartNs = PerfNowNs();
    const VkResult waitResult = vkWaitForFences(Device, 1, &context.FrameFence, VK_TRUE, UINT64_MAX);
    if (waitResult != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: render context fence wait failed (%d)", static_cast<int>(waitResult));
        return false;
    }

    const u64 waitDurationNs = PerfNowNs() - waitStartNs;
    FenceWaitCpuWindow.Add(waitDurationNs);
    if (waitDurationNs >= 1000000ull)
        LateFrameCount++;
    consumeGpuTiming(&context);

    return true;
}

bool VulkanRenderer3D::tryAcquireRenderContext(RenderContext& context, bool countMisses)
{
    if (Device == VK_NULL_HANDLE || context.FrameFence == VK_NULL_HANDLE)
        return false;

    const VkResult fenceStatus = vkGetFenceStatus(Device, context.FrameFence);
    if (fenceStatus == VK_SUCCESS)
    {
        FenceWaitCpuWindow.Add(0);
        consumeGpuTiming(&context);
        return true;
    }

    if (fenceStatus == VK_NOT_READY)
    {
        if (countMisses)
        {
            ContextMissCount++;
            DroppedFrameCount++;
        }
        return false;
    }

    Log(LogLevel::Error, "VulkanRenderer3D: render context fence status failed (%d)", static_cast<int>(fenceStatus));
    return false;
}

VulkanRenderer3D::RenderContext* VulkanRenderer3D::tryAcquireReadyRenderContext() noexcept
{
    if (!Threaded || Device == VK_NULL_HANDLE)
        return nullptr;

    for (size_t i = 0; i < AsyncRenderContextCount; i++)
    {
        const size_t contextIndex = (NextRenderContextIndex + i) % AsyncRenderContextCount;
        RenderContext& context = RenderContexts[contextIndex];
        if (!tryAcquireRenderContext(context, false))
            continue;

        NextRenderContextIndex = (contextIndex + 1) % AsyncRenderContextCount;
        return &context;
    }

    return nullptr;
}

bool VulkanRenderer3D::waitForAllRenderContexts()
{
    for (RenderContext& renderContext : RenderContexts)
    {
        if (!waitForRenderContext(renderContext))
            return false;
    }

    return true;
}

bool VulkanRenderer3D::waitForReadbackSource()
{
    if (!Threaded)
        return true;

    // Render and presentation share one queue. Waiting for the newest submitted
    // render context fence guarantees all older submissions are finished too.
    if (LastSubmittedRenderContext != nullptr)
        return waitForRenderContext(*LastSubmittedRenderContext);

    return waitForAllRenderContexts();
}

bool VulkanRenderer3D::finalizeCaptureReadback(bool blocking)
{
    if (!CaptureReadbackPending)
        return HasCpuFrame;

    bool waitOk = false;
    if (PendingCaptureReadbackContext != nullptr)
    {
        if (blocking)
        {
            waitOk = waitForRenderContext(*PendingCaptureReadbackContext);
        }
        else if (Device != VK_NULL_HANDLE && PendingCaptureReadbackContext->FrameFence != VK_NULL_HANDLE)
        {
            const VkResult fenceStatus = vkGetFenceStatus(Device, PendingCaptureReadbackContext->FrameFence);
            if (fenceStatus == VK_SUCCESS)
            {
                FenceWaitCpuWindow.Add(0);
                consumeGpuTiming(PendingCaptureReadbackContext);
                waitOk = true;
            }
            else if (fenceStatus == VK_NOT_READY)
            {
                return false;
            }
            else
            {
                CaptureReadbackPending = false;
                PendingCaptureReadbackContext = nullptr;
                return false;
            }
        }
        else
        {
            CaptureReadbackPending = false;
            PendingCaptureReadbackContext = nullptr;
            return false;
        }
    }
    else
    {
        if (blocking)
            waitOk = waitForReadbackSource();
        else
            return false;
    }

    if (!waitOk || ReadbackMapped == nullptr || RawReadbackWidth == 0 || RawReadbackHeight == 0)
    {
        CaptureReadbackPending = false;
        PendingCaptureReadbackContext = nullptr;
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(RawReadbackWidth) * static_cast<size_t>(RawReadbackHeight);
    if (RawReadbackRgba.size() != pixelCount)
        RawReadbackRgba.resize(pixelCount);
    std::memcpy(RawReadbackRgba.data(), ReadbackMapped, pixelCount * sizeof(u32));

    CaptureReadbackPending = false;
    PendingCaptureReadbackContext = nullptr;
    HasCpuFrame = true;
    ColorImageInitialized = true;
    return true;
}

void VulkanRenderer3D::resetCaptureLineState()
{
    CaptureLinePending = false;
    CaptureLineReady = false;
    PendingCaptureLineContext = nullptr;
    ReadyCaptureLineData = nullptr;
}

bool VulkanRenderer3D::finalizeCaptureLineFrame(bool blocking)
{
    if (!CaptureLinePending)
        return CaptureLineReady;

    bool waitOk = false;
    if (PendingCaptureLineContext != nullptr)
    {
        if (blocking)
        {
            waitOk = waitForRenderContext(*PendingCaptureLineContext);
        }
        else if (Device != VK_NULL_HANDLE && PendingCaptureLineContext->FrameFence != VK_NULL_HANDLE)
        {
            const VkResult fenceStatus = vkGetFenceStatus(Device, PendingCaptureLineContext->FrameFence);
            if (fenceStatus == VK_SUCCESS)
            {
                FenceWaitCpuWindow.Add(0);
                consumeGpuTiming(PendingCaptureLineContext);
                waitOk = true;
            }
            else if (fenceStatus == VK_NOT_READY)
            {
                return false;
            }
            else
            {
                resetCaptureLineState();
                return false;
            }
        }
        else
        {
            resetCaptureLineState();
            return false;
        }
    }
    else
    {
        if (blocking)
            waitOk = waitForReadbackSource();
        else
            return false;
    }
    if (!waitOk)
    {
        resetCaptureLineState();
        return false;
    }

    if (PendingCaptureLineContext != nullptr)
        ReadyCaptureLineData = reinterpret_cast<const u32*>(PendingCaptureLineContext->CaptureLineMapped);
    else
        ReadyCaptureLineData = reinterpret_cast<const u32*>(CaptureLineMapped);

    CaptureLinePending = false;
    PendingCaptureLineContext = nullptr;
    CaptureLineReady = ReadyCaptureLineData != nullptr;
    return CaptureLineReady;
}

bool VulkanRenderer3D::waitForDeviceIdle(const char* reason)
{
    if (Device == VK_NULL_HANDLE)
        return false;

    const VkResult waitResult = vkDeviceWaitIdle(Device);
    if (waitResult != VK_SUCCESS)
    {
        Log(
            LogLevel::Error,
            "VulkanRenderer3D: vkDeviceWaitIdle failed while waiting for %s (%d)",
            reason != nullptr ? reason : "device idle",
            static_cast<int>(waitResult)
        );
        return false;
    }

    return true;
}

VulkanRenderer3D::RenderContext& VulkanRenderer3D::acquireNextRenderContext() noexcept
{
    RenderContext& renderContext = RenderContexts[NextRenderContextIndex];
    NextRenderContextIndex = (NextRenderContextIndex + 1) % AsyncRenderContextCount;
    return renderContext;
}

void VulkanRenderer3D::requestPostFastForwardDrain()
{
    if (!Threaded)
        return;

    PostFastForwardDrainFrames = static_cast<u32>(AsyncRenderContextCount * 3);
}

void VulkanRenderer3D::consumeGpuTiming(RenderContext* context)
{
    VkQueryPool queryPool = context != nullptr ? context->TimestampQueryPool : TimestampQueryPool;
    bool& timestampPending = context != nullptr ? context->TimestampPending : TimestampPending;

    if (!timestampPending || queryPool == VK_NULL_HANDLE || TimestampPeriodNs <= 0.0f)
        return;

    u64 timestamps[TimestampQueryCount]{};
    const VkResult queryResult = vkGetQueryPoolResults(
        Device,
        queryPool,
        0,
        TimestampQueryCount,
        sizeof(timestamps),
        timestamps,
        sizeof(u64),
        VK_QUERY_RESULT_64_BIT
    );
    if (queryResult == VK_SUCCESS && timestamps[TimestampQueryCount - 1] >= timestamps[0])
    {
        auto toGpuNs = [&](u32 startIndex, u32 endIndex) -> u64 {
            if (endIndex >= TimestampQueryCount || timestamps[endIndex] < timestamps[startIndex])
                return 0;
            return static_cast<u64>(static_cast<double>(timestamps[endIndex] - timestamps[startIndex]) * static_cast<double>(TimestampPeriodNs));
        };

        const u64 gpuTimeNs = toGpuNs(0, TimestampQueryCount - 1);
        GpuWindow.Add(gpuTimeNs);
        InterpGpuWindow.Add(toGpuNs(0, 1));
        BinGpuWindow.Add(toGpuNs(1, 2));
        WorkOffsetsGpuWindow.Add(toGpuNs(2, 3));
        SortGpuWindow.Add(toGpuNs(3, 4));
        RasterGpuWindow.Add(toGpuNs(4, 5));
        DepthBlendGpuWindow.Add(toGpuNs(5, 6));
        FinalGpuWindow.Add(toGpuNs(6, 7));
        CaptureLineExportGpuWindow.Add(toGpuNs(7, 8));
    }

    timestampPending = false;
}

void VulkanRenderer3D::logPerformanceIfNeeded()
{
    if (!MelonDSAndroid::areRendererDebugToolsEnabled())
        return;

    if (!RenderCpuWindow.Ready())
        return;

    const PerfSampleWindow<120>::Summary renderSummary = RenderCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary waitSummary = FenceWaitCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary gpuSummary = GpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary triangleSummary = TriangleCountWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary passSummary = PassCountWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary interpCpuSummary = InterpCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary binCpuSummary = BinCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary workOffsetsCpuSummary = WorkOffsetsCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary sortCpuSummary = SortCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary rasterCpuSummary = RasterCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary depthBlendCpuSummary = DepthBlendCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary finalCpuSummary = FinalCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary captureLineExportCpuSummary = CaptureLineExportCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary cpuActiveTileSummary = CpuActiveTileCountWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary cpuTileCountSummary = CpuTileCountWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary cpuActiveGroupSummary = CpuActiveGroupCountWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary cpuActiveDispatchSummary = CpuActiveDispatchWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary interpGpuSummary = InterpGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary binGpuSummary = BinGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary workOffsetsGpuSummary = WorkOffsetsGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary sortGpuSummary = SortGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary rasterGpuSummary = RasterGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary depthBlendGpuSummary = DepthBlendGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary finalGpuSummary = FinalGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary captureLineExportGpuSummary = CaptureLineExportGpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary earlySubmitCpuSummary = EarlySubmitCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary earlySubmitWaitSummary = EarlySubmitContextWaitCpuWindow.SummarizeAndReset();
    const double cpuTileCoveragePercent = cpuTileCountSummary.MeanNs > 0
        ? (static_cast<double>(cpuActiveTileSummary.MeanNs) * 100.0 / static_cast<double>(cpuTileCountSummary.MeanNs))
        : 0.0;

    Log(
        LogLevel::Warn,
        "VulkanPerf[GPU3D]: path=%s scale=%d render cpu avg=%.3fms p95=%.3fms max=%.3fms wait avg=%.3fms p95=%.3fms max=%.3fms gpu avg=%.3fms p95=%.3fms max=%.3fms triangles avg=%llu passes avg=%llu p95=%llu cpuTiles avg=%llu/%llu (%.1f%%) cpuGroups avg=%llu activeDispatch=%llu%% contextMisses=%llu late=%llu dropped=%llu readbackColor=%llu readbackResult=%llu capturePrepare=%llu captureEnabled=%llu captureSrc3d=%llu capMode=%llu/%llu/%llu/%llu capSize=%llu/%llu/%llu/%llu capExport=%llu capExportCpu avg=%.3fms p95=%.3fms capExportGpu avg=%.3fms p95=%.3fms rasterSpec tex=%llu alpha=%llu shade=%llu all=%llu earlySubmit hit=%llu/%llu miss=%llu skip215=%llu cpu avg=%.3fms p95=%.3fms wait avg=%.3fms p95=%.3fms",
        CpuDirectTilesPathCount > 0 && DirectTilesPathCount == 0 && LegacyWorklistPathCount == 0
            ? "cpu_direct_tiles"
            : (LegacyWorklistPathCount > 0 && DirectTilesPathCount == 0 ? "legacy" : "direct_tiles"),
        std::max(1, ScaleFactor),
        PerfNsToMs(renderSummary.MeanNs),
        PerfNsToMs(renderSummary.P95Ns),
        PerfNsToMs(renderSummary.MaxNs),
        PerfNsToMs(waitSummary.MeanNs),
        PerfNsToMs(waitSummary.P95Ns),
        PerfNsToMs(waitSummary.MaxNs),
        PerfNsToMs(gpuSummary.MeanNs),
        PerfNsToMs(gpuSummary.P95Ns),
        PerfNsToMs(gpuSummary.MaxNs),
        static_cast<unsigned long long>(triangleSummary.MeanNs),
        static_cast<unsigned long long>(passSummary.MeanNs),
        static_cast<unsigned long long>(passSummary.P95Ns),
        static_cast<unsigned long long>(cpuActiveTileSummary.MeanNs),
        static_cast<unsigned long long>(cpuTileCountSummary.MeanNs),
        cpuTileCoveragePercent,
        static_cast<unsigned long long>(cpuActiveGroupSummary.MeanNs),
        static_cast<unsigned long long>(cpuActiveDispatchSummary.MeanNs),
        static_cast<unsigned long long>(ContextMissCount),
        static_cast<unsigned long long>(LateFrameCount),
        static_cast<unsigned long long>(DroppedFrameCount),
        static_cast<unsigned long long>(ReadbackColorRequestCount),
        static_cast<unsigned long long>(ReadbackResultRequestCount),
        static_cast<unsigned long long>(CapturePrepareRequestCount),
        static_cast<unsigned long long>(CaptureEnabledCount),
        static_cast<unsigned long long>(CaptureSource3dCount),
        static_cast<unsigned long long>(CaptureModeCounts[0]),
        static_cast<unsigned long long>(CaptureModeCounts[1]),
        static_cast<unsigned long long>(CaptureModeCounts[2]),
        static_cast<unsigned long long>(CaptureModeCounts[3]),
        static_cast<unsigned long long>(CaptureSizeModeCounts[0]),
        static_cast<unsigned long long>(CaptureSizeModeCounts[1]),
        static_cast<unsigned long long>(CaptureSizeModeCounts[2]),
        static_cast<unsigned long long>(CaptureSizeModeCounts[3]),
        static_cast<unsigned long long>(CaptureLineExportCount),
        PerfNsToMs(captureLineExportCpuSummary.MeanNs),
        PerfNsToMs(captureLineExportCpuSummary.P95Ns),
        PerfNsToMs(captureLineExportGpuSummary.MeanNs),
        PerfNsToMs(captureLineExportGpuSummary.P95Ns),
        static_cast<unsigned long long>(RasterSpecializedTextureModeCount),
        static_cast<unsigned long long>(RasterSpecializedTranslucencyModeCount),
        static_cast<unsigned long long>(RasterSpecializedShadeModeCount),
        static_cast<unsigned long long>(RasterSpecializedAllModesCount),
        static_cast<unsigned long long>(EarlySubmitHitCount),
        static_cast<unsigned long long>(EarlySubmitAttemptCount),
        static_cast<unsigned long long>(EarlySubmitMissCount),
        static_cast<unsigned long long>(EarlySubmitSkipVCount215Count),
        PerfNsToMs(earlySubmitCpuSummary.MeanNs),
        PerfNsToMs(earlySubmitCpuSummary.P95Ns),
        PerfNsToMs(earlySubmitWaitSummary.MeanNs),
        PerfNsToMs(earlySubmitWaitSummary.P95Ns)
    );
    Log(
        LogLevel::Warn,
        "VulkanPerf[GPU3DPasses]: interp cpu avg=%.3fms gpu avg=%.3fms bin cpu avg=%.3fms gpu avg=%.3fms work cpu avg=%.3fms gpu avg=%.3fms sort cpu avg=%.3fms gpu avg=%.3fms raster cpu avg=%.3fms gpu avg=%.3fms resolve cpu avg=%.3fms gpu avg=%.3fms final cpu avg=%.3fms gpu avg=%.3fms captureExport cpu avg=%.3fms gpu avg=%.3fms",
        PerfNsToMs(interpCpuSummary.MeanNs),
        PerfNsToMs(interpGpuSummary.MeanNs),
        PerfNsToMs(binCpuSummary.MeanNs),
        PerfNsToMs(binGpuSummary.MeanNs),
        PerfNsToMs(workOffsetsCpuSummary.MeanNs),
        PerfNsToMs(workOffsetsGpuSummary.MeanNs),
        PerfNsToMs(sortCpuSummary.MeanNs),
        PerfNsToMs(sortGpuSummary.MeanNs),
        PerfNsToMs(rasterCpuSummary.MeanNs),
        PerfNsToMs(rasterGpuSummary.MeanNs),
        PerfNsToMs(depthBlendCpuSummary.MeanNs),
        PerfNsToMs(depthBlendGpuSummary.MeanNs),
        PerfNsToMs(finalCpuSummary.MeanNs),
        PerfNsToMs(finalGpuSummary.MeanNs),
        PerfNsToMs(captureLineExportCpuSummary.MeanNs),
        PerfNsToMs(captureLineExportGpuSummary.MeanNs)
    );

    ContextMissCount = 0;
    LateFrameCount = 0;
    DroppedFrameCount = 0;
    CpuDirectTilesPathCount = 0;
    DirectTilesPathCount = 0;
    LegacyWorklistPathCount = 0;
    ReadbackColorRequestCount = 0;
    ReadbackResultRequestCount = 0;
    CapturePrepareRequestCount = 0;
    CaptureEnabledCount = 0;
    CaptureSource3dCount = 0;
    CaptureModeCounts.fill(0);
    CaptureSizeModeCounts.fill(0);
    CaptureLineExportCount = 0;
    RasterSpecializedShadeModeCount = 0;
    RasterSpecializedTextureModeCount = 0;
    RasterSpecializedTranslucencyModeCount = 0;
    RasterSpecializedAllModesCount = 0;
    EarlySubmitAttemptCount = 0;
    EarlySubmitHitCount = 0;
    EarlySubmitMissCount = 0;
    EarlySubmitSkipVCount215Count = 0;
}

bool VulkanRenderer3D::useCpuTileBinning() const noexcept
{
    return CpuTileBinningEnabled;
}

bool VulkanRenderer3D::prepareCpuTileBins(RenderContext& context, const RasterPushConstants& pushConstants)
{
    if (context.SpanSetupMapped == nullptr
        || context.BinMaskMapped == nullptr
        || context.GroupListMapped == nullptr
        || context.WorkOffsetMapped == nullptr
        || pushConstants.width == 0
        || pushConstants.height == 0)
    {
        return false;
    }

    constexpr u32 kTileSize = 8u;
    const u32 tilesPerLine = (pushConstants.width + (kTileSize - 1u)) / kTileSize;
    const u32 tileLineCount = (pushConstants.height + (kTileSize - 1u)) / kTileSize;
    const u32 tileCount = std::max<u32>(1u, tilesPerLine * tileLineCount);
    const u32 triangleCount = std::min<u32>(pushConstants.triangleCount, static_cast<u32>(Triangles.size()));
    const u32 groupCount = std::max<u32>(1u, (triangleCount + 31u) / 32u);
    const size_t requiredSpanSetupCount = std::max<size_t>(1u, static_cast<size_t>(triangleCount));
    const size_t requiredBinMaskWords = static_cast<size_t>(tileCount) * static_cast<size_t>(groupCount);
    const size_t requiredGroupListWords = static_cast<size_t>(tileCount) * static_cast<size_t>(groupCount + 1u);
    const u32 maxGroupListEntries = tileCount * groupCount;
    const u32 requiredWorkOffsetWords = kWorkTileOffsetsBase + (tileCount + 1u) + tileCount + maxGroupListEntries;

    if (context.SpanSetupBufferSize < static_cast<VkDeviceSize>(requiredSpanSetupCount * sizeof(SpanSetupGpu))
        || context.BinMaskBufferSize < static_cast<VkDeviceSize>(requiredBinMaskWords * sizeof(u32))
        || context.GroupListBufferSize < static_cast<VkDeviceSize>(requiredGroupListWords * sizeof(u32))
        || context.WorkOffsetBufferSize < static_cast<VkDeviceSize>(requiredWorkOffsetWords * sizeof(u32)))
    {
        return false;
    }

    auto* spanSetups = reinterpret_cast<SpanSetupGpu*>(context.SpanSetupMapped);
    auto* binMaskValues = reinterpret_cast<u32*>(context.BinMaskMapped);
    auto* groupListValues = reinterpret_cast<u32*>(context.GroupListMapped);
    auto* workOffsetValues = reinterpret_cast<u32*>(context.WorkOffsetMapped);
    std::memset(binMaskValues, 0, requiredBinMaskWords * sizeof(u32));
    std::memset(groupListValues, 0, static_cast<size_t>(tileCount) * sizeof(u32));

    const auto variantMatchesPass = [](u32 triangleVariantKey, u32 passVariantKey) noexcept -> bool {
        constexpr u32 kVariantWildcard = 0xFFFFFFFFu;
        constexpr u32 kVariantPipelineMask = (1u << 8u) - 1u;
        if (passVariantKey == kVariantWildcard)
            return true;
        return (triangleVariantKey & kVariantPipelineMask) == (passVariantKey & kVariantPipelineMask);
    };

    for (u32 triangleIdx = 0; triangleIdx < triangleCount; triangleIdx++)
    {
        const TriangleGpu& tri = Triangles[triangleIdx];
        SpanSetupGpu& span = spanSetups[triangleIdx];
        const float p0x = tri.x0;
        const float p0y = tri.y0;
        const float p1x = tri.x1;
        const float p1y = tri.y1;
        const float p2x = tri.x2;
        const float p2y = tri.y2;

        float minX = std::min(std::min(p0x, p1x), p2x);
        float minY = std::min(std::min(p0y, p1y), p2y);
        float maxX = std::max(std::max(p0x, p1x), p2x);
        float maxY = std::max(std::max(p0y, p1y), p2y);
        minX = std::clamp(minX, -1.0f, static_cast<float>(pushConstants.width) + 1.0f);
        maxX = std::clamp(maxX, -1.0f, static_cast<float>(pushConstants.width) + 1.0f);
        minY = std::clamp(minY, -1.0f, static_cast<float>(pushConstants.height) + 1.0f);
        maxY = std::clamp(maxY, -1.0f, static_cast<float>(pushConstants.height) + 1.0f);

        const u32 boundsYMin = tri.yBounds & 0xFFFFu;
        const u32 boundsYMax = (tri.yBounds >> 16u) & 0xFFFFu;
        u32 geomYMin = static_cast<u32>(std::max(0, static_cast<int>(std::floor(minY))));
        u32 geomYMax = static_cast<u32>(std::max(0, static_cast<int>(std::ceil(maxY))));
        geomYMin = std::min(geomYMin, pushConstants.height);
        geomYMax = std::min(geomYMax, pushConstants.height);

        const u32 yMin = std::max(boundsYMin, geomYMin);
        const u32 yMax = std::min(boundsYMax, geomYMax);
        span.minX = minX;
        span.minY = minY;
        span.maxX = maxX;
        span.maxY = maxY;
        span.yMin = yMin;
        span.yMax = yMax;
        span.variantKey = tri.variantKey & ((1u << 8u) - 1u);
        span.valid = variantMatchesPass(tri.variantKey, pushConstants.variantKey) ? 1u : 0u;
        if (maxX < minX || maxY < minY || yMax <= yMin)
            span.valid = 0u;
        if (span.valid == 0u)
            continue;

        const float triangleArea = edgeFunction2D(p0x, p0y, p1x, p1y, p2x, p2y);
        if (std::abs(triangleArea) < kTriangleAreaEpsilon)
        {
            span.valid = 0u;
            continue;
        }
        const bool positiveArea = triangleArea > 0.0f;
        const float edge0dx = p2x - p1x;
        const float edge0dy = p2y - p1y;
        const float edge1dx = p0x - p2x;
        const float edge1dy = p0y - p2y;
        const float edge2dx = p1x - p0x;
        const float edge2dy = p1y - p0y;
        const float edge0LengthSquared = edge0dx * edge0dx + edge0dy * edge0dy;
        const float edge1LengthSquared = edge1dx * edge1dx + edge1dy * edge1dy;
        const float edge2LengthSquared = edge2dx * edge2dx + edge2dy * edge2dy;
        span.edgeInv0 = edge0LengthSquared > 1e-12f ? 1.0f / std::sqrt(edge0LengthSquared) : 0.0f;
        span.edgeInv1 = edge1LengthSquared > 1e-12f ? 1.0f / std::sqrt(edge1LengthSquared) : 0.0f;
        span.edgeInv2 = edge2LengthSquared > 1e-12f ? 1.0f / std::sqrt(edge2LengthSquared) : 0.0f;

        const int minTileX = std::clamp(static_cast<int>(std::floor(minX / static_cast<float>(kTileSize))), 0, static_cast<int>(tilesPerLine) - 1);
        const int maxTileX = std::clamp(static_cast<int>(std::floor(maxX / static_cast<float>(kTileSize))), 0, static_cast<int>(tilesPerLine) - 1);
        const int minTileY = std::clamp(static_cast<int>(yMin / kTileSize), 0, static_cast<int>(tileLineCount) - 1);
        const int maxTileY = std::clamp(static_cast<int>((yMax - 1u) / kTileSize), 0, static_cast<int>(tileLineCount) - 1);

        const EdgeEquation2D edge0 = makeEdgeEquation2D(p0x, p0y, p1x, p1y);
        const EdgeEquation2D edge1 = makeEdgeEquation2D(p1x, p1y, p2x, p2y);
        const EdgeEquation2D edge2 = makeEdgeEquation2D(p2x, p2y, p0x, p0y);
        constexpr float kTileSizeF = static_cast<float>(kTileSize);
        const float frameWidth = static_cast<float>(pushConstants.width);
        const float frameHeight = static_cast<float>(pushConstants.height);
        const float startTileMinX = static_cast<float>(minTileX * static_cast<int>(kTileSize));
        const float edge0StepX = edge0.a * kTileSizeF;
        const float edge1StepX = edge1.a * kTileSizeF;
        const float edge2StepX = edge2.a * kTileSizeF;
        const u32 groupIdx = triangleIdx / 32u;
        const u32 groupBit = 1u << (triangleIdx & 31u);
        for (int tileY = minTileY; tileY <= maxTileY; tileY++)
        {
            const size_t rowTileBase = static_cast<size_t>(tileY) * static_cast<size_t>(tilesPerLine);
            const float tileMinY = static_cast<float>(tileY * static_cast<int>(kTileSize));
            const float tileMaxY = std::min(tileMinY + kTileSizeF, frameHeight);
            const float tileHeight = tileMaxY - tileMinY;
            float tileMinX = startTileMinX;
            float edge0MinMin = evaluateEdge2D(edge0, tileMinX, tileMinY);
            float edge0MinMax = edge0MinMin + edge0.b * tileHeight;
            float edge1MinMin = evaluateEdge2D(edge1, tileMinX, tileMinY);
            float edge1MinMax = edge1MinMin + edge1.b * tileHeight;
            float edge2MinMin = evaluateEdge2D(edge2, tileMinX, tileMinY);
            float edge2MinMax = edge2MinMin + edge2.b * tileHeight;
            for (int tileX = minTileX; tileX <= maxTileX; tileX++)
            {
                const float tileMaxX = std::min(tileMinX + kTileSizeF, frameWidth);
                const float tileWidth = tileMaxX - tileMinX;
                const float edge0MaxMin = edge0MinMin + edge0.a * tileWidth;
                const float edge0MaxMax = edge0MinMax + edge0.a * tileWidth;
                const float edge1MaxMin = edge1MinMin + edge1.a * tileWidth;
                const float edge1MaxMax = edge1MinMax + edge1.a * tileWidth;
                const float edge2MaxMin = edge2MinMin + edge2.a * tileWidth;
                const float edge2MaxMax = edge2MinMax + edge2.a * tileWidth;
                if (edgeSeparatesTile(edge0MinMin, edge0MaxMin, edge0MinMax, edge0MaxMax, positiveArea)
                    || edgeSeparatesTile(edge1MinMin, edge1MaxMin, edge1MinMax, edge1MaxMax, positiveArea)
                    || edgeSeparatesTile(edge2MinMin, edge2MaxMin, edge2MinMax, edge2MaxMax, positiveArea))
                {
                    tileMinX += kTileSizeF;
                    edge0MinMin += edge0StepX;
                    edge0MinMax += edge0StepX;
                    edge1MinMin += edge1StepX;
                    edge1MinMax += edge1StepX;
                    edge2MinMin += edge2StepX;
                    edge2MinMax += edge2StepX;
                    continue;
                }

                const size_t linearTile = rowTileBase + static_cast<size_t>(tileX);
                const size_t maskIndex = linearTile * static_cast<size_t>(groupCount) + static_cast<size_t>(groupIdx);
                const u32 previousMask = binMaskValues[maskIndex];
                binMaskValues[maskIndex] = previousMask | groupBit;
                if (previousMask == 0u)
                {
                    const u32 slot = groupListValues[linearTile]++;
                    if (slot < groupCount)
                    {
                        const size_t listIndex =
                            static_cast<size_t>(tileCount)
                            + linearTile * static_cast<size_t>(groupCount)
                            + static_cast<size_t>(slot);
                        groupListValues[listIndex] = groupIdx;
                    }
                }

                tileMinX += kTileSizeF;
                edge0MinMin += edge0StepX;
                edge0MinMax += edge0StepX;
                edge1MinMin += edge1StepX;
                edge1MinMax += edge1StepX;
                edge2MinMin += edge2StepX;
                edge2MinMax += edge2StepX;
            }
        }
    }

    const size_t tileGroupListBase = static_cast<size_t>(tileCount);
    const size_t compactGroupListBase = static_cast<size_t>(kWorkTileOffsetsBase)
        + static_cast<size_t>(tileCount + 1u)
        + static_cast<size_t>(tileCount);
    u32 runningGroupOffset = 0u;
    u32 activeTileCount = 0u;
    for (u32 linearTile = 0; linearTile < tileCount; linearTile++)
    {
        const u32 tileGroupCount = std::min(groupListValues[linearTile], groupCount);
        if (tileGroupCount == 0u)
            continue;

        runningGroupOffset += tileGroupCount;
        activeTileCount++;
    }

    const bool hasSparseCoverage = activeTileCount == 0u
        || static_cast<u64>(activeTileCount) * 100ull
            < static_cast<u64>(tileCount) * static_cast<u64>(kCpuActiveTileDispatchMaxCoveragePercent);
    if (hasSparseCoverage)
    {
        u32 compactGroupOffset = 0u;
        u32 compactActiveTileIndex = 0u;
        for (u32 linearTile = 0; linearTile < tileCount; linearTile++)
        {
            workOffsetValues[kWorkTileOffsetsBase + linearTile] = compactGroupOffset;

            const u32 tileGroupCount = std::min(groupListValues[linearTile], groupCount);
            if (tileGroupCount == 0u)
                continue;

            workOffsetValues[kWorkTileOffsetsBase + tileCount + 1u + compactActiveTileIndex] = linearTile;
            const size_t sourceOffset = tileGroupListBase + static_cast<size_t>(linearTile) * static_cast<size_t>(groupCount);
            const size_t destinationOffset = compactGroupListBase + static_cast<size_t>(compactGroupOffset);
            std::memcpy(
                &workOffsetValues[destinationOffset],
                &groupListValues[sourceOffset],
                static_cast<size_t>(tileGroupCount) * sizeof(u32));
            compactGroupOffset += tileGroupCount;
            compactActiveTileIndex++;
        }
        workOffsetValues[kWorkTileOffsetsBase + tileCount] = compactGroupOffset;
    }

    workOffsetValues[kWorkSortDispatchX] = std::max<u32>(1u, (activeTileCount + 63u) / 64u);
    workOffsetValues[kWorkSortDispatchY] = 1u;
    workOffsetValues[kWorkSortDispatchZ] = 1u;
    workOffsetValues[kWorkRasterDispatchX] = std::max<u32>(1u, activeTileCount);
    workOffsetValues[kWorkRasterDispatchY] = 1u;
    workOffsetValues[kWorkRasterDispatchZ] = 1u;
    workOffsetValues[kWorkActiveTileCount] = activeTileCount;
    workOffsetValues[kWorkActiveGroupCount] = runningGroupOffset;

    return true;
}

bool VulkanRenderer3D::createDescriptorObjects()
{
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(PhysicalDevice, &deviceProperties);
    if (deviceProperties.limits.maxPerStageDescriptorSampledImages < MaxTextureDescriptors
        || deviceProperties.limits.maxDescriptorSetSampledImages < MaxTextureDescriptors)
    {
        Log(
            LogLevel::Error,
            "VulkanRenderer3D: descriptor limits too low (per-stage=%u, set=%u, required=%u)",
            deviceProperties.limits.maxPerStageDescriptorSampledImages,
            deviceProperties.limits.maxDescriptorSetSampledImages,
            MaxTextureDescriptors
        );
        return false;
    }

    VkDescriptorSetLayoutBinding imageBinding{};
    imageBinding.binding = 0;
    imageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageBinding.descriptorCount = 1;
    imageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding triangleBinding{};
    triangleBinding.binding = 1;
    triangleBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    triangleBinding.descriptorCount = 1;
    triangleBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding textureBinding{};
    textureBinding.binding = 2;
    textureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureBinding.descriptorCount = MaxTextureDescriptors;
    textureBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding resultBinding{};
    resultBinding.binding = 3;
    resultBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resultBinding.descriptorCount = 1;
    resultBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding binMaskBinding{};
    binMaskBinding.binding = 4;
    binMaskBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binMaskBinding.descriptorCount = 1;
    binMaskBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding groupListBinding{};
    groupListBinding.binding = 5;
    groupListBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    groupListBinding.descriptorCount = 1;
    groupListBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding toonBinding{};
    toonBinding.binding = 6;
    toonBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    toonBinding.descriptorCount = 1;
    toonBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding spanSetupBinding{};
    spanSetupBinding.binding = 7;
    spanSetupBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    spanSetupBinding.descriptorCount = 1;
    spanSetupBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding workOffsetBinding{};
    workOffsetBinding.binding = 8;
    workOffsetBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    workOffsetBinding.descriptorCount = 1;
    workOffsetBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding captureLineBinding{};
    captureLineBinding.binding = 9;
    captureLineBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    captureLineBinding.descriptorCount = 1;
    captureLineBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 10> bindings = {
        imageBinding,
        triangleBinding,
        textureBinding,
        resultBinding,
        binMaskBinding,
        groupListBinding,
        toonBinding,
        spanSetupBinding,
        workOffsetBinding,
        captureLineBinding};

    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutCreateInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(Device, &layoutCreateInfo, nullptr, &DescriptorSetLayout) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create descriptor set layout");
        return false;
    }

    constexpr u32 descriptorSetCount = static_cast<u32>(AsyncRenderContextCount + 1);
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = descriptorSetCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 8u * descriptorSetCount;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = MaxTextureDescriptors * descriptorSetCount;

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.maxSets = descriptorSetCount;
    poolCreateInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolCreateInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(Device, &poolCreateInfo, nullptr, &DescriptorPool) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create descriptor pool");
        return false;
    }

    std::array<VkDescriptorSetLayout, AsyncRenderContextCount + 1> descriptorSetLayouts{};
    descriptorSetLayouts.fill(DescriptorSetLayout);
    std::array<VkDescriptorSet, AsyncRenderContextCount + 1> descriptorSets{};

    VkDescriptorSetAllocateInfo descriptorAllocInfo{};
    descriptorAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorAllocInfo.descriptorPool = DescriptorPool;
    descriptorAllocInfo.descriptorSetCount = descriptorSetCount;
    descriptorAllocInfo.pSetLayouts = descriptorSetLayouts.data();

    if (vkAllocateDescriptorSets(Device, &descriptorAllocInfo, descriptorSets.data()) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate descriptor set");
        return false;
    }

    DescriptorSet = descriptorSets[0];
    for (size_t i = 0; i < AsyncRenderContextCount; i++)
        RenderContexts[i].DescriptorSet = descriptorSets[i + 1];
    invalidateAllDescriptorSetCaches();

    if (!createFallbackTexture())
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create fallback texture");
        return false;
    }

    return true;
}

std::string VulkanRenderer3D::buildPipelineCacheFileName(bool useNonUniformTextureIndexing) const
{
    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(PhysicalDevice, &deviceProperties);

    const u64 versionHash = fnv1a64(MELONDS_VERSION);
    char cacheFileName[256]{};
    std::snprintf(
        cacheFileName,
        sizeof(cacheFileName),
        "vulkan_pipeline_cache_v%u_%08x_%08x_%08x_%016llx_%s.bin",
        kPipelineCacheFileVersion,
        deviceProperties.vendorID,
        deviceProperties.deviceID,
        deviceProperties.driverVersion,
        static_cast<unsigned long long>(versionHash),
        useNonUniformTextureIndexing ? "nonuniform" : "compat"
    );
    return cacheFileName;
}

bool VulkanRenderer3D::createPipelineCache(bool useNonUniformTextureIndexing)
{
    if (ComputePipelineCache != VK_NULL_HANDLE)
        return true;

    ComputePipelineCacheFile = buildPipelineCacheFileName(useNonUniformTextureIndexing);

    std::vector<u8> cacheData;
    if (Platform::FileHandle* cacheFile = Platform::OpenLocalFile(ComputePipelineCacheFile, Platform::FileMode::Read))
    {
        const u64 cacheSize = Platform::FileLength(cacheFile);
        if (cacheSize > 0 && cacheSize <= (256ull * 1024ull * 1024ull))
        {
            cacheData.resize(static_cast<size_t>(cacheSize));
            if (Platform::FileRead(cacheData.data(), 1, cacheSize, cacheFile) != cacheSize)
            {
                Log(LogLevel::Warn, "VulkanRenderer3D: failed to read pipeline cache %s", ComputePipelineCacheFile.c_str());
                cacheData.clear();
            }
            else
            {
                Log(
                    LogLevel::Info,
                    "VulkanRenderer3D: loaded pipeline cache (%s, %llu bytes)",
                    ComputePipelineCacheFile.c_str(),
                    static_cast<unsigned long long>(cacheSize)
                );
            }
        }

        Platform::CloseFile(cacheFile);
    }

    VkPipelineCacheCreateInfo cacheCreateInfo{};
    cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheCreateInfo.initialDataSize = cacheData.size();
    cacheCreateInfo.pInitialData = cacheData.empty() ? nullptr : cacheData.data();

    VkResult cacheResult = vkCreatePipelineCache(Device, &cacheCreateInfo, nullptr, &ComputePipelineCache);
    if (cacheResult != VK_SUCCESS && !cacheData.empty())
    {
        Log(
            LogLevel::Warn,
            "VulkanRenderer3D: pipeline cache data rejected, recreating empty cache (%d)",
            static_cast<int>(cacheResult)
        );
        cacheCreateInfo.initialDataSize = 0;
        cacheCreateInfo.pInitialData = nullptr;
        cacheResult = vkCreatePipelineCache(Device, &cacheCreateInfo, nullptr, &ComputePipelineCache);
    }

    if (cacheResult != VK_SUCCESS)
    {
        Log(LogLevel::Warn, "VulkanRenderer3D: failed to create pipeline cache (%d)", static_cast<int>(cacheResult));
        ComputePipelineCache = VK_NULL_HANDLE;
        return false;
    }

    return true;
}

void VulkanRenderer3D::savePipelineCache()
{
    if (Device == VK_NULL_HANDLE || ComputePipelineCache == VK_NULL_HANDLE || ComputePipelineCacheFile.empty())
        return;

    size_t cacheSize = 0;
    if (vkGetPipelineCacheData(Device, ComputePipelineCache, &cacheSize, nullptr) != VK_SUCCESS || cacheSize == 0)
        return;

    std::vector<u8> cacheData(cacheSize);
    if (vkGetPipelineCacheData(Device, ComputePipelineCache, &cacheSize, cacheData.data()) != VK_SUCCESS || cacheSize == 0)
        return;

    Platform::FileHandle* cacheFile = Platform::OpenLocalFile(ComputePipelineCacheFile, Platform::FileMode::ReadWrite);
    if (cacheFile == nullptr)
    {
        Log(LogLevel::Warn, "VulkanRenderer3D: failed to open pipeline cache for writing (%s)", ComputePipelineCacheFile.c_str());
        return;
    }

    const u64 written = Platform::FileWrite(cacheData.data(), 1, cacheSize, cacheFile);
    Platform::FileFlush(cacheFile);
    Platform::CloseFile(cacheFile);

    if (written != cacheSize)
    {
        Log(
            LogLevel::Warn,
            "VulkanRenderer3D: incomplete pipeline cache write (%s %llu/%llu)",
            ComputePipelineCacheFile.c_str(),
            static_cast<unsigned long long>(written),
            static_cast<unsigned long long>(cacheSize)
        );
        return;
    }

    Log(
        LogLevel::Info,
        "VulkanRenderer3D: saved pipeline cache (%s, %llu bytes)",
        ComputePipelineCacheFile.c_str(),
        static_cast<unsigned long long>(cacheSize)
    );
}

bool VulkanRenderer3D::createComputePipeline()
{
    if (melonDS_gpu3d_vulkan_interp_spans_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_bin_combined_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_calc_work_offsets_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_sort_work_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_tri_raster_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_tri_raster_compat_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_depth_blend_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_final_pass_comp_spv_len == 0
        || melonDS_gpu3d_vulkan_capture_line_export_comp_spv_len == 0)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: empty SPIR-V blob(s)");
        return false;
    }

    const bool useNonUniformTextureIndexing = VulkanContext::Get().SupportsNonUniformTextureIndexing();
    if (!createPipelineCache(useNonUniformTextureIndexing))
    {
        Log(
            LogLevel::Warn,
            "VulkanRenderer3D: pipeline cache unavailable, continuing without persistent cache"
        );
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(RasterPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &DescriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(Device, &pipelineLayoutCreateInfo, nullptr, &PipelineLayout) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create pipeline layout");
        return false;
    }

    const auto createPipelineFromSpirv = [&](const unsigned char* spirvBytes,
                                             size_t spirvLength,
                                             const char* pipelineName,
                                             const VkSpecializationInfo* specializationInfo,
                                             VkPipeline* outPipeline) -> bool {
        std::vector<u32> shaderWords((spirvLength + sizeof(u32) - 1u) / sizeof(u32));
        std::memcpy(shaderWords.data(), spirvBytes, spirvLength);

        VkShaderModuleCreateInfo shaderCreateInfo{};
        shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderCreateInfo.codeSize = spirvLength;
        shaderCreateInfo.pCode = shaderWords.data();

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(Device, &shaderCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: failed to create %s shader module", pipelineName);
            return false;
        }

        VkPipelineShaderStageCreateInfo shaderStage{};
        shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStage.module = shaderModule;
        shaderStage.pName = "main";
        shaderStage.pSpecializationInfo = specializationInfo;

        VkComputePipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStage;
        pipelineCreateInfo.layout = PipelineLayout;

        const VkResult pipelineResult = vkCreateComputePipelines(
            Device,
            ComputePipelineCache,
            1,
            &pipelineCreateInfo,
            nullptr,
            outPipeline
        );

        vkDestroyShaderModule(Device, shaderModule, nullptr);

        if (pipelineResult != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: failed to create %s compute pipeline (%d)", pipelineName, static_cast<int>(pipelineResult));
            return false;
        }

        return true;
    };

    if (!createPipelineFromSpirv(
            melonDS_gpu3d_vulkan_interp_spans_comp_spv,
            melonDS_gpu3d_vulkan_interp_spans_comp_spv_len,
            "interp_spans",
            nullptr,
            &InterpPipeline))
    {
        return false;
    }

    if (!createPipelineFromSpirv(
            melonDS_gpu3d_vulkan_bin_combined_comp_spv,
            melonDS_gpu3d_vulkan_bin_combined_comp_spv_len,
            "bin",
            nullptr,
            &BinPipeline))
    {
        return false;
    }

    if (!createPipelineFromSpirv(
            melonDS_gpu3d_vulkan_calc_work_offsets_comp_spv,
            melonDS_gpu3d_vulkan_calc_work_offsets_comp_spv_len,
            "work_offsets",
            nullptr,
            &WorkOffsetsPipeline))
    {
        return false;
    }

    if (!createPipelineFromSpirv(
            melonDS_gpu3d_vulkan_sort_work_comp_spv,
            melonDS_gpu3d_vulkan_sort_work_comp_spv_len,
            "sort",
            nullptr,
            &SortPipeline))
    {
        return false;
    }

    if (!createPipelineFromSpirv(
            melonDS_gpu3d_vulkan_depth_blend_comp_spv,
            melonDS_gpu3d_vulkan_depth_blend_comp_spv_len,
            "depth_blend",
            nullptr,
            &DepthBlendPipeline))
    {
        return false;
    }

    struct RasterSpecializationData
    {
        u32 expectWBufferMode;
        u32 expectShadeMode;
        u32 expectTextureMode;
        u32 expectTranslucencyMode;
    };

    std::array<VkSpecializationMapEntry, 4> rasterSpecializationEntries{};
    rasterSpecializationEntries[0].constantID = 0;
    rasterSpecializationEntries[0].offset = offsetof(RasterSpecializationData, expectWBufferMode);
    rasterSpecializationEntries[0].size = sizeof(u32);
    rasterSpecializationEntries[1].constantID = 1;
    rasterSpecializationEntries[1].offset = offsetof(RasterSpecializationData, expectShadeMode);
    rasterSpecializationEntries[1].size = sizeof(u32);
    rasterSpecializationEntries[2].constantID = 2;
    rasterSpecializationEntries[2].offset = offsetof(RasterSpecializationData, expectTextureMode);
    rasterSpecializationEntries[2].size = sizeof(u32);
    rasterSpecializationEntries[3].constantID = 3;
    rasterSpecializationEntries[3].offset = offsetof(RasterSpecializationData, expectTranslucencyMode);
    rasterSpecializationEntries[3].size = sizeof(u32);

    const char* rasterWModeNames[] = {"z", "w", "any"};
    const char* rasterShadeModeNames[] = {"modulate", "decal", "toon", "highlight", "shadow", "any"};
    const char* rasterTextureModeNames[] = {"notexture", "texture", "anytex"};
    const char* rasterTranslucencyModeNames[] = {"opaque", "translucent", "anyalpha"};
    const unsigned char* triRasterSpirv = useNonUniformTextureIndexing
        ? melonDS_gpu3d_vulkan_tri_raster_comp_spv
        : melonDS_gpu3d_vulkan_tri_raster_compat_comp_spv;
    const size_t triRasterSpirvLen = useNonUniformTextureIndexing
        ? melonDS_gpu3d_vulkan_tri_raster_comp_spv_len
        : melonDS_gpu3d_vulkan_tri_raster_compat_comp_spv_len;
    Log(
        LogLevel::Warn,
        "VulkanRenderer3D: using %s texture sampling path",
        useNonUniformTextureIndexing ? "non-uniform descriptor indexing" : "compatibility (dynamic-uniform descriptor indexing)"
    );
    auto makeRasterPipelineIndex = [&](u32 rasterWMode, u32 rasterShadeMode, u32 rasterTextureMode, u32 rasterTranslucencyMode) -> u32
    {
        return (((rasterWMode * RasterShadeModeCount) + rasterShadeMode) * RasterTextureModeCount + rasterTextureMode)
            * RasterTranslucencyModeCount
            + rasterTranslucencyMode;
    };
    for (u32 rasterWMode = 0; rasterWMode < RasterWModeCount; rasterWMode++)
    {
        for (u32 rasterShadeMode = 0; rasterShadeMode < RasterShadeModeCount; rasterShadeMode++)
        {
            for (u32 rasterTextureMode = 0; rasterTextureMode < RasterTextureModeCount; rasterTextureMode++)
            {
                for (u32 rasterTranslucencyMode = 0; rasterTranslucencyMode < RasterTranslucencyModeCount; rasterTranslucencyMode++)
                {
                    RasterSpecializationData rasterSpecializationData{};
                    rasterSpecializationData.expectWBufferMode = rasterWMode;
                    rasterSpecializationData.expectShadeMode = rasterShadeMode;
                    rasterSpecializationData.expectTextureMode = rasterTextureMode;
                    rasterSpecializationData.expectTranslucencyMode = rasterTranslucencyMode;

                    VkSpecializationInfo rasterSpecializationInfo{};
                    rasterSpecializationInfo.mapEntryCount = static_cast<u32>(rasterSpecializationEntries.size());
                    rasterSpecializationInfo.pMapEntries = rasterSpecializationEntries.data();
                    rasterSpecializationInfo.dataSize = sizeof(rasterSpecializationData);
                    rasterSpecializationInfo.pData = &rasterSpecializationData;

                    const u32 rasterPipelineIndex = makeRasterPipelineIndex(
                        rasterWMode,
                        rasterShadeMode,
                        rasterTextureMode,
                        rasterTranslucencyMode
                    );
                    char rasterPipelineName[96]{};
                    std::snprintf(
                        rasterPipelineName,
                        sizeof(rasterPipelineName),
                        "raster_%s_%s_%s_%s",
                        rasterWModeNames[rasterWMode],
                        rasterShadeModeNames[rasterShadeMode],
                        rasterTextureModeNames[rasterTextureMode],
                        rasterTranslucencyModeNames[rasterTranslucencyMode]
                    );

                    if (!createPipelineFromSpirv(
                            triRasterSpirv,
                            triRasterSpirvLen,
                            rasterPipelineName,
                            &rasterSpecializationInfo,
                            &RasterPipelines[rasterPipelineIndex]))
                    {
                        return false;
                    }
                }
            }
        }
    }

    struct FinalSpecializationData
    {
        u32 enableEdgeMarking;
        u32 enableFog;
        u32 enableAntiAliasing;
    };

    std::array<VkSpecializationMapEntry, 3> finalSpecializationEntries{};
    finalSpecializationEntries[0].constantID = 0;
    finalSpecializationEntries[0].offset = offsetof(FinalSpecializationData, enableEdgeMarking);
    finalSpecializationEntries[0].size = sizeof(u32);
    finalSpecializationEntries[1].constantID = 1;
    finalSpecializationEntries[1].offset = offsetof(FinalSpecializationData, enableFog);
    finalSpecializationEntries[1].size = sizeof(u32);
    finalSpecializationEntries[2].constantID = 2;
    finalSpecializationEntries[2].offset = offsetof(FinalSpecializationData, enableAntiAliasing);
    finalSpecializationEntries[2].size = sizeof(u32);

    for (u32 finalPipelineIndex = 0; finalPipelineIndex < FinalPipelineVariantCount; finalPipelineIndex++)
    {
        FinalSpecializationData finalSpecializationData{};
        finalSpecializationData.enableEdgeMarking = finalPipelineIndex & 0x1u;
        finalSpecializationData.enableFog = (finalPipelineIndex >> 1u) & 0x1u;
        finalSpecializationData.enableAntiAliasing = (finalPipelineIndex >> 2u) & 0x1u;

        VkSpecializationInfo finalSpecializationInfo{};
        finalSpecializationInfo.mapEntryCount = static_cast<u32>(finalSpecializationEntries.size());
        finalSpecializationInfo.pMapEntries = finalSpecializationEntries.data();
        finalSpecializationInfo.dataSize = sizeof(finalSpecializationData);
        finalSpecializationInfo.pData = &finalSpecializationData;

        char finalPipelineName[32]{};
        std::snprintf(finalPipelineName, sizeof(finalPipelineName), "final_%u", finalPipelineIndex);
        if (!createPipelineFromSpirv(
                melonDS_gpu3d_vulkan_final_pass_comp_spv,
                melonDS_gpu3d_vulkan_final_pass_comp_spv_len,
                finalPipelineName,
                &finalSpecializationInfo,
                &FinalPipelines[finalPipelineIndex]))
        {
            return false;
        }
    }

    if (!createPipelineFromSpirv(
            melonDS_gpu3d_vulkan_capture_line_export_comp_spv,
            melonDS_gpu3d_vulkan_capture_line_export_comp_spv_len,
            "capture_line_export",
            nullptr,
            &CaptureLineExportPipeline))
    {
        return false;
    }

    return true;
}

bool VulkanRenderer3D::ensureRenderTarget(u32 width, u32 height)
{
    if (width == 0 || height == 0)
        return false;

    if (ColorImage != VK_NULL_HANDLE && ColorImageWidth == width && ColorImageHeight == height)
        return true;

    if ((ColorImage != VK_NULL_HANDLE
            || CaptureReadbackImage != VK_NULL_HANDLE
            || ReadbackBuffer != VK_NULL_HANDLE
            || ResultReadbackBuffer != VK_NULL_HANDLE
            || ResultBuffer != VK_NULL_HANDLE
            || BinMaskBuffer != VK_NULL_HANDLE
            || GroupListBuffer != VK_NULL_HANDLE
            || SpanSetupBuffer != VK_NULL_HANDLE
            || WorkOffsetBuffer != VK_NULL_HANDLE)
        && !waitForDeviceIdle("recreate render target"))
    {
        return false;
    }

    destroyReadbackBuffer();
    destroyCaptureReadbackImage();
    destroyResultReadbackBuffer();
    destroyBinMaskBuffer();
    destroyGroupListBuffer();
    destroySpanSetupBuffer();
    destroyWorkOffsetBuffer();
    destroyResultBuffer();
    destroyRenderTarget();

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageCreateInfo.extent.width = width;
    imageCreateInfo.extent.height = height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(Device, &imageCreateInfo, nullptr, &ColorImage) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create color image");
        return false;
    }

    VkMemoryRequirements imageMemoryRequirements{};
    vkGetImageMemoryRequirements(Device, ColorImage, &imageMemoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = imageMemoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = findMemoryType(imageMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
        memoryAllocateInfo.memoryTypeIndex = findMemoryType(imageMemoryRequirements.memoryTypeBits, 0);
    if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: unable to find memory type for color image");
        destroyRenderTarget();
        return false;
    }

    if (vkAllocateMemory(Device, &memoryAllocateInfo, nullptr, &ColorImageMemory) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate image memory");
        destroyRenderTarget();
        return false;
    }

    if (vkBindImageMemory(Device, ColorImage, ColorImageMemory, 0) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to bind image memory");
        destroyRenderTarget();
        return false;
    }

    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = ColorImage;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(Device, &imageViewCreateInfo, nullptr, &ColorImageView) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create image view");
        destroyRenderTarget();
        return false;
    }

    ColorImageWidth = width;
    ColorImageHeight = height;
    ColorImageInitialized = false;

    updateDescriptorSet(nullptr);

    return true;
}

void VulkanRenderer3D::destroyRenderTarget()
{
    invalidateAllDescriptorSetCaches();

    if (ColorImageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(Device, ColorImageView, nullptr);
        ColorImageView = VK_NULL_HANDLE;
    }

    if (ColorImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(Device, ColorImage, nullptr);
        ColorImage = VK_NULL_HANDLE;
    }

    if (ColorImageMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, ColorImageMemory, nullptr);
        ColorImageMemory = VK_NULL_HANDLE;
    }

    ColorImageWidth = 0;
    ColorImageHeight = 0;
    ColorImageInitialized = false;
}

bool VulkanRenderer3D::ensureTriangleBuffer(RenderContext* context, size_t triangleCount)
{
    static_assert(sizeof(TriangleGpu) == 120, "TriangleGpu layout must match std430 shader struct");
    VkBuffer& triangleBuffer = context != nullptr ? context->TriangleBuffer : TriangleBuffer;
    VkDeviceMemory& triangleMemory = context != nullptr ? context->TriangleMemory : TriangleMemory;
    VkDeviceSize& triangleBufferSize = context != nullptr ? context->TriangleBufferSize : TriangleBufferSize;
    void*& triangleMapped = context != nullptr ? context->TriangleMapped : TriangleMapped;
    const size_t requiredTriangleCount = std::max<size_t>(1, triangleCount);
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(requiredTriangleCount * sizeof(TriangleGpu));
    if (triangleBuffer != VK_NULL_HANDLE && triangleBufferSize >= requiredSize)
    {
        updateDescriptorSet(context);
        return true;
    }

    destroyTriangleBuffer(context);

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            triangleBuffer,
            triangleMemory,
            &triangleMapped))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate triangle buffer");
        destroyTriangleBuffer(context);
        return false;
    }

    triangleBufferSize = requiredSize;
    updateDescriptorSet(context);
    return true;
}

void VulkanRenderer3D::destroyTriangleBuffer(RenderContext* context)
{
    if (context != nullptr)
        invalidateDescriptorSetCache(context);
    else
        invalidateAllDescriptorSetCaches();

    VkBuffer& triangleBuffer = context != nullptr ? context->TriangleBuffer : TriangleBuffer;
    VkDeviceMemory& triangleMemory = context != nullptr ? context->TriangleMemory : TriangleMemory;
    VkDeviceSize& triangleBufferSize = context != nullptr ? context->TriangleBufferSize : TriangleBufferSize;
    void*& triangleMapped = context != nullptr ? context->TriangleMapped : TriangleMapped;

    if (triangleMapped != nullptr)
    {
        vkUnmapMemory(Device, triangleMemory);
        triangleMapped = nullptr;
    }

    if (triangleBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, triangleBuffer, nullptr);
        triangleBuffer = VK_NULL_HANDLE;
    }

    if (triangleMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, triangleMemory, nullptr);
        triangleMemory = VK_NULL_HANDLE;
    }

    triangleBufferSize = 0;
}

bool VulkanRenderer3D::ensureCpuBinBuffers(RenderContext& context, size_t triangleCount, u32 width, u32 height)
{
    if (width == 0 || height == 0)
        return false;

    constexpr u32 TileSize = 8;
    const u32 tilesPerLine = (width + (TileSize - 1u)) / TileSize;
    const u32 tileLines = (height + (TileSize - 1u)) / TileSize;
    const u32 tileCount = std::max<u32>(1u, tilesPerLine * tileLines);
    const u32 groupCount = std::max<u32>(1u, static_cast<u32>((triangleCount + 31u) / 32u));
    const VkDeviceSize requiredBinMaskSize = static_cast<VkDeviceSize>(tileCount) * static_cast<VkDeviceSize>(groupCount) * sizeof(u32);
    const VkDeviceSize requiredGroupListSize = static_cast<VkDeviceSize>(tileCount) * static_cast<VkDeviceSize>(groupCount + 1u) * sizeof(u32);

    if (context.BinMaskBuffer != VK_NULL_HANDLE
        && context.BinMaskMapped != nullptr
        && context.BinMaskBufferSize >= requiredBinMaskSize
        && context.GroupListBuffer != VK_NULL_HANDLE
        && context.GroupListMapped != nullptr
        && context.GroupListBufferSize >= requiredGroupListSize)
    {
        updateDescriptorSet(&context);
        return true;
    }

    destroyCpuBinBuffers(context);

    const auto createMappedStorageBuffer = [&](VkDeviceSize bufferSize,
                                               VkBuffer& buffer,
                                               VkDeviceMemory& memory,
                                               void*& mappedMemory) -> bool {
        return createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            buffer,
            memory,
            &mappedMemory);
    };

    if (!createMappedStorageBuffer(
            requiredBinMaskSize,
            context.BinMaskBuffer,
            context.BinMaskMemory,
            context.BinMaskMapped))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create CPU bin mask buffer");
        destroyCpuBinBuffers(context);
        return false;
    }

    if (!createMappedStorageBuffer(
            requiredGroupListSize,
            context.GroupListBuffer,
            context.GroupListMemory,
            context.GroupListMapped))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create CPU group list buffer");
        destroyCpuBinBuffers(context);
        return false;
    }

    context.BinMaskBufferSize = requiredBinMaskSize;
    context.GroupListBufferSize = requiredGroupListSize;
    updateDescriptorSet(&context);
    return true;
}

bool VulkanRenderer3D::ensureCpuSpanSetupBuffer(RenderContext& context, size_t triangleCount)
{
    static_assert(sizeof(SpanSetupGpu) == 44u, "SpanSetupGpu layout must match std430 shader struct");
    const size_t requiredTriangleCount = std::max<size_t>(1, triangleCount);
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(requiredTriangleCount * sizeof(SpanSetupGpu));

    if (context.SpanSetupBuffer != VK_NULL_HANDLE
        && context.SpanSetupMapped != nullptr
        && context.SpanSetupBufferSize >= requiredSize)
    {
        updateDescriptorSet(&context);
        return true;
    }

    destroyCpuSpanSetupBuffer(context);

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            context.SpanSetupBuffer,
            context.SpanSetupMemory,
            &context.SpanSetupMapped))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate CPU span setup buffer");
        destroyCpuSpanSetupBuffer(context);
        return false;
    }

    context.SpanSetupBufferSize = requiredSize;
    updateDescriptorSet(&context);
    return true;
}

void VulkanRenderer3D::destroyCpuSpanSetupBuffer(RenderContext& context)
{
    invalidateDescriptorSetCache(&context);

    if (context.SpanSetupMapped != nullptr)
    {
        vkUnmapMemory(Device, context.SpanSetupMemory);
        context.SpanSetupMapped = nullptr;
    }

    if (context.SpanSetupBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, context.SpanSetupBuffer, nullptr);
        context.SpanSetupBuffer = VK_NULL_HANDLE;
    }

    if (context.SpanSetupMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, context.SpanSetupMemory, nullptr);
        context.SpanSetupMemory = VK_NULL_HANDLE;
    }

    context.SpanSetupBufferSize = 0;
}

void VulkanRenderer3D::destroyCpuBinBuffers(RenderContext& context)
{
    invalidateDescriptorSetCache(&context);

    if (context.BinMaskMapped != nullptr)
    {
        vkUnmapMemory(Device, context.BinMaskMemory);
        context.BinMaskMapped = nullptr;
    }
    if (context.GroupListMapped != nullptr)
    {
        vkUnmapMemory(Device, context.GroupListMemory);
        context.GroupListMapped = nullptr;
    }

    if (context.BinMaskBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, context.BinMaskBuffer, nullptr);
        context.BinMaskBuffer = VK_NULL_HANDLE;
    }
    if (context.BinMaskMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, context.BinMaskMemory, nullptr);
        context.BinMaskMemory = VK_NULL_HANDLE;
    }
    context.BinMaskBufferSize = 0;

    if (context.GroupListBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, context.GroupListBuffer, nullptr);
        context.GroupListBuffer = VK_NULL_HANDLE;
    }
    if (context.GroupListMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, context.GroupListMemory, nullptr);
        context.GroupListMemory = VK_NULL_HANDLE;
    }
    context.GroupListBufferSize = 0;
}

bool VulkanRenderer3D::ensureCpuWorkOffsetBuffer(RenderContext& context, u32 width, u32 height, size_t triangleCount)
{
    if (width == 0 || height == 0)
        return false;

    constexpr u32 kTileSize = 8u;
    const u32 tilesPerLine = (width + (kTileSize - 1u)) / kTileSize;
    const u32 tileLines = (height + (kTileSize - 1u)) / kTileSize;
    const u32 tileCount = std::max<u32>(1u, tilesPerLine * tileLines);
    const u32 groupCount = std::max<u32>(1u, static_cast<u32>((triangleCount + 31u) / 32u));
    constexpr u32 kWorkOffsetHeaderWords = 8u;
    const u32 maxGroupListEntries = tileCount * groupCount;
    const u32 requiredWords = kWorkOffsetHeaderWords + (tileCount + 1u) + tileCount + maxGroupListEntries;
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(requiredWords) * sizeof(u32);

    if (context.WorkOffsetBuffer != VK_NULL_HANDLE
        && context.WorkOffsetMapped != nullptr
        && context.WorkOffsetBufferSize >= requiredSize)
    {
        updateDescriptorSet(&context);
        return true;
    }

    destroyCpuWorkOffsetBuffer(context);

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            context.WorkOffsetBuffer,
            context.WorkOffsetMemory,
            &context.WorkOffsetMapped))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate CPU work offset buffer");
        destroyCpuWorkOffsetBuffer(context);
        return false;
    }

    context.WorkOffsetBufferSize = requiredSize;
    updateDescriptorSet(&context);
    return true;
}

void VulkanRenderer3D::destroyCpuWorkOffsetBuffer(RenderContext& context)
{
    invalidateDescriptorSetCache(&context);

    if (context.WorkOffsetMapped != nullptr)
    {
        vkUnmapMemory(Device, context.WorkOffsetMemory);
        context.WorkOffsetMapped = nullptr;
    }

    if (context.WorkOffsetBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, context.WorkOffsetBuffer, nullptr);
        context.WorkOffsetBuffer = VK_NULL_HANDLE;
    }

    if (context.WorkOffsetMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, context.WorkOffsetMemory, nullptr);
        context.WorkOffsetMemory = VK_NULL_HANDLE;
    }

    context.WorkOffsetBufferSize = 0;
}

bool VulkanRenderer3D::ensureResultBuffer(u32 width, u32 height)
{
    if (width == 0 || height == 0)
        return false;

    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * ResultLayerCount * sizeof(u32);
    if (ResultBuffer != VK_NULL_HANDLE && ResultBufferSize == requiredSize)
    {
        updateDescriptorSet(nullptr);
        return true;
    }

    if (ResultBuffer != VK_NULL_HANDLE && !waitForDeviceIdle("resize result buffer"))
        return false;

    destroyResultBuffer();

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            0,
            ResultBuffer,
            ResultMemory))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate result buffer");
        destroyResultBuffer();
        return false;
    }

    ResultBufferSize = requiredSize;
    updateDescriptorSet(nullptr);
    return true;
}

void VulkanRenderer3D::destroyResultBuffer()
{
    invalidateAllDescriptorSetCaches();

    if (ResultBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, ResultBuffer, nullptr);
        ResultBuffer = VK_NULL_HANDLE;
    }

    if (ResultMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, ResultMemory, nullptr);
        ResultMemory = VK_NULL_HANDLE;
    }

    ResultBufferSize = 0;
}

bool VulkanRenderer3D::ensureBinMaskBuffer(size_t triangleCount, u32 width, u32 height)
{
    if (width == 0 || height == 0)
        return false;

    constexpr u32 TileSize = 8;
    const u32 tilesPerLine = (width + (TileSize - 1u)) / TileSize;
    const u32 tileLines = (height + (TileSize - 1u)) / TileSize;
    const u32 tileCount = std::max<u32>(1u, tilesPerLine * tileLines);
    const u32 groupCount = std::max<u32>(1u, static_cast<u32>((triangleCount + 31u) / 32u));

    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(tileCount) * static_cast<VkDeviceSize>(groupCount) * sizeof(u32);
    if (BinMaskBuffer != VK_NULL_HANDLE && BinMaskBufferSize >= requiredSize)
    {
        updateDescriptorSet(nullptr);
        return true;
    }

    if (BinMaskBuffer != VK_NULL_HANDLE && !waitForDeviceIdle("resize bin mask buffer"))
        return false;

    destroyBinMaskBuffer();

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            0,
            BinMaskBuffer,
            BinMaskMemory))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate bin mask buffer");
        destroyBinMaskBuffer();
        return false;
    }

    BinMaskBufferSize = requiredSize;
    updateDescriptorSet(nullptr);
    return true;
}

void VulkanRenderer3D::destroyBinMaskBuffer()
{
    invalidateAllDescriptorSetCaches();

    if (BinMaskBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, BinMaskBuffer, nullptr);
        BinMaskBuffer = VK_NULL_HANDLE;
    }

    if (BinMaskMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, BinMaskMemory, nullptr);
        BinMaskMemory = VK_NULL_HANDLE;
    }

    BinMaskBufferSize = 0;
}

bool VulkanRenderer3D::ensureGroupListBuffer(size_t triangleCount, u32 width, u32 height)
{
    if (width == 0 || height == 0)
        return false;

    constexpr u32 TileSize = 8;
    const u32 tilesPerLine = (width + (TileSize - 1u)) / TileSize;
    const u32 tileLines = (height + (TileSize - 1u)) / TileSize;
    const u32 tileCount = std::max<u32>(1u, tilesPerLine * tileLines);
    const u32 groupCount = std::max<u32>(1u, static_cast<u32>((triangleCount + 31u) / 32u));
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(tileCount) * static_cast<VkDeviceSize>(groupCount + 1u) * sizeof(u32);

    if (GroupListBuffer != VK_NULL_HANDLE && GroupListBufferSize >= requiredSize)
    {
        updateDescriptorSet(nullptr);
        return true;
    }

    if (GroupListBuffer != VK_NULL_HANDLE && !waitForDeviceIdle("resize group list buffer"))
        return false;

    destroyGroupListBuffer();

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            0,
            GroupListBuffer,
            GroupListMemory))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate group list buffer");
        destroyGroupListBuffer();
        return false;
    }

    GroupListBufferSize = requiredSize;
    updateDescriptorSet(nullptr);
    return true;
}

void VulkanRenderer3D::destroyGroupListBuffer()
{
    invalidateAllDescriptorSetCaches();

    if (GroupListBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, GroupListBuffer, nullptr);
        GroupListBuffer = VK_NULL_HANDLE;
    }

    if (GroupListMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, GroupListMemory, nullptr);
        GroupListMemory = VK_NULL_HANDLE;
    }

    GroupListBufferSize = 0;
}

bool VulkanRenderer3D::ensureSpanSetupBuffer(size_t triangleCount)
{
    const size_t requiredTriangleCount = std::max<size_t>(1, triangleCount);
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(requiredTriangleCount * sizeof(SpanSetupGpu));

    if (SpanSetupBuffer != VK_NULL_HANDLE && SpanSetupBufferSize >= requiredSize)
    {
        updateDescriptorSet(nullptr);
        return true;
    }

    if (SpanSetupBuffer != VK_NULL_HANDLE && !waitForDeviceIdle("resize span setup buffer"))
        return false;

    destroySpanSetupBuffer();

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            0,
            SpanSetupBuffer,
            SpanSetupMemory))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate span setup buffer");
        destroySpanSetupBuffer();
        return false;
    }

    SpanSetupBufferSize = requiredSize;
    updateDescriptorSet(nullptr);
    return true;
}

void VulkanRenderer3D::destroySpanSetupBuffer()
{
    invalidateAllDescriptorSetCaches();

    if (SpanSetupBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, SpanSetupBuffer, nullptr);
        SpanSetupBuffer = VK_NULL_HANDLE;
    }

    if (SpanSetupMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, SpanSetupMemory, nullptr);
        SpanSetupMemory = VK_NULL_HANDLE;
    }

    SpanSetupBufferSize = 0;
}

bool VulkanRenderer3D::ensureWorkOffsetBuffer(u32 width, u32 height, size_t triangleCount)
{
    if (width == 0 || height == 0)
        return false;

    constexpr u32 TileSize = 8;
    const u32 tilesPerLine = (width + (TileSize - 1u)) / TileSize;
    const u32 tileLines = (height + (TileSize - 1u)) / TileSize;
    const u32 tileCount = std::max<u32>(1u, tilesPerLine * tileLines);
    const u32 groupCount = std::max<u32>(1u, static_cast<u32>((triangleCount + 31u) / 32u));
    constexpr u32 WorkOffsetHeaderWords = 8u;
    const u32 maxGroupListEntries = tileCount * groupCount;
    const u32 requiredWords = WorkOffsetHeaderWords + (tileCount + 1u) + tileCount + maxGroupListEntries;
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(requiredWords) * sizeof(u32);

    if (WorkOffsetBuffer != VK_NULL_HANDLE && WorkOffsetBufferSize >= requiredSize)
    {
        updateDescriptorSet(nullptr);
        return true;
    }

    if (WorkOffsetBuffer != VK_NULL_HANDLE && !waitForDeviceIdle("resize work offset buffer"))
        return false;

    destroyWorkOffsetBuffer();

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            0,
            WorkOffsetBuffer,
            WorkOffsetMemory))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate work offset buffer");
        destroyWorkOffsetBuffer();
        return false;
    }

    WorkOffsetBufferSize = requiredSize;
    updateDescriptorSet(nullptr);
    return true;
}

void VulkanRenderer3D::destroyWorkOffsetBuffer()
{
    invalidateAllDescriptorSetCaches();

    if (WorkOffsetBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, WorkOffsetBuffer, nullptr);
        WorkOffsetBuffer = VK_NULL_HANDLE;
    }

    if (WorkOffsetMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, WorkOffsetMemory, nullptr);
        WorkOffsetMemory = VK_NULL_HANDLE;
    }

    WorkOffsetBufferSize = 0;
}

bool VulkanRenderer3D::ensureToonBuffer(RenderContext* context)
{
    VkBuffer& toonBuffer = context != nullptr ? context->ToonBuffer : ToonBuffer;
    VkDeviceMemory& toonMemory = context != nullptr ? context->ToonMemory : ToonMemory;
    VkDeviceSize& toonBufferSize = context != nullptr ? context->ToonBufferSize : ToonBufferSize;
    void*& toonMapped = context != nullptr ? context->ToonMapped : ToonMapped;
    const VkDeviceSize requiredSize = static_cast<VkDeviceSize>(ToonTableEntryCount) * sizeof(u32);
    if (toonBuffer != VK_NULL_HANDLE && toonBufferSize >= requiredSize)
    {
        updateDescriptorSet(context);
        return true;
    }

    destroyToonBuffer(context);

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            toonBuffer,
            toonMemory,
            &toonMapped))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate toon buffer");
        destroyToonBuffer(context);
        return false;
    }

    toonBufferSize = requiredSize;
    updateDescriptorSet(context);
    return true;
}

void VulkanRenderer3D::destroyToonBuffer(RenderContext* context)
{
    if (context != nullptr)
        invalidateDescriptorSetCache(context);
    else
        invalidateAllDescriptorSetCaches();

    VkBuffer& toonBuffer = context != nullptr ? context->ToonBuffer : ToonBuffer;
    VkDeviceMemory& toonMemory = context != nullptr ? context->ToonMemory : ToonMemory;
    VkDeviceSize& toonBufferSize = context != nullptr ? context->ToonBufferSize : ToonBufferSize;
    void*& toonMapped = context != nullptr ? context->ToonMapped : ToonMapped;

    if (toonMapped != nullptr)
    {
        vkUnmapMemory(Device, toonMemory);
        toonMapped = nullptr;
    }

    if (toonBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, toonBuffer, nullptr);
        toonBuffer = VK_NULL_HANDLE;
    }

    if (toonMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, toonMemory, nullptr);
        toonMemory = VK_NULL_HANDLE;
    }

    toonBufferSize = 0;
}

bool VulkanRenderer3D::updateToonBuffer(RenderContext* context, const u16* toonTable)
{
    const VkBuffer toonBuffer = context != nullptr ? context->ToonBuffer : ToonBuffer;
    const VkDeviceMemory toonMemory = context != nullptr ? context->ToonMemory : ToonMemory;
    const void* toonMapped = context != nullptr ? context->ToonMapped : ToonMapped;

    if (toonBuffer == VK_NULL_HANDLE || toonMemory == VK_NULL_HANDLE || toonMapped == nullptr)
        return false;

    u32* packedToon = reinterpret_cast<u32*>(const_cast<void*>(toonMapped));
    for (u32 i = 0; i < ToonTableEntryCount; i++)
    {
        const u32 toonColor = toonTable != nullptr ? static_cast<u32>(toonTable[i]) : 0u;
        u32 r = (toonColor << 1u) & 0x3Eu;
        u32 g = (toonColor >> 4u) & 0x3Eu;
        u32 b = (toonColor >> 9u) & 0x3Eu;
        if (r) r++;
        if (g) g++;
        if (b) b++;
        packedToon[i] = (r & 0x3Fu) | ((g & 0x3Fu) << 8u) | ((b & 0x3Fu) << 16u);
    }

    return true;
}

bool VulkanRenderer3D::ensureCaptureLineBuffer(RenderContext* context)
{
    VkBuffer& captureLineBuffer = context != nullptr ? context->CaptureLineBuffer : CaptureLineBuffer;
    VkDeviceMemory& captureLineMemory = context != nullptr ? context->CaptureLineMemory : CaptureLineMemory;
    VkDeviceSize& captureLineBufferSize = context != nullptr ? context->CaptureLineBufferSize : CaptureLineBufferSize;
    void*& captureLineMapped = context != nullptr ? context->CaptureLineMapped : CaptureLineMapped;
    constexpr VkDeviceSize requiredSize = static_cast<VkDeviceSize>(256u * 192u) * sizeof(u32);

    if (captureLineBuffer != VK_NULL_HANDLE && captureLineBufferSize >= requiredSize && captureLineMapped != nullptr)
    {
        updateDescriptorSet(context);
        return true;
    }

    destroyCaptureLineBuffer(context);

    if (!createBufferAllocation(
            Device,
            [&](u32 typeBits, VkMemoryPropertyFlags properties) { return findMemoryType(typeBits, properties); },
            requiredSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            captureLineBuffer,
            captureLineMemory,
            &captureLineMapped))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate capture line buffer");
        destroyCaptureLineBuffer(context);
        return false;
    }

    std::memset(captureLineMapped, 0, static_cast<size_t>(requiredSize));
    captureLineBufferSize = requiredSize;
    updateDescriptorSet(context);
    return true;
}

void VulkanRenderer3D::destroyCaptureLineBuffer(RenderContext* context)
{
    if (context != nullptr)
        invalidateDescriptorSetCache(context);
    else
        invalidateAllDescriptorSetCaches();

    VkBuffer& captureLineBuffer = context != nullptr ? context->CaptureLineBuffer : CaptureLineBuffer;
    VkDeviceMemory& captureLineMemory = context != nullptr ? context->CaptureLineMemory : CaptureLineMemory;
    VkDeviceSize& captureLineBufferSize = context != nullptr ? context->CaptureLineBufferSize : CaptureLineBufferSize;
    void*& captureLineMapped = context != nullptr ? context->CaptureLineMapped : CaptureLineMapped;

    if (captureLineMapped != nullptr && captureLineMemory != VK_NULL_HANDLE)
    {
        vkUnmapMemory(Device, captureLineMemory);
        captureLineMapped = nullptr;
    }

    if (captureLineBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, captureLineBuffer, nullptr);
        captureLineBuffer = VK_NULL_HANDLE;
    }

    if (captureLineMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, captureLineMemory, nullptr);
        captureLineMemory = VK_NULL_HANDLE;
    }

    captureLineBufferSize = 0;

    if (context == nullptr)
        resetCaptureLineState();
}

bool VulkanRenderer3D::createFallbackTexture()
{
    destroyFallbackTexture();

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UINT;
    imageCreateInfo.extent.width = 1;
    imageCreateInfo.extent.height = 1;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(Device, &imageCreateInfo, nullptr, &FallbackTextureImage) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create fallback texture image");
        return false;
    }

    VkMemoryRequirements imageRequirements{};
    vkGetImageMemoryRequirements(Device, FallbackTextureImage, &imageRequirements);

    VkMemoryAllocateInfo imageMemoryAllocateInfo{};
    imageMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imageMemoryAllocateInfo.allocationSize = imageRequirements.size;
    imageMemoryAllocateInfo.memoryTypeIndex = findMemoryType(imageRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (imageMemoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
        imageMemoryAllocateInfo.memoryTypeIndex = findMemoryType(imageRequirements.memoryTypeBits, 0);
    if (imageMemoryAllocateInfo.memoryTypeIndex == UINT32_MAX
        || vkAllocateMemory(Device, &imageMemoryAllocateInfo, nullptr, &FallbackTextureMemory) != VK_SUCCESS
        || vkBindImageMemory(Device, FallbackTextureImage, FallbackTextureMemory, 0) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate fallback texture memory");
        destroyFallbackTexture();
        return false;
    }

    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = FallbackTextureImage;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    imageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UINT;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(Device, &imageViewCreateInfo, nullptr, &FallbackTextureView) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create fallback texture view");
        destroyFallbackTexture();
        return false;
    }

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    if (vkCreateSampler(Device, &samplerCreateInfo, nullptr, &FallbackTextureSampler) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create fallback texture sampler");
        destroyFallbackTexture();
        return false;
    }

    VkBufferCreateInfo stagingCreateInfo{};
    stagingCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCreateInfo.size = sizeof(u32);
    stagingCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(Device, &stagingCreateInfo, nullptr, &FallbackTextureStagingBuffer) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create fallback staging buffer");
        destroyFallbackTexture();
        return false;
    }

    VkMemoryRequirements stagingRequirements{};
    vkGetBufferMemoryRequirements(Device, FallbackTextureStagingBuffer, &stagingRequirements);

    VkMemoryAllocateInfo stagingMemoryAllocateInfo{};
    stagingMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingMemoryAllocateInfo.allocationSize = stagingRequirements.size;
    stagingMemoryAllocateInfo.memoryTypeIndex = findMemoryType(
        stagingRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    if (stagingMemoryAllocateInfo.memoryTypeIndex == UINT32_MAX
        || vkAllocateMemory(Device, &stagingMemoryAllocateInfo, nullptr, &FallbackTextureStagingMemory) != VK_SUCCESS
        || vkBindBufferMemory(Device, FallbackTextureStagingBuffer, FallbackTextureStagingMemory, 0) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate fallback staging memory");
        destroyFallbackTexture();
        return false;
    }

    void* mappedMemory = nullptr;
    if (vkMapMemory(Device, FallbackTextureStagingMemory, 0, sizeof(u32), 0, &mappedMemory) != VK_SUCCESS)
    {
        destroyFallbackTexture();
        return false;
    }

    *reinterpret_cast<u32*>(mappedMemory) = 0x1F3F3F3Fu;
    vkUnmapMemory(Device, FallbackTextureStagingMemory);

    if (vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS
        || vkResetFences(Device, 1, &FrameFence) != VK_SUCCESS
        || vkResetCommandBuffer(CommandBuffer, 0) != VK_SUCCESS)
    {
        destroyFallbackTexture();
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(CommandBuffer, &beginInfo) != VK_SUCCESS)
    {
        destroyFallbackTexture();
        return false;
    }

    VkImageMemoryBarrier toTransferBarrier{};
    toTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferBarrier.srcAccessMask = 0;
    toTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.image = FallbackTextureImage;
    toTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferBarrier.subresourceRange.baseMipLevel = 0;
    toTransferBarrier.subresourceRange.levelCount = 1;
    toTransferBarrier.subresourceRange.baseArrayLayer = 0;
    toTransferBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toTransferBarrier
    );

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent.width = 1;
    copyRegion.imageExtent.height = 1;
    copyRegion.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(
        CommandBuffer,
        FallbackTextureStagingBuffer,
        FallbackTextureImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copyRegion
    );

    VkImageMemoryBarrier toGeneralBarrier{};
    toGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.image = FallbackTextureImage;
    toGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneralBarrier.subresourceRange.baseMipLevel = 0;
    toGeneralBarrier.subresourceRange.levelCount = 1;
    toGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    toGeneralBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toGeneralBarrier
    );

    if (vkEndCommandBuffer(CommandBuffer) != VK_SUCCESS)
    {
        destroyFallbackTexture();
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &CommandBuffer;
    {
        std::scoped_lock queueLock(VulkanContext::Get().GetQueueLock());
        const VkResult submitResult = vkQueueSubmit(Queue, 1, &submitInfo, FrameFence);
        if (submitResult != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: fallback texture vkQueueSubmit failed (%d)", static_cast<int>(submitResult));
            destroyFallbackTexture();
            return false;
        }
    }

    if (vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
    {
        destroyFallbackTexture();
        return false;
    }

    return true;
}

void VulkanRenderer3D::destroyFallbackTexture()
{
    invalidateAllDescriptorSetCaches();

    if (FallbackTextureSampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(Device, FallbackTextureSampler, nullptr);
        FallbackTextureSampler = VK_NULL_HANDLE;
    }

    if (FallbackTextureView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(Device, FallbackTextureView, nullptr);
        FallbackTextureView = VK_NULL_HANDLE;
    }

    if (FallbackTextureImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(Device, FallbackTextureImage, nullptr);
        FallbackTextureImage = VK_NULL_HANDLE;
    }

    if (FallbackTextureMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, FallbackTextureMemory, nullptr);
        FallbackTextureMemory = VK_NULL_HANDLE;
    }

    if (FallbackTextureStagingBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, FallbackTextureStagingBuffer, nullptr);
        FallbackTextureStagingBuffer = VK_NULL_HANDLE;
    }

    if (FallbackTextureStagingMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, FallbackTextureStagingMemory, nullptr);
        FallbackTextureStagingMemory = VK_NULL_HANDLE;
    }
}

bool VulkanRenderer3D::createReadbackBuffer(u32 width, u32 height)
{
    ReadbackSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * sizeof(u32);

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = ReadbackSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(Device, &bufferCreateInfo, nullptr, &ReadbackBuffer) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create readback buffer");
        return false;
    }

    VkMemoryRequirements bufferMemoryRequirements{};
    vkGetBufferMemoryRequirements(Device, ReadbackBuffer, &bufferMemoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = bufferMemoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = findMemoryType(
        bufferMemoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: unable to find memory type for readback buffer");
        destroyReadbackBuffer();
        return false;
    }

    if (vkAllocateMemory(Device, &memoryAllocateInfo, nullptr, &ReadbackMemory) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate readback memory");
        destroyReadbackBuffer();
        return false;
    }

    if (vkBindBufferMemory(Device, ReadbackBuffer, ReadbackMemory, 0) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to bind readback memory");
        destroyReadbackBuffer();
        return false;
    }

    if (vkMapMemory(Device, ReadbackMemory, 0, ReadbackSize, 0, &ReadbackMapped) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to map readback memory");
        destroyReadbackBuffer();
        return false;
    }

    RawReadbackRgba.assign(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    RawReadbackWidth = width;
    RawReadbackHeight = height;
    return true;
}

void VulkanRenderer3D::destroyReadbackBuffer()
{
    if (ReadbackMapped != nullptr && ReadbackMemory != VK_NULL_HANDLE)
    {
        vkUnmapMemory(Device, ReadbackMemory);
        ReadbackMapped = nullptr;
    }

    if (ReadbackBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, ReadbackBuffer, nullptr);
        ReadbackBuffer = VK_NULL_HANDLE;
    }

    if (ReadbackMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, ReadbackMemory, nullptr);
        ReadbackMemory = VK_NULL_HANDLE;
    }

    ReadbackSize = 0;
    RawReadbackWidth = 0;
    RawReadbackHeight = 0;
    RawReadbackRgba.clear();
}

bool VulkanRenderer3D::ensureCaptureReadbackImage()
{
    constexpr u32 kCaptureReadbackWidth = 256u;
    constexpr u32 kCaptureReadbackHeight = 192u;

    if (CaptureReadbackImage != VK_NULL_HANDLE)
        return true;

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageCreateInfo.extent.width = kCaptureReadbackWidth;
    imageCreateInfo.extent.height = kCaptureReadbackHeight;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(Device, &imageCreateInfo, nullptr, &CaptureReadbackImage) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create capture readback image");
        return false;
    }

    VkMemoryRequirements imageMemoryRequirements{};
    vkGetImageMemoryRequirements(Device, CaptureReadbackImage, &imageMemoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = imageMemoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = findMemoryType(imageMemoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
        memoryAllocateInfo.memoryTypeIndex = findMemoryType(imageMemoryRequirements.memoryTypeBits, 0);
    if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: unable to find memory type for capture readback image");
        destroyCaptureReadbackImage();
        return false;
    }

    if (vkAllocateMemory(Device, &memoryAllocateInfo, nullptr, &CaptureReadbackMemory) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate capture readback image memory");
        destroyCaptureReadbackImage();
        return false;
    }

    if (vkBindImageMemory(Device, CaptureReadbackImage, CaptureReadbackMemory, 0) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to bind capture readback image memory");
        destroyCaptureReadbackImage();
        return false;
    }

    CaptureReadbackImageInitialized = false;
    return true;
}

void VulkanRenderer3D::destroyCaptureReadbackImage()
{
    if (CaptureReadbackImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(Device, CaptureReadbackImage, nullptr);
        CaptureReadbackImage = VK_NULL_HANDLE;
    }

    if (CaptureReadbackMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, CaptureReadbackMemory, nullptr);
        CaptureReadbackMemory = VK_NULL_HANDLE;
    }

    CaptureReadbackImageInitialized = false;
}

bool VulkanRenderer3D::createResultReadbackBuffer()
{
    ResultReadbackSize = ResultBufferSize;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = ResultReadbackSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(Device, &bufferCreateInfo, nullptr, &ResultReadbackBuffer) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to create result readback buffer");
        return false;
    }

    VkMemoryRequirements bufferMemoryRequirements{};
    vkGetBufferMemoryRequirements(Device, ResultReadbackBuffer, &bufferMemoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = bufferMemoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = findMemoryType(
        bufferMemoryRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: unable to find memory type for result readback buffer");
        destroyResultReadbackBuffer();
        return false;
    }

    if (vkAllocateMemory(Device, &memoryAllocateInfo, nullptr, &ResultReadbackMemory) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to allocate result readback memory");
        destroyResultReadbackBuffer();
        return false;
    }

    if (vkBindBufferMemory(Device, ResultReadbackBuffer, ResultReadbackMemory, 0) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to bind result readback memory");
        destroyResultReadbackBuffer();
        return false;
    }

    if (vkMapMemory(Device, ResultReadbackMemory, 0, ResultReadbackSize, 0, &ResultReadbackMapped) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to map result readback memory");
        destroyResultReadbackBuffer();
        return false;
    }

    RawResultReadback.assign(static_cast<size_t>(ResultReadbackSize / sizeof(u32)), 0u);
    return true;
}

void VulkanRenderer3D::destroyResultReadbackBuffer()
{
    if (ResultReadbackMapped != nullptr && ResultReadbackMemory != VK_NULL_HANDLE)
    {
        vkUnmapMemory(Device, ResultReadbackMemory);
        ResultReadbackMapped = nullptr;
    }

    if (ResultReadbackBuffer != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(Device, ResultReadbackBuffer, nullptr);
        ResultReadbackBuffer = VK_NULL_HANDLE;
    }

    if (ResultReadbackMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(Device, ResultReadbackMemory, nullptr);
        ResultReadbackMemory = VK_NULL_HANDLE;
    }

    ResultReadbackSize = 0;
    RawResultReadback.clear();
}

bool VulkanRenderer3D::descriptorImageInfoEquals(const VkDescriptorImageInfo& lhs, const VkDescriptorImageInfo& rhs)
{
    return lhs.sampler == rhs.sampler
        && lhs.imageView == rhs.imageView
        && lhs.imageLayout == rhs.imageLayout;
}

VulkanRenderer3D::DescriptorSetCache& VulkanRenderer3D::getDescriptorSetCache(RenderContext* context)
{
    return context != nullptr ? context->DescriptorCache : DescriptorCache;
}

void VulkanRenderer3D::invalidateDescriptorSetCache(RenderContext* context)
{
    getDescriptorSetCache(context).Ready = false;
}

void VulkanRenderer3D::invalidateAllDescriptorSetCaches()
{
    DescriptorCache.Ready = false;
    for (RenderContext& renderContext : RenderContexts)
        renderContext.DescriptorCache.Ready = false;
}

void VulkanRenderer3D::updateDescriptorSet(RenderContext* context)
{
    const VkDescriptorSet DescriptorSet = context != nullptr ? context->DescriptorSet : this->DescriptorSet;
    const VkBuffer TriangleBuffer = context != nullptr ? context->TriangleBuffer : this->TriangleBuffer;
    const VkBuffer BinMaskBuffer = context != nullptr && context->BinMaskBuffer != VK_NULL_HANDLE
        ? context->BinMaskBuffer
        : this->BinMaskBuffer;
    const VkBuffer GroupListBuffer = context != nullptr && context->GroupListBuffer != VK_NULL_HANDLE
        ? context->GroupListBuffer
        : this->GroupListBuffer;
    const VkBuffer SpanSetupBuffer = context != nullptr && context->SpanSetupBuffer != VK_NULL_HANDLE
        ? context->SpanSetupBuffer
        : this->SpanSetupBuffer;
    const VkBuffer WorkOffsetBuffer = context != nullptr && context->WorkOffsetBuffer != VK_NULL_HANDLE
        ? context->WorkOffsetBuffer
        : this->WorkOffsetBuffer;
    const VkBuffer ToonBuffer = context != nullptr ? context->ToonBuffer : this->ToonBuffer;
    const VkBuffer CaptureLineBuffer = context != nullptr ? context->CaptureLineBuffer : this->CaptureLineBuffer;

    if (DescriptorSet == VK_NULL_HANDLE
        || ColorImageView == VK_NULL_HANDLE
        || TriangleBuffer == VK_NULL_HANDLE
        || ResultBuffer == VK_NULL_HANDLE
        || BinMaskBuffer == VK_NULL_HANDLE
        || GroupListBuffer == VK_NULL_HANDLE
        || SpanSetupBuffer == VK_NULL_HANDLE
        || WorkOffsetBuffer == VK_NULL_HANDLE
        || ToonBuffer == VK_NULL_HANDLE
        || CaptureLineBuffer == VK_NULL_HANDLE
        || FallbackTextureView == VK_NULL_HANDLE
        || FallbackTextureSampler == VK_NULL_HANDLE)
        return;

    DescriptorSetCache& descriptorCache = getDescriptorSetCache(context);

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = ColorImageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo triangleInfo{};
    triangleInfo.buffer = TriangleBuffer;
    triangleInfo.offset = 0;
    triangleInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo resultInfo{};
    resultInfo.buffer = ResultBuffer;
    resultInfo.offset = 0;
    resultInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo binMaskInfo{};
    binMaskInfo.buffer = BinMaskBuffer;
    binMaskInfo.offset = 0;
    binMaskInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo groupListInfo{};
    groupListInfo.buffer = GroupListBuffer;
    groupListInfo.offset = 0;
    groupListInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo toonInfo{};
    toonInfo.buffer = ToonBuffer;
    toonInfo.offset = 0;
    toonInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo spanSetupInfo{};
    spanSetupInfo.buffer = SpanSetupBuffer;
    spanSetupInfo.offset = 0;
    spanSetupInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo workOffsetInfo{};
    workOffsetInfo.buffer = WorkOffsetBuffer;
    workOffsetInfo.offset = 0;
    workOffsetInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo captureLineInfo{};
    captureLineInfo.buffer = CaptureLineBuffer;
    captureLineInfo.offset = 0;
    captureLineInfo.range = VK_WHOLE_SIZE;

    std::array<VkDescriptorImageInfo, MaxTextureDescriptors> textureInfos{};
    VkDescriptorImageInfo fallbackInfo{};
    fallbackInfo.sampler = FallbackTextureSampler;
    fallbackInfo.imageView = FallbackTextureView;
    fallbackInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    textureInfos.fill(fallbackInfo);
    for (u32 i = 0; i < ActiveTextureDescriptorCount && i < MaxActiveTextureDescriptors; i++)
    {
        textureInfos[i] = ActiveTextureDescriptors[i];
    }

    std::array<VkWriteDescriptorSet, 10> writes{};
    u32 writeCount = 0;
    if (!descriptorCache.Ready || descriptorCache.ColorImageView != ColorImageView)
    {
        writes[writeCount++] = makeImageDescriptorWrite(DescriptorSet, 0, &imageInfo, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    }

    if (!descriptorCache.Ready || descriptorCache.TriangleBuffer != TriangleBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 1, &triangleInfo);
    }

    bool texturesChanged = !descriptorCache.Ready;
    if (!texturesChanged)
    {
        for (u32 i = 0; i < MaxTextureDescriptors; i++)
        {
            if (!descriptorImageInfoEquals(descriptorCache.TextureInfos[i], textureInfos[i]))
            {
                texturesChanged = true;
                break;
            }
        }
    }
    if (texturesChanged)
    {
        writes[writeCount++] = makeImageDescriptorWrite(
            DescriptorSet,
            2,
            textureInfos.data(),
            MaxTextureDescriptors,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    }

    if (!descriptorCache.Ready || descriptorCache.ResultBuffer != ResultBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 3, &resultInfo);
    }
    if (!descriptorCache.Ready || descriptorCache.BinMaskBuffer != BinMaskBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 4, &binMaskInfo);
    }
    if (!descriptorCache.Ready || descriptorCache.GroupListBuffer != GroupListBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 5, &groupListInfo);
    }
    if (!descriptorCache.Ready || descriptorCache.ToonBuffer != ToonBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 6, &toonInfo);
    }
    if (!descriptorCache.Ready || descriptorCache.SpanSetupBuffer != SpanSetupBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 7, &spanSetupInfo);
    }
    if (!descriptorCache.Ready || descriptorCache.WorkOffsetBuffer != WorkOffsetBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 8, &workOffsetInfo);
    }
    if (!descriptorCache.Ready || descriptorCache.CaptureLineBuffer != CaptureLineBuffer)
    {
        writes[writeCount++] = makeBufferDescriptorWrite(DescriptorSet, 9, &captureLineInfo);
    }

    if (writeCount > 0)
        vkUpdateDescriptorSets(Device, writeCount, writes.data(), 0, nullptr);

    descriptorCache.Ready = true;
    descriptorCache.ColorImageView = ColorImageView;
    descriptorCache.TriangleBuffer = TriangleBuffer;
    descriptorCache.FallbackTextureView = FallbackTextureView;
    descriptorCache.FallbackTextureSampler = FallbackTextureSampler;
    descriptorCache.ResultBuffer = ResultBuffer;
    descriptorCache.BinMaskBuffer = BinMaskBuffer;
    descriptorCache.GroupListBuffer = GroupListBuffer;
    descriptorCache.ToonBuffer = ToonBuffer;
    descriptorCache.SpanSetupBuffer = SpanSetupBuffer;
    descriptorCache.WorkOffsetBuffer = WorkOffsetBuffer;
    descriptorCache.CaptureLineBuffer = CaptureLineBuffer;
    descriptorCache.TextureInfos = textureInfos;
}

u32 VulkanRenderer3D::findMemoryType(u32 typeBits, VkMemoryPropertyFlags properties) const
{
    return VulkanContext::Get().FindMemoryType(typeBits, properties);
}

bool VulkanRenderer3D::dispatchRasterAndReadback(
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
    bool captureReadbackPath)
{
    auto hasAllRasterPipelines = [&]() -> bool {
        for (const VkPipeline pipeline : RasterPipelines)
        {
            if (pipeline == VK_NULL_HANDLE)
                return false;
        }
        return true;
    };

    auto hasAllFinalPipelines = [&]() -> bool {
        for (const VkPipeline pipeline : FinalPipelines)
        {
            if (pipeline == VK_NULL_HANDLE)
                return false;
        }
        return true;
    };

    const bool useSynchronousContext = context == nullptr;
    const VkCommandBuffer CommandBuffer = context != nullptr ? context->CommandBuffer : this->CommandBuffer;
    const VkFence FrameFence = context != nullptr ? context->FrameFence : this->FrameFence;
    const VkDescriptorSet DescriptorSet = context != nullptr ? context->DescriptorSet : this->DescriptorSet;
    const VkBuffer TriangleBuffer = context != nullptr ? context->TriangleBuffer : this->TriangleBuffer;
    const VkDeviceMemory TriangleMemory = context != nullptr ? context->TriangleMemory : this->TriangleMemory;
    const VkDeviceSize TriangleBufferSize = context != nullptr ? context->TriangleBufferSize : this->TriangleBufferSize;
    const VkBuffer BinMaskBuffer = context != nullptr && context->BinMaskBuffer != VK_NULL_HANDLE
        ? context->BinMaskBuffer
        : this->BinMaskBuffer;
    const VkDeviceSize BinMaskBufferSize = context != nullptr && context->BinMaskBuffer != VK_NULL_HANDLE
        ? context->BinMaskBufferSize
        : this->BinMaskBufferSize;
    const VkBuffer GroupListBuffer = context != nullptr && context->GroupListBuffer != VK_NULL_HANDLE
        ? context->GroupListBuffer
        : this->GroupListBuffer;
    const VkDeviceSize GroupListBufferSize = context != nullptr && context->GroupListBuffer != VK_NULL_HANDLE
        ? context->GroupListBufferSize
        : this->GroupListBufferSize;
    const VkBuffer SpanSetupBuffer = context != nullptr && context->SpanSetupBuffer != VK_NULL_HANDLE
        ? context->SpanSetupBuffer
        : this->SpanSetupBuffer;
    const VkDeviceSize SpanSetupBufferSize = context != nullptr && context->SpanSetupBuffer != VK_NULL_HANDLE
        ? context->SpanSetupBufferSize
        : this->SpanSetupBufferSize;
    const VkBuffer WorkOffsetBuffer = context != nullptr && context->WorkOffsetBuffer != VK_NULL_HANDLE
        ? context->WorkOffsetBuffer
        : this->WorkOffsetBuffer;
    const VkDeviceSize WorkOffsetBufferSize = context != nullptr && context->WorkOffsetBuffer != VK_NULL_HANDLE
        ? context->WorkOffsetBufferSize
        : this->WorkOffsetBufferSize;
    const VkBuffer ToonBuffer = context != nullptr ? context->ToonBuffer : this->ToonBuffer;
    const VkDeviceMemory ToonMemory = context != nullptr ? context->ToonMemory : this->ToonMemory;
    const VkDeviceSize ToonBufferSize = context != nullptr ? context->ToonBufferSize : this->ToonBufferSize;
    const VkBuffer CaptureLineBuffer = context != nullptr ? context->CaptureLineBuffer : this->CaptureLineBuffer;
    void* captureLineMapped = context != nullptr ? context->CaptureLineMapped : CaptureLineMapped;
    const bool exportCaptureLine = captureReadbackPath;
    const bool useCpuDirectTiles = context != nullptr && useCpuTileBinning();
    const bool useLegacyRasterWorklist = ActiveRasterDispatchPath == RasterDispatchPath::LegacyWorklist;
    bool useCpuActiveTileDispatch = false;

    if (Device == VK_NULL_HANDLE
        || Queue == VK_NULL_HANDLE
        || InterpPipeline == VK_NULL_HANDLE
        || BinPipeline == VK_NULL_HANDLE
        || WorkOffsetsPipeline == VK_NULL_HANDLE
        || SortPipeline == VK_NULL_HANDLE
        || DepthBlendPipeline == VK_NULL_HANDLE
        || !hasAllRasterPipelines()
        || !hasAllFinalPipelines()
        || ColorImage == VK_NULL_HANDLE
        || ResultBuffer == VK_NULL_HANDLE
        || BinMaskBuffer == VK_NULL_HANDLE
        || GroupListBuffer == VK_NULL_HANDLE
        || SpanSetupBuffer == VK_NULL_HANDLE
        || WorkOffsetBuffer == VK_NULL_HANDLE
        || ToonBuffer == VK_NULL_HANDLE
        || TriangleBuffer == VK_NULL_HANDLE
        || TriangleMemory == VK_NULL_HANDLE)
        return false;

    if (exportCaptureLine && (CaptureLineBuffer == VK_NULL_HANDLE || captureLineMapped == nullptr))
        return false;

    constexpr u32 kCaptureReadbackWidth = 256u;
    constexpr u32 kCaptureReadbackHeight = 192u;
    const bool useCaptureDownscaleForReadback = readbackToCpu
        && captureReadbackPath
        && (ColorImageWidth != kCaptureReadbackWidth || ColorImageHeight != kCaptureReadbackHeight);
    const u32 readbackWidth = useCaptureDownscaleForReadback ? kCaptureReadbackWidth : ColorImageWidth;
    const u32 readbackHeight = useCaptureDownscaleForReadback ? kCaptureReadbackHeight : ColorImageHeight;

    if (readbackToCpu)
    {
        if (useCaptureDownscaleForReadback && !ensureCaptureReadbackImage())
            return false;

        const VkDeviceSize requiredReadbackSize = static_cast<VkDeviceSize>(readbackWidth) * static_cast<VkDeviceSize>(readbackHeight) * sizeof(u32);
        if (ReadbackBuffer == VK_NULL_HANDLE || ReadbackMemory == VK_NULL_HANDLE || ReadbackSize != requiredReadbackSize)
        {
            destroyReadbackBuffer();
            if (!createReadbackBuffer(readbackWidth, readbackHeight))
                return false;
        }
    }

    if (useSynchronousContext)
    {
        const u64 waitStartNs = PerfNowNs();
        const VkResult waitResult = vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX);
        if (waitResult != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: vkWaitForFences failed (%d)", static_cast<int>(waitResult));
            return false;
        }

        FenceWaitCpuWindow.Add(PerfNowNs() - waitStartNs);
        consumeGpuTiming(nullptr);
    }

    if (vkResetFences(Device, 1, &FrameFence) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: vkResetFences failed");
        return false;
    }

    if (vkResetCommandBuffer(CommandBuffer, 0) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: vkResetCommandBuffer failed");
        return false;
    }

    RasterPushConstants pushConstants{};
    pushConstants.width = ColorImageWidth;
    pushConstants.height = ColorImageHeight;
    pushConstants.clearColor = rgbaColor;
    pushConstants.clearDepth = clearDepth;
    pushConstants.triangleCount = static_cast<u32>(Triangles.size());
    pushConstants.dispCnt = dispCnt;
    pushConstants.alphaRef = alphaRef;
    pushConstants.fogColor = fogColor;
    pushConstants.fogOffset = fogOffset;
    pushConstants.fogShift = fogShift;
    pushConstants.clearAttr = clearAttr;

    for (u32 i = 0; i < 34; i++)
    {
        const u32 density = fogDensityTable != nullptr ? static_cast<u32>(fogDensityTable[i]) : 0u;
        const u32 packedWord = i / 4;
        const u32 packedShift = (i % 4) * 8;
        pushConstants.fogDensityPacked[packedWord] |= (density & 0xFFu) << packedShift;
    }

    for (u32 i = 0; i < 8; i++)
    {
        const u32 edgeColor = edgeColorTable != nullptr ? static_cast<u32>(edgeColorTable[i]) : 0u;
        u32 r = (edgeColor << 1u) & 0x3Eu;
        u32 g = (edgeColor >> 4u) & 0x3Eu;
        u32 b = (edgeColor >> 9u) & 0x3Eu;
        if (r) r++;
        if (g) g++;
        if (b) b++;
        pushConstants.edgeColorPacked[i] = (r & 0x3Fu) | ((g & 0x3Fu) << 8u) | ((b & 0x3Fu) << 16u);
    }
    constexpr u32 kVariantWildcard = 0xFFFFFFFFu;
    constexpr u32 kVariantFlagTextured = 1u << 0u;
    constexpr u32 kVariantFlagDecal = 1u << 1u;
    constexpr u32 kVariantFlagModulate = 1u << 2u;
    constexpr u32 kVariantFlagToon = 1u << 3u;
    constexpr u32 kVariantFlagHighlight = 1u << 4u;
    constexpr u32 kVariantFlagShadow = 1u << 5u;
    constexpr u32 kVariantFlagWBuffer = 1u << 6u;
    constexpr u32 kVariantFlagTranslucent = 1u << 7u;
    constexpr u32 kDebugFlagFinalActiveTileMask = 1u << 31u;
    constexpr u32 kVariantPipelineMask = (1u << 8u) - 1u;
    constexpr u32 kTriangleFlagWBuffer = 1u << 4u;
    constexpr u32 kRasterWModeZ = 0u;
    constexpr u32 kRasterWModeW = 1u;
    constexpr u32 kRasterWModeAny = 2u;
    constexpr u32 kRasterShadeModeModulate = 0u;
    constexpr u32 kRasterShadeModeDecal = 1u;
    constexpr u32 kRasterShadeModeToon = 2u;
    constexpr u32 kRasterShadeModeHighlight = 3u;
    constexpr u32 kRasterShadeModeShadow = 4u;
    constexpr u32 kRasterShadeModeAny = 5u;
    constexpr u32 kRasterTextureModeNoTexture = 0u;
    constexpr u32 kRasterTextureModeUseTexture = 1u;
    constexpr u32 kRasterTextureModeAny = 2u;
    constexpr u32 kRasterTranslucencyModeOpaque = 0u;
    constexpr u32 kRasterTranslucencyModeTranslucent = 1u;
    constexpr u32 kRasterTranslucencyModeAny = 2u;
    const bool frameWBufferMode = !Triangles.empty() && ((Triangles[0].flags & kTriangleFlagWBuffer) != 0u);
    for (TriangleGpu& triangle : Triangles)
    {
        if (frameWBufferMode)
        {
            triangle.flags |= kTriangleFlagWBuffer;
            triangle.variantKey |= kVariantFlagWBuffer;
        }
        else
        {
            triangle.flags &= ~kTriangleFlagWBuffer;
            triangle.variantKey &= ~kVariantFlagWBuffer;
        }
    }
    auto resolveRasterWMode = [frameWBufferMode](u32 variantKey) -> u32 {
        (void)variantKey;
        return frameWBufferMode ? kRasterWModeW : kRasterWModeZ;
    };
    auto resolveRasterShadeMode = [](u32 variantKey) -> u32 {
        if (variantKey == kVariantWildcard)
            return kRasterShadeModeAny;
        if ((variantKey & kVariantFlagShadow) != 0u)
            return kRasterShadeModeShadow;
        if ((variantKey & kVariantFlagHighlight) != 0u)
            return kRasterShadeModeHighlight;
        if ((variantKey & kVariantFlagToon) != 0u)
            return kRasterShadeModeToon;
        if ((variantKey & kVariantFlagDecal) != 0u)
            return kRasterShadeModeDecal;
        if ((variantKey & kVariantFlagModulate) != 0u)
            return kRasterShadeModeModulate;
        return kRasterShadeModeAny;
    };
    auto resolveRasterTextureMode = [](u32 variantKey) -> u32 {
        if (variantKey == kVariantWildcard)
            return kRasterTextureModeAny;
        return (variantKey & kVariantFlagTextured) != 0u ? kRasterTextureModeUseTexture : kRasterTextureModeNoTexture;
    };
    auto resolveRasterTranslucencyMode = [](u32 variantKey) -> u32 {
        if (variantKey == kVariantWildcard)
            return kRasterTranslucencyModeAny;
        return (variantKey & kVariantFlagTranslucent) != 0u
            ? kRasterTranslucencyModeTranslucent
            : kRasterTranslucencyModeOpaque;
    };
    auto makeRasterPipelineIndex = [&](u32 rasterWMode, u32 rasterShadeMode, u32 rasterTextureMode, u32 rasterTranslucencyMode) -> u32 {
        return (((rasterWMode * RasterShadeModeCount) + rasterShadeMode) * RasterTextureModeCount + rasterTextureMode)
            * RasterTranslucencyModeCount
            + rasterTranslucencyMode;
    };
    pushConstants.variantKey = kVariantWildcard;
    pushConstants.passIndex = 0;
    pushConstants.triangleBase = 0;
    pushConstants.depthBlendMode = frameWBufferMode ? 1u : 0u;
    pushConstants.debugFlags = MelonDSAndroid::getVulkanDiagnosticFlags();
    pushConstants.debugFlags &= ~kDebugFlagFinalActiveTileMask;
    pushConstants.captureScale = std::max(1u, ScaleFactor > 0 ? static_cast<u32>(ScaleFactor) : 1u);
    pushConstants.captureSampleOffset = pushConstants.captureScale > 1u ? (pushConstants.captureScale / 2u) : 0u;
    pushConstants.captureFlags = exportCaptureLine ? 1u : 0u;

    TriangleCountWindow.Add(static_cast<u64>(Triangles.size()));
    PassCountWindow.Add(1);

    void* mappedTriangles = context != nullptr ? context->TriangleMapped : TriangleMapped;
    if (mappedTriangles == nullptr)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: triangle buffer is not mapped");
        return false;
    }

    if (!Triangles.empty())
        std::memcpy(mappedTriangles, Triangles.data(), Triangles.size() * sizeof(TriangleGpu));
    else
        std::memset(mappedTriangles, 0, sizeof(TriangleGpu));

    if (!updateToonBuffer(context, toonTable))
    {
        Log(LogLevel::Error, "VulkanRenderer3D: failed to update toon buffer");
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(CommandBuffer, &beginInfo) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: vkBeginCommandBuffer failed");
        return false;
    }

    VkQueryPool timestampQueryPool = context != nullptr ? context->TimestampQueryPool : TimestampQueryPool;
    if (timestampQueryPool != VK_NULL_HANDLE && ResetQueryPool != nullptr)
        ResetQueryPool(Device, timestampQueryPool, 0, TimestampQueryCount);
    if (timestampQueryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueryPool, 0);

    VkImageMemoryBarrier toGeneralBarrier{};
    toGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneralBarrier.srcAccessMask = ColorImageInitialized ? (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT) : 0;
    toGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toGeneralBarrier.oldLayout = ColorImageInitialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.image = ColorImage;
    toGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneralBarrier.subresourceRange.baseMipLevel = 0;
    toGeneralBarrier.subresourceRange.levelCount = 1;
    toGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    toGeneralBarrier.subresourceRange.layerCount = 1;

    const VkPipelineStageFlags toGeneralSrcStage = ColorImageInitialized
        ? (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT)
        : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    vkCmdPipelineBarrier(
        CommandBuffer,
        toGeneralSrcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toGeneralBarrier
    );

    VkBufferMemoryBarrier triangleBufferBarrier{};
    triangleBufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    triangleBufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    triangleBufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    triangleBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    triangleBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    triangleBufferBarrier.buffer = TriangleBuffer;
    triangleBufferBarrier.offset = 0;
    triangleBufferBarrier.size = TriangleBufferSize;

    VkBufferMemoryBarrier toonBufferBarrier{};
    toonBufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toonBufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    toonBufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toonBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toonBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toonBufferBarrier.buffer = ToonBuffer;
    toonBufferBarrier.offset = 0;
    toonBufferBarrier.size = ToonBufferSize;

    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &triangleBufferBarrier,
        0,
        nullptr
    );

    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &toonBufferBarrier,
        0,
        nullptr
    );

    VkBufferMemoryBarrier binMaskToWriteBarrier{};
    binMaskToWriteBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    binMaskToWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    binMaskToWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    binMaskToWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    binMaskToWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    binMaskToWriteBarrier.buffer = BinMaskBuffer;
    binMaskToWriteBarrier.offset = 0;
    binMaskToWriteBarrier.size = BinMaskBufferSize;

    VkBufferMemoryBarrier groupListToWriteBarrier{};
    groupListToWriteBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    groupListToWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    groupListToWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    groupListToWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    groupListToWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    groupListToWriteBarrier.buffer = GroupListBuffer;
    groupListToWriteBarrier.offset = 0;
    groupListToWriteBarrier.size = GroupListBufferSize;

    VkBufferMemoryBarrier spanSetupToWriteBarrier{};
    spanSetupToWriteBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    spanSetupToWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    spanSetupToWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    spanSetupToWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    spanSetupToWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    spanSetupToWriteBarrier.buffer = SpanSetupBuffer;
    spanSetupToWriteBarrier.offset = 0;
    spanSetupToWriteBarrier.size = SpanSetupBufferSize;

    VkBufferMemoryBarrier spanSetupToReadBarrier{};
    spanSetupToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    spanSetupToReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    spanSetupToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    spanSetupToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    spanSetupToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    spanSetupToReadBarrier.buffer = SpanSetupBuffer;
    spanSetupToReadBarrier.offset = 0;
    spanSetupToReadBarrier.size = SpanSetupBufferSize;

    VkBufferMemoryBarrier spanSetupHostToReadBarrier{};
    spanSetupHostToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    spanSetupHostToReadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    spanSetupHostToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    spanSetupHostToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    spanSetupHostToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    spanSetupHostToReadBarrier.buffer = SpanSetupBuffer;
    spanSetupHostToReadBarrier.offset = 0;
    spanSetupHostToReadBarrier.size = SpanSetupBufferSize;

    VkBufferMemoryBarrier resultToReadWriteBarrier{};
    resultToReadWriteBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    resultToReadWriteBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    resultToReadWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    resultToReadWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToReadWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToReadWriteBarrier.buffer = ResultBuffer;
    resultToReadWriteBarrier.offset = 0;
    resultToReadWriteBarrier.size = ResultBufferSize;

    VkBufferMemoryBarrier resultToWriteBarrier{};
    resultToWriteBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    resultToWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    resultToWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    resultToWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToWriteBarrier.buffer = ResultBuffer;
    resultToWriteBarrier.offset = 0;
    resultToWriteBarrier.size = ResultBufferSize;

    VkBufferMemoryBarrier resultToTransferWriteBarrier{};
    resultToTransferWriteBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    resultToTransferWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    resultToTransferWriteBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    resultToTransferWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToTransferWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToTransferWriteBarrier.buffer = ResultBuffer;
    resultToTransferWriteBarrier.offset = 0;
    resultToTransferWriteBarrier.size = ResultBufferSize;

    if (useLegacyRasterWorklist)
    {
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            1,
            &resultToTransferWriteBarrier,
            0,
            nullptr
        );

        const VkDeviceSize framebufferStrideBytes = static_cast<VkDeviceSize>(ColorImageWidth)
            * static_cast<VkDeviceSize>(ColorImageHeight)
            * sizeof(u32);
        const u32 clearColor6A5 = ((rgbaColor & 0xFFu) >> 2)
            | ((((rgbaColor >> 8u) & 0xFFu) >> 2) << 8u)
            | ((((rgbaColor >> 16u) & 0xFFu) >> 2) << 16u)
            | ((((rgbaColor >> 24u) & 0xFFu) >> 3) << 24u);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 0u * framebufferStrideBytes, framebufferStrideBytes, clearColor6A5);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 1u * framebufferStrideBytes, framebufferStrideBytes, clearColor6A5);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 2u * framebufferStrideBytes, framebufferStrideBytes, clearDepth);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 3u * framebufferStrideBytes, framebufferStrideBytes, clearDepth);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 4u * framebufferStrideBytes, framebufferStrideBytes, clearAttr);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 5u * framebufferStrideBytes, framebufferStrideBytes, clearAttr);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 6u * framebufferStrideBytes, framebufferStrideBytes, 0u);
        vkCmdFillBuffer(CommandBuffer, ResultBuffer, 7u * framebufferStrideBytes, framebufferStrideBytes, 0u);

        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            1,
            &resultToReadWriteBarrier,
            0,
            nullptr
        );
    }
    else
    {
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            1,
            &resultToWriteBarrier,
            0,
            nullptr
        );
    }

    vkCmdBindDescriptorSets(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, PipelineLayout, 0, 1, &DescriptorSet, 0, nullptr);
    constexpr u32 binTileSize = 8;
    const u32 binTilesX = (ColorImageWidth + (binTileSize - 1u)) / binTileSize;
    const u32 binTilesY = (ColorImageHeight + (binTileSize - 1u)) / binTileSize;
    const u32 tileCount = std::max<u32>(1u, binTilesX * binTilesY);

    VkBufferMemoryBarrier binMaskToReadBarrier{};
    binMaskToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    binMaskToReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    binMaskToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    binMaskToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    binMaskToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    binMaskToReadBarrier.buffer = BinMaskBuffer;
    binMaskToReadBarrier.offset = 0;
    binMaskToReadBarrier.size = BinMaskBufferSize;

    VkBufferMemoryBarrier binMaskHostToReadBarrier{};
    binMaskHostToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    binMaskHostToReadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    binMaskHostToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    binMaskHostToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    binMaskHostToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    binMaskHostToReadBarrier.buffer = BinMaskBuffer;
    binMaskHostToReadBarrier.offset = 0;
    binMaskHostToReadBarrier.size = BinMaskBufferSize;

    VkBufferMemoryBarrier groupListToReadBarrier{};
    groupListToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    groupListToReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    groupListToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    groupListToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    groupListToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    groupListToReadBarrier.buffer = GroupListBuffer;
    groupListToReadBarrier.offset = 0;
    groupListToReadBarrier.size = GroupListBufferSize;

    VkBufferMemoryBarrier groupListHostToReadBarrier{};
    groupListHostToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    groupListHostToReadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    groupListHostToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    groupListHostToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    groupListHostToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    groupListHostToReadBarrier.buffer = GroupListBuffer;
    groupListHostToReadBarrier.offset = 0;
    groupListHostToReadBarrier.size = GroupListBufferSize;

    VkBufferMemoryBarrier workOffsetToWriteBarrier{};
    workOffsetToWriteBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    workOffsetToWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    workOffsetToWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    workOffsetToWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    workOffsetToWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    workOffsetToWriteBarrier.buffer = WorkOffsetBuffer;
    workOffsetToWriteBarrier.offset = 0;
    workOffsetToWriteBarrier.size = WorkOffsetBufferSize;

    VkBufferMemoryBarrier workOffsetToReadBarrier{};
    workOffsetToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    workOffsetToReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    workOffsetToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    workOffsetToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    workOffsetToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    workOffsetToReadBarrier.buffer = WorkOffsetBuffer;
    workOffsetToReadBarrier.offset = 0;
    workOffsetToReadBarrier.size = WorkOffsetBufferSize;

    VkBufferMemoryBarrier workOffsetHostToReadBarrier{};
    workOffsetHostToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    workOffsetHostToReadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    workOffsetHostToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    workOffsetHostToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    workOffsetHostToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    workOffsetHostToReadBarrier.buffer = WorkOffsetBuffer;
    workOffsetHostToReadBarrier.offset = 0;
    workOffsetHostToReadBarrier.size = WorkOffsetBufferSize;

    constexpr VkDeviceSize kSortDispatchIndirectOffset = 0;
    constexpr VkDeviceSize kRasterDispatchIndirectOffset = sizeof(u32) * 3u;

    pushConstants.triangleBase = 0u;
    pushConstants.triangleCount = static_cast<u32>(Triangles.size());
    pushConstants.variantKey = kVariantWildcard;
    pushConstants.passIndex = 0u;

    const u32 binGroupCount = std::max<u32>(1u, (pushConstants.triangleCount + 31u) / 32u);
    const u32 interpGroupCount = std::max<u32>(1u, (pushConstants.triangleCount + 63u) / 64u);
    const u32 rasterWMode = resolveRasterWMode(kVariantWildcard);
    u32 rasterShadeMode = kRasterShadeModeAny;
    u32 rasterTextureMode = kRasterTextureModeAny;
    u32 rasterTranslucencyMode = kRasterTranslucencyModeAny;
    if (!Triangles.empty())
    {
        bool allTextured = true;
        bool allUntextured = true;
        bool allTranslucent = true;
        bool allOpaque = true;
        const u32 firstShadeMode = resolveRasterShadeMode(Triangles[0].variantKey);
        bool uniformShadeMode = firstShadeMode != kRasterShadeModeAny;

        for (const TriangleGpu& triangle : Triangles)
        {
            const bool isTextured = (triangle.variantKey & kVariantFlagTextured) != 0u;
            const bool isTranslucent = (triangle.variantKey & kVariantFlagTranslucent) != 0u;
            allTextured &= isTextured;
            allUntextured &= !isTextured;
            allTranslucent &= isTranslucent;
            allOpaque &= !isTranslucent;

            if (uniformShadeMode && resolveRasterShadeMode(triangle.variantKey) != firstShadeMode)
                uniformShadeMode = false;
        }

        if (allTextured)
            rasterTextureMode = kRasterTextureModeUseTexture;
        else if (allUntextured)
            rasterTextureMode = kRasterTextureModeNoTexture;

        if (allTranslucent)
            rasterTranslucencyMode = kRasterTranslucencyModeTranslucent;
        else if (allOpaque)
            rasterTranslucencyMode = kRasterTranslucencyModeOpaque;

        if (uniformShadeMode)
            rasterShadeMode = firstShadeMode;
    }
    if (rasterTextureMode != kRasterTextureModeAny)
        RasterSpecializedTextureModeCount++;
    if (rasterTranslucencyMode != kRasterTranslucencyModeAny)
        RasterSpecializedTranslucencyModeCount++;
    if (rasterShadeMode != kRasterShadeModeAny)
        RasterSpecializedShadeModeCount++;
    if (rasterTextureMode != kRasterTextureModeAny
        && rasterTranslucencyMode != kRasterTranslucencyModeAny
        && rasterShadeMode != kRasterShadeModeAny)
    {
        RasterSpecializedAllModesCount++;
    }
    const u32 rasterPipelineIndex = makeRasterPipelineIndex(
        rasterWMode,
        rasterShadeMode,
        rasterTextureMode,
        rasterTranslucencyMode
    );
    if (rasterPipelineIndex >= RasterPipelineVariantCount)
        return false;
    VkPipeline rasterPipeline = RasterPipelines[rasterPipelineIndex];
    if (rasterPipeline == VK_NULL_HANDLE)
        return false;
    if (!useCpuDirectTiles)
    {
        const u64 interpCpuStartNs = PerfNowNs();
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            1,
            &spanSetupToWriteBarrier,
            0,
            nullptr
        );
        vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, InterpPipeline);
        vkCmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(CommandBuffer, interpGroupCount, 1, 1);
        if (timestampQueryPool != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 1);
        InterpCpuWindow.Add(PerfNowNs() - interpCpuStartNs);
    }
    else
    {
        InterpCpuWindow.Add(0);
        if (timestampQueryPool != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 1);
    }

    u32 cpuActiveTileCountSample = 0;
    u32 cpuTileCountSample = 0;
    u32 cpuActiveGroupCountSample = 0;
    u32 cpuActiveDispatchSample = 0;

    const u64 binCpuStartNs = PerfNowNs();
    if (useCpuDirectTiles)
    {
        if (!prepareCpuTileBins(*context, pushConstants))
        {
            Log(LogLevel::Error, "VulkanRenderer3D: failed to prepare CPU tile bins");
            return false;
        }

        const u32* workOffsetValues = reinterpret_cast<const u32*>(context->WorkOffsetMapped);
        if (workOffsetValues != nullptr)
        {
            cpuActiveTileCountSample = workOffsetValues[kWorkActiveTileCount];
            cpuActiveGroupCountSample = workOffsetValues[kWorkActiveGroupCount];
        }
        cpuTileCountSample = tileCount;
        const bool hasSparseCoverage = static_cast<u64>(cpuActiveTileCountSample) * 100ull
            < static_cast<u64>(tileCount) * static_cast<u64>(kCpuActiveTileDispatchMaxCoveragePercent);
        useCpuActiveTileDispatch = cpuActiveTileCountSample == 0u || hasSparseCoverage;
        cpuActiveDispatchSample = useCpuActiveTileDispatch ? 100u : 0u;
        if (useCpuActiveTileDispatch)
            pushConstants.debugFlags |= kDebugFlagFinalActiveTileMask;

        BinCpuWindow.Add(PerfNowNs() - binCpuStartNs);
        WorkOffsetsCpuWindow.Add(0);
        SortCpuWindow.Add(0);
        CpuDirectTilesPathCount++;

        std::array<VkBufferMemoryBarrier, 4> rasterReadBarriers = {
            spanSetupHostToReadBarrier,
            binMaskHostToReadBarrier,
            groupListHostToReadBarrier,
            workOffsetHostToReadBarrier,
        };
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0,
            0,
            nullptr,
            static_cast<u32>(rasterReadBarriers.size()),
            rasterReadBarriers.data(),
            0,
            nullptr
        );
        if (timestampQueryPool != VK_NULL_HANDLE)
        {
            vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 2);
            vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 3);
            vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 4);
        }
    }
    else
    {
        if (!useLegacyRasterWorklist)
        {
            vkCmdFillBuffer(
                CommandBuffer,
                GroupListBuffer,
                0,
                static_cast<VkDeviceSize>(tileCount) * sizeof(u32),
                0
            );
        }
        std::array<VkBufferMemoryBarrier, 3> binWriteBarriers = {
            spanSetupToReadBarrier,
            binMaskToWriteBarrier,
            groupListToWriteBarrier,
        };
        const u32 binWriteBarrierCount = useLegacyRasterWorklist ? 2u : static_cast<u32>(binWriteBarriers.size());
        const VkPipelineStageFlags binWriteSrcStages = useLegacyRasterWorklist
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdPipelineBarrier(
            CommandBuffer,
            binWriteSrcStages,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            binWriteBarrierCount,
            binWriteBarriers.data(),
            0,
            nullptr
        );
        vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, BinPipeline);
        vkCmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(CommandBuffer, binGroupCount, binTilesX, binTilesY);
        if (timestampQueryPool != VK_NULL_HANDLE)
            vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 2);
        BinCpuWindow.Add(PerfNowNs() - binCpuStartNs);

        if (useLegacyRasterWorklist)
        {
            LegacyWorklistPathCount++;

            const u64 workOffsetsCpuStartNs = PerfNowNs();
            std::array<VkBufferMemoryBarrier, 2> workOffsetsBarriers = {
                binMaskToReadBarrier,
                workOffsetToWriteBarrier,
            };
            vkCmdPipelineBarrier(
                CommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                nullptr,
                static_cast<u32>(workOffsetsBarriers.size()),
                workOffsetsBarriers.data(),
                0,
                nullptr
            );
            vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, WorkOffsetsPipeline);
            vkCmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
            vkCmdDispatch(CommandBuffer, 1, 1, 1);
            if (timestampQueryPool != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 3);
            WorkOffsetsCpuWindow.Add(PerfNowNs() - workOffsetsCpuStartNs);

            const u64 sortCpuStartNs = PerfNowNs();
            vkCmdPipelineBarrier(
                CommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                0,
                0,
                nullptr,
                1,
                &workOffsetToReadBarrier,
                0,
                nullptr
            );
            vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, SortPipeline);
            vkCmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
            vkCmdDispatchIndirect(CommandBuffer, WorkOffsetBuffer, kSortDispatchIndirectOffset);
            if (timestampQueryPool != VK_NULL_HANDLE)
                vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 4);
            SortCpuWindow.Add(PerfNowNs() - sortCpuStartNs);

            std::array<VkBufferMemoryBarrier, 2> rasterReadBarriers = {
                binMaskToReadBarrier,
                workOffsetToReadBarrier,
            };
            vkCmdPipelineBarrier(
                CommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                0,
                0,
                nullptr,
                static_cast<u32>(rasterReadBarriers.size()),
                rasterReadBarriers.data(),
                0,
                nullptr
            );
        }
        else
        {
            DirectTilesPathCount++;
            WorkOffsetsCpuWindow.Add(0);
            SortCpuWindow.Add(0);
            std::array<VkBufferMemoryBarrier, 2> rasterReadBarriers = {
                binMaskToReadBarrier,
                groupListToReadBarrier,
            };
            vkCmdPipelineBarrier(
                CommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                nullptr,
                static_cast<u32>(rasterReadBarriers.size()),
                rasterReadBarriers.data(),
                0,
                nullptr
            );
            if (timestampQueryPool != VK_NULL_HANDLE)
            {
                vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 3);
                vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 4);
            }
        }
    }

    CpuActiveTileCountWindow.Add(cpuActiveTileCountSample);
    CpuTileCountWindow.Add(cpuTileCountSample);
    CpuActiveGroupCountWindow.Add(cpuActiveGroupCountSample);
    CpuActiveDispatchWindow.Add(cpuActiveDispatchSample);

    const u64 rasterCpuStartNs = PerfNowNs();
    vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, rasterPipeline);
    vkCmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    if (useLegacyRasterWorklist || useCpuActiveTileDispatch)
        vkCmdDispatchIndirect(CommandBuffer, WorkOffsetBuffer, kRasterDispatchIndirectOffset);
    else
        vkCmdDispatch(CommandBuffer, binTilesX, binTilesY, 1);
    if (timestampQueryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 5);
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &resultToReadWriteBarrier,
        0,
        nullptr
    );
    RasterCpuWindow.Add(PerfNowNs() - rasterCpuStartNs);

    pushConstants.variantKey = kVariantWildcard;
    pushConstants.passIndex = 0;
    pushConstants.triangleBase = 0;
    pushConstants.depthBlendMode = frameWBufferMode ? 1u : 0u;

    DepthBlendCpuWindow.Add(0);

    VkBufferMemoryBarrier resultToReadBarrier{};
    resultToReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    resultToReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    resultToReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    resultToReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultToReadBarrier.buffer = ResultBuffer;
    resultToReadBarrier.offset = 0;
    resultToReadBarrier.size = ResultBufferSize;

    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &resultToReadBarrier,
        0,
        nullptr
    );
    if (timestampQueryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 6);

    u32 finalPipelineIndex = 0;
    if ((dispCnt & (1u << 5u)) != 0u) // edge marking
        finalPipelineIndex |= 0x1u;
    if ((dispCnt & (1u << 7u)) != 0u) // fog
        finalPipelineIndex |= 0x2u;
    if ((dispCnt & (1u << 4u)) != 0u) // anti-aliasing
        finalPipelineIndex |= 0x4u;

    VkPipeline finalPipeline = FinalPipelines[finalPipelineIndex];
    if (finalPipeline == VK_NULL_HANDLE)
        return false;
    if (exportCaptureLine && CaptureLineExportPipeline == VK_NULL_HANDLE)
        return false;

    if (useCpuActiveTileDispatch)
    {
        VkImageMemoryBarrier imageToTransferWriteBarrier{};
        imageToTransferWriteBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageToTransferWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageToTransferWriteBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imageToTransferWriteBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageToTransferWriteBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageToTransferWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageToTransferWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageToTransferWriteBarrier.image = ColorImage;
        imageToTransferWriteBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageToTransferWriteBarrier.subresourceRange.baseMipLevel = 0;
        imageToTransferWriteBarrier.subresourceRange.levelCount = 1;
        imageToTransferWriteBarrier.subresourceRange.baseArrayLayer = 0;
        imageToTransferWriteBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &imageToTransferWriteBarrier
        );

        const u32 clearColor6A5 = ((rgbaColor & 0xFFu) >> 2u)
            | ((((rgbaColor >> 8u) & 0xFFu) >> 2u) << 8u)
            | ((((rgbaColor >> 16u) & 0xFFu) >> 2u) << 16u)
            | ((((rgbaColor >> 24u) & 0xFFu) >> 3u) << 24u);
        VkClearColorValue clearColorValue{};
        clearColorValue.float32[0] = static_cast<float>(clearColor6A5 & 0x3Fu) / 63.0f;
        clearColorValue.float32[1] = static_cast<float>((clearColor6A5 >> 8u) & 0x3Fu) / 63.0f;
        clearColorValue.float32[2] = static_cast<float>((clearColor6A5 >> 16u) & 0x3Fu) / 63.0f;
        clearColorValue.float32[3] = static_cast<float>((clearColor6A5 >> 24u) & 0x1Fu) / 31.0f;
        VkImageSubresourceRange clearRange{};
        clearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        clearRange.baseMipLevel = 0;
        clearRange.levelCount = 1;
        clearRange.baseArrayLayer = 0;
        clearRange.layerCount = 1;
        vkCmdClearColorImage(CommandBuffer, ColorImage, VK_IMAGE_LAYOUT_GENERAL, &clearColorValue, 1, &clearRange);

        VkImageMemoryBarrier imageToComputeWriteBarrier{};
        imageToComputeWriteBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageToComputeWriteBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imageToComputeWriteBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageToComputeWriteBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageToComputeWriteBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageToComputeWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageToComputeWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageToComputeWriteBarrier.image = ColorImage;
        imageToComputeWriteBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageToComputeWriteBarrier.subresourceRange.baseMipLevel = 0;
        imageToComputeWriteBarrier.subresourceRange.levelCount = 1;
        imageToComputeWriteBarrier.subresourceRange.baseArrayLayer = 0;
        imageToComputeWriteBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &imageToComputeWriteBarrier
        );

    }

    const u64 finalCpuStartNs = PerfNowNs();
    vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, finalPipeline);
    vkCmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    if (useCpuActiveTileDispatch)
        vkCmdDispatchIndirect(CommandBuffer, WorkOffsetBuffer, kRasterDispatchIndirectOffset);
    else
        vkCmdDispatch(CommandBuffer, binTilesX, binTilesY, 1);
    if (timestampQueryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampQueryPool, 7);
    FinalCpuWindow.Add(PerfNowNs() - finalCpuStartNs);

    if (exportCaptureLine)
    {
        const u64 captureLineExportCpuStartNs = PerfNowNs();
        VkImageMemoryBarrier colorToCaptureReadBarrier{};
        colorToCaptureReadBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        colorToCaptureReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        colorToCaptureReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        colorToCaptureReadBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorToCaptureReadBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorToCaptureReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorToCaptureReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        colorToCaptureReadBarrier.image = ColorImage;
        colorToCaptureReadBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorToCaptureReadBarrier.subresourceRange.baseMipLevel = 0;
        colorToCaptureReadBarrier.subresourceRange.levelCount = 1;
        colorToCaptureReadBarrier.subresourceRange.baseArrayLayer = 0;
        colorToCaptureReadBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &colorToCaptureReadBarrier
        );

        vkCmdBindPipeline(CommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, CaptureLineExportPipeline);
        vkCmdPushConstants(CommandBuffer, PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(CommandBuffer, 32u, 24u, 1u);

        VkBufferMemoryBarrier captureToHostReadBarrier{};
        captureToHostReadBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        captureToHostReadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        captureToHostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        captureToHostReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureToHostReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureToHostReadBarrier.buffer = CaptureLineBuffer;
        captureToHostReadBarrier.offset = 0;
        captureToHostReadBarrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0,
            nullptr,
            1,
            &captureToHostReadBarrier,
            0,
            nullptr
        );
        CaptureLineExportCpuWindow.Add(PerfNowNs() - captureLineExportCpuStartNs);
        CaptureLineExportCount++;
    }

    if (timestampQueryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(CommandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueryPool, 8);

    if (readbackToCpu)
    {
        VkImageMemoryBarrier toTransferSrcBarrier{};
        toTransferSrcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // Match the prepared snapshot path: capture readback must see the
        // fully resolved ColorImage, including transfer-based clears and any
        // earlier transfer usage on the same image.
        toTransferSrcBarrier.srcAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT;
        toTransferSrcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toTransferSrcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toTransferSrcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toTransferSrcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferSrcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransferSrcBarrier.image = ColorImage;
        toTransferSrcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toTransferSrcBarrier.subresourceRange.baseMipLevel = 0;
        toTransferSrcBarrier.subresourceRange.levelCount = 1;
        toTransferSrcBarrier.subresourceRange.baseArrayLayer = 0;
        toTransferSrcBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &toTransferSrcBarrier
        );

        if (useCaptureDownscaleForReadback)
        {
            VkImageMemoryBarrier captureToTransferDstBarrier{};
            captureToTransferDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            captureToTransferDstBarrier.srcAccessMask = CaptureReadbackImageInitialized ? VK_ACCESS_TRANSFER_READ_BIT : 0u;
            captureToTransferDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            captureToTransferDstBarrier.oldLayout = CaptureReadbackImageInitialized
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_UNDEFINED;
            captureToTransferDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            captureToTransferDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            captureToTransferDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            captureToTransferDstBarrier.image = CaptureReadbackImage;
            captureToTransferDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            captureToTransferDstBarrier.subresourceRange.baseMipLevel = 0;
            captureToTransferDstBarrier.subresourceRange.levelCount = 1;
            captureToTransferDstBarrier.subresourceRange.baseArrayLayer = 0;
            captureToTransferDstBarrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(
                CommandBuffer,
                CaptureReadbackImageInitialized ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &captureToTransferDstBarrier
            );

            VkImageBlit blitRegion{};
            blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.srcSubresource.mipLevel = 0;
            blitRegion.srcSubresource.baseArrayLayer = 0;
            blitRegion.srcSubresource.layerCount = 1;
            blitRegion.srcOffsets[0] = {0, 0, 0};
            blitRegion.srcOffsets[1] = {static_cast<int32_t>(ColorImageWidth), static_cast<int32_t>(ColorImageHeight), 1};
            blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.dstSubresource.mipLevel = 0;
            blitRegion.dstSubresource.baseArrayLayer = 0;
            blitRegion.dstSubresource.layerCount = 1;
            blitRegion.dstOffsets[0] = {0, 0, 0};
            blitRegion.dstOffsets[1] = {static_cast<int32_t>(readbackWidth), static_cast<int32_t>(readbackHeight), 1};
            vkCmdBlitImage(
                CommandBuffer,
                ColorImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                CaptureReadbackImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &blitRegion,
                VK_FILTER_NEAREST
            );

            VkImageMemoryBarrier captureToTransferSrcBarrier{};
            captureToTransferSrcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            captureToTransferSrcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            captureToTransferSrcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            captureToTransferSrcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            captureToTransferSrcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            captureToTransferSrcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            captureToTransferSrcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            captureToTransferSrcBarrier.image = CaptureReadbackImage;
            captureToTransferSrcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            captureToTransferSrcBarrier.subresourceRange.baseMipLevel = 0;
            captureToTransferSrcBarrier.subresourceRange.levelCount = 1;
            captureToTransferSrcBarrier.subresourceRange.baseArrayLayer = 0;
            captureToTransferSrcBarrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(
                CommandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &captureToTransferSrcBarrier
            );

            VkBufferImageCopy copyRegion{};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = 0;
            copyRegion.imageSubresource.baseArrayLayer = 0;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageOffset = {0, 0, 0};
            copyRegion.imageExtent.width = readbackWidth;
            copyRegion.imageExtent.height = readbackHeight;
            copyRegion.imageExtent.depth = 1;

            vkCmdCopyImageToBuffer(
                CommandBuffer,
                CaptureReadbackImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                ReadbackBuffer,
                1,
                &copyRegion
            );
        }
        else
        {
            VkBufferImageCopy copyRegion{};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = 0;
            copyRegion.imageSubresource.baseArrayLayer = 0;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageOffset = {0, 0, 0};
            copyRegion.imageExtent.width = ColorImageWidth;
            copyRegion.imageExtent.height = ColorImageHeight;
            copyRegion.imageExtent.depth = 1;

            vkCmdCopyImageToBuffer(
                CommandBuffer,
                ColorImage,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                ReadbackBuffer,
                1,
                &copyRegion
            );
        }

        VkBufferMemoryBarrier toHostBarrier{};
        toHostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        toHostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        toHostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHostBarrier.buffer = ReadbackBuffer;
        toHostBarrier.offset = 0;
        toHostBarrier.size = ReadbackSize;

        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0,
            nullptr,
            1,
            &toHostBarrier,
            0,
            nullptr
        );

        if (useCaptureDownscaleForReadback)
        {
            VkImageMemoryBarrier captureBackToGeneralBarrier{};
            captureBackToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            captureBackToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            captureBackToGeneralBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            captureBackToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            captureBackToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            captureBackToGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            captureBackToGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            captureBackToGeneralBarrier.image = CaptureReadbackImage;
            captureBackToGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            captureBackToGeneralBarrier.subresourceRange.baseMipLevel = 0;
            captureBackToGeneralBarrier.subresourceRange.levelCount = 1;
            captureBackToGeneralBarrier.subresourceRange.baseArrayLayer = 0;
            captureBackToGeneralBarrier.subresourceRange.layerCount = 1;

            vkCmdPipelineBarrier(
                CommandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                nullptr,
                0,
                nullptr,
                1,
                &captureBackToGeneralBarrier
            );
        }

        VkImageMemoryBarrier backToGeneralBarrier{};
        backToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        backToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        backToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        backToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        backToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        backToGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        backToGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        backToGeneralBarrier.image = ColorImage;
        backToGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        backToGeneralBarrier.subresourceRange.baseMipLevel = 0;
        backToGeneralBarrier.subresourceRange.levelCount = 1;
        backToGeneralBarrier.subresourceRange.baseArrayLayer = 0;
        backToGeneralBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &backToGeneralBarrier
        );
    }

    if (vkEndCommandBuffer(CommandBuffer) != VK_SUCCESS)
    {
        Log(LogLevel::Error, "VulkanRenderer3D: vkEndCommandBuffer failed");
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &CommandBuffer;

    {
        std::scoped_lock queueLock(VulkanContext::Get().GetQueueLock());
        const VkResult submitResult = vkQueueSubmit(Queue, 1, &submitInfo, FrameFence);
        if (submitResult != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: vkQueueSubmit failed (%d)", static_cast<int>(submitResult));
            return false;
        }
    }

    if (context != nullptr && Threaded)
        LastSubmittedRenderContext = context;

    if (timestampQueryPool != VK_NULL_HANDLE)
    {
        if (context != nullptr)
            context->TimestampPending = true;
        else
            TimestampPending = true;
    }

    const bool deferCaptureReadbackCompletion = readbackToCpu && captureReadbackPath && context != nullptr && Threaded;

    if ((readbackToCpu && !deferCaptureReadbackCompletion) || useSynchronousContext)
    {
        const u64 waitStartNs = PerfNowNs();
        if (vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: completion fence wait failed");
            return false;
        }

        FenceWaitCpuWindow.Add(PerfNowNs() - waitStartNs);
        consumeGpuTiming(context);
    }

    if (readbackToCpu && !deferCaptureReadbackCompletion)
    {
        if (ReadbackMapped == nullptr)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: readback buffer is not mapped");
            return false;
        }

        const size_t pixelCount = static_cast<size_t>(readbackWidth) * static_cast<size_t>(readbackHeight);
        if (RawReadbackRgba.size() != pixelCount)
            RawReadbackRgba.resize(pixelCount);
        std::memcpy(RawReadbackRgba.data(), ReadbackMapped, pixelCount * sizeof(u32));

        if (useCaptureDownscaleForReadback)
            CaptureReadbackImageInitialized = true;
    }
    else if (deferCaptureReadbackCompletion)
    {
        CaptureReadbackPending = true;
        PendingCaptureReadbackContext = context;
        if (useCaptureDownscaleForReadback)
            CaptureReadbackImageInitialized = true;
    }

    if (exportCaptureLine)
    {
        ReadyCaptureLineData = reinterpret_cast<const u32*>(captureLineMapped);
        if (useSynchronousContext)
        {
            CaptureLinePending = false;
            PendingCaptureLineContext = nullptr;
            CaptureLineReady = ReadyCaptureLineData != nullptr;
        }
        else
        {
            CaptureLinePending = true;
            PendingCaptureLineContext = context;
            CaptureLineReady = false;
        }
    }

    ColorImageInitialized = true;
    HasCpuFrame = readbackToCpu && !deferCaptureReadbackCompletion;
    return true;
}

bool VulkanRenderer3D::readbackColorTargetToCpu(bool capturePath)
{
    ReadbackColorRequestCount++;

    if (!ensureInitialized() || ColorImage == VK_NULL_HANDLE || ColorImageWidth == 0 || ColorImageHeight == 0)
        return false;

    if (capturePath && !waitForReadbackSource())
        return false;

    constexpr u32 kCaptureReadbackWidth = 256u;
    constexpr u32 kCaptureReadbackHeight = 192u;
    const bool canUseCaptureDownscale = capturePath
        && (ColorImageWidth != kCaptureReadbackWidth || ColorImageHeight != kCaptureReadbackHeight);
    const u32 readbackWidth = canUseCaptureDownscale ? kCaptureReadbackWidth : ColorImageWidth;
    const u32 readbackHeight = canUseCaptureDownscale ? kCaptureReadbackHeight : ColorImageHeight;
    const VkDeviceSize requiredReadbackSize = static_cast<VkDeviceSize>(readbackWidth) * static_cast<VkDeviceSize>(readbackHeight) * sizeof(u32);

    if (canUseCaptureDownscale && !ensureCaptureReadbackImage())
        return false;

    if (ReadbackBuffer == VK_NULL_HANDLE || ReadbackMemory == VK_NULL_HANDLE || ReadbackSize != requiredReadbackSize)
    {
        destroyReadbackBuffer();
        if (!createReadbackBuffer(readbackWidth, readbackHeight))
            return false;
    }

    if (vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return false;

    if (vkResetFences(Device, 1, &FrameFence) != VK_SUCCESS)
        return false;

    if (vkResetCommandBuffer(CommandBuffer, 0) != VK_SUCCESS)
        return false;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(CommandBuffer, &beginInfo) != VK_SUCCESS)
        return false;

    VkImageMemoryBarrier colorToTransferSrcBarrier{};
    colorToTransferSrcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    colorToTransferSrcBarrier.srcAccessMask =
        VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT;
    colorToTransferSrcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    colorToTransferSrcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorToTransferSrcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    colorToTransferSrcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorToTransferSrcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorToTransferSrcBarrier.image = ColorImage;
    colorToTransferSrcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorToTransferSrcBarrier.subresourceRange.baseMipLevel = 0;
    colorToTransferSrcBarrier.subresourceRange.levelCount = 1;
    colorToTransferSrcBarrier.subresourceRange.baseArrayLayer = 0;
    colorToTransferSrcBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &colorToTransferSrcBarrier
    );

    if (canUseCaptureDownscale)
    {
        VkImageMemoryBarrier captureToTransferDstBarrier{};
        captureToTransferDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        captureToTransferDstBarrier.srcAccessMask = CaptureReadbackImageInitialized ? VK_ACCESS_TRANSFER_READ_BIT : 0u;
        captureToTransferDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        captureToTransferDstBarrier.oldLayout = CaptureReadbackImageInitialized
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_UNDEFINED;
        captureToTransferDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        captureToTransferDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureToTransferDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureToTransferDstBarrier.image = CaptureReadbackImage;
        captureToTransferDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        captureToTransferDstBarrier.subresourceRange.baseMipLevel = 0;
        captureToTransferDstBarrier.subresourceRange.levelCount = 1;
        captureToTransferDstBarrier.subresourceRange.baseArrayLayer = 0;
        captureToTransferDstBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(
            CommandBuffer,
            CaptureReadbackImageInitialized ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &captureToTransferDstBarrier
        );

        VkImageBlit blitRegion{};
        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.srcSubresource.mipLevel = 0;
        blitRegion.srcSubresource.baseArrayLayer = 0;
        blitRegion.srcSubresource.layerCount = 1;
        blitRegion.srcOffsets[0] = {0, 0, 0};
        blitRegion.srcOffsets[1] = {static_cast<int32_t>(ColorImageWidth), static_cast<int32_t>(ColorImageHeight), 1};
        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.dstSubresource.mipLevel = 0;
        blitRegion.dstSubresource.baseArrayLayer = 0;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.dstOffsets[0] = {0, 0, 0};
        blitRegion.dstOffsets[1] = {static_cast<int32_t>(readbackWidth), static_cast<int32_t>(readbackHeight), 1};
        vkCmdBlitImage(
            CommandBuffer,
            ColorImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            CaptureReadbackImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blitRegion,
            VK_FILTER_NEAREST
        );

        VkImageMemoryBarrier captureToTransferSrcBarrier{};
        captureToTransferSrcBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        captureToTransferSrcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        captureToTransferSrcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        captureToTransferSrcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        captureToTransferSrcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        captureToTransferSrcBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureToTransferSrcBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureToTransferSrcBarrier.image = CaptureReadbackImage;
        captureToTransferSrcBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        captureToTransferSrcBarrier.subresourceRange.baseMipLevel = 0;
        captureToTransferSrcBarrier.subresourceRange.levelCount = 1;
        captureToTransferSrcBarrier.subresourceRange.baseArrayLayer = 0;
        captureToTransferSrcBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &captureToTransferSrcBarrier
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = {0, 0, 0};
        copyRegion.imageExtent.width = readbackWidth;
        copyRegion.imageExtent.height = readbackHeight;
        copyRegion.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(
            CommandBuffer,
            CaptureReadbackImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            ReadbackBuffer,
            1,
            &copyRegion
        );
    }
    else
    {
        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = {0, 0, 0};
        copyRegion.imageExtent.width = ColorImageWidth;
        copyRegion.imageExtent.height = ColorImageHeight;
        copyRegion.imageExtent.depth = 1;
        vkCmdCopyImageToBuffer(
            CommandBuffer,
            ColorImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            ReadbackBuffer,
            1,
            &copyRegion
        );
    }

    VkBufferMemoryBarrier toHostBarrier{};
    toHostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toHostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    toHostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHostBarrier.buffer = ReadbackBuffer;
    toHostBarrier.offset = 0;
    toHostBarrier.size = ReadbackSize;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0,
        nullptr,
        1,
        &toHostBarrier,
        0,
        nullptr
    );

    if (canUseCaptureDownscale)
    {
        VkImageMemoryBarrier captureBackToGeneralBarrier{};
        captureBackToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        captureBackToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        captureBackToGeneralBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        captureBackToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        captureBackToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        captureBackToGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureBackToGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        captureBackToGeneralBarrier.image = CaptureReadbackImage;
        captureBackToGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        captureBackToGeneralBarrier.subresourceRange.baseMipLevel = 0;
        captureBackToGeneralBarrier.subresourceRange.levelCount = 1;
        captureBackToGeneralBarrier.subresourceRange.baseArrayLayer = 0;
        captureBackToGeneralBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(
            CommandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &captureBackToGeneralBarrier
        );
    }

    VkImageMemoryBarrier colorBackToGeneralBarrier{};
    colorBackToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    colorBackToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    colorBackToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    colorBackToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    colorBackToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorBackToGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBackToGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorBackToGeneralBarrier.image = ColorImage;
    colorBackToGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorBackToGeneralBarrier.subresourceRange.baseMipLevel = 0;
    colorBackToGeneralBarrier.subresourceRange.levelCount = 1;
    colorBackToGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    colorBackToGeneralBarrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &colorBackToGeneralBarrier
    );

    if (vkEndCommandBuffer(CommandBuffer) != VK_SUCCESS)
        return false;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &CommandBuffer;
    {
        std::scoped_lock queueLock(VulkanContext::Get().GetQueueLock());
        const VkResult submitResult = vkQueueSubmit(Queue, 1, &submitInfo, FrameFence);
        if (submitResult != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: readback vkQueueSubmit failed (%d)", static_cast<int>(submitResult));
            return false;
        }
    }

    if (vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return false;

    if (ReadbackMapped == nullptr)
        return false;

    const size_t pixelCount = static_cast<size_t>(readbackWidth) * static_cast<size_t>(readbackHeight);
    if (RawReadbackRgba.size() != pixelCount)
        RawReadbackRgba.resize(pixelCount);
    std::memcpy(RawReadbackRgba.data(), ReadbackMapped, pixelCount * sizeof(u32));

    HasCpuFrame = true;
    ColorImageInitialized = true;
    if (canUseCaptureDownscale)
        CaptureReadbackImageInitialized = true;
    return true;
}

bool VulkanRenderer3D::readbackResultBufferToCpu()
{
    ReadbackResultRequestCount++;

    if (!ensureInitialized() || ResultBuffer == VK_NULL_HANDLE || ResultBufferSize == 0)
        return false;

    if (!waitForReadbackSource())
        return false;

    if (ResultReadbackBuffer == VK_NULL_HANDLE || ResultReadbackMemory == VK_NULL_HANDLE || ResultReadbackSize != ResultBufferSize)
    {
        destroyResultReadbackBuffer();
        if (!createResultReadbackBuffer())
            return false;
    }

    if (vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return false;

    if (vkResetFences(Device, 1, &FrameFence) != VK_SUCCESS)
        return false;

    if (vkResetCommandBuffer(CommandBuffer, 0) != VK_SUCCESS)
        return false;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(CommandBuffer, &beginInfo) != VK_SUCCESS)
        return false;

    VkBufferMemoryBarrier toTransferBarrier{};
    toTransferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toTransferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    toTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.buffer = ResultBuffer;
    toTransferBarrier.offset = 0;
    toTransferBarrier.size = ResultBufferSize;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        1,
        &toTransferBarrier,
        0,
        nullptr
    );

    VkBufferCopy copyRegion{};
    copyRegion.size = ResultBufferSize;
    vkCmdCopyBuffer(CommandBuffer, ResultBuffer, ResultReadbackBuffer, 1, &copyRegion);

    VkBufferMemoryBarrier toHostBarrier{};
    toHostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toHostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    toHostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHostBarrier.buffer = ResultReadbackBuffer;
    toHostBarrier.offset = 0;
    toHostBarrier.size = ResultReadbackSize;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0,
        nullptr,
        1,
        &toHostBarrier,
        0,
        nullptr
    );

    VkBufferMemoryBarrier backToComputeBarrier{};
    backToComputeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    backToComputeBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    backToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    backToComputeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToComputeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToComputeBarrier.buffer = ResultBuffer;
    backToComputeBarrier.offset = 0;
    backToComputeBarrier.size = ResultBufferSize;
    vkCmdPipelineBarrier(
        CommandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &backToComputeBarrier,
        0,
        nullptr
    );

    if (vkEndCommandBuffer(CommandBuffer) != VK_SUCCESS)
        return false;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &CommandBuffer;
    {
        std::scoped_lock queueLock(VulkanContext::Get().GetQueueLock());
        const VkResult submitResult = vkQueueSubmit(Queue, 1, &submitInfo, FrameFence);
        if (submitResult != VK_SUCCESS)
        {
            Log(LogLevel::Error, "VulkanRenderer3D: result readback vkQueueSubmit failed (%d)", static_cast<int>(submitResult));
            return false;
        }
    }

    if (vkWaitForFences(Device, 1, &FrameFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return false;

    if (ResultReadbackMapped == nullptr)
        return false;

    const size_t wordCount = static_cast<size_t>(ResultReadbackSize / sizeof(u32));
    if (RawResultReadback.size() != wordCount)
        RawResultReadback.resize(wordCount);
    std::memcpy(RawResultReadback.data(), ResultReadbackMapped, ResultReadbackSize);
    return true;
}

void VulkanRenderer3D::buildTriangleList(GPU& gpu)
{
    Triangles.clear();
    ActiveTextureDescriptorCount = 0;
    ActiveTextureDescriptors.fill(VkDescriptorImageInfo{});
    Triangles.reserve(static_cast<size_t>(gpu.GPU3D.RenderNumPolygons) * 3u);

    struct TextureFrameData
    {
        u32 DescriptorIndex = 0;
    };

    struct TextureLookupKey
    {
        TexcacheVulkanLoader::TextureHandle Handle = 0;

        bool operator==(const TextureLookupKey& other) const noexcept
        {
            return Handle == other.Handle;
        }
    };

    struct TextureLookupHasher
    {
        size_t operator()(const TextureLookupKey& key) const noexcept
        {
            return std::hash<u64>{}(key.Handle);
        }
    };

    std::unordered_map<TextureLookupKey, TextureFrameData, TextureLookupHasher> textureLookup{};
    textureLookup.reserve(static_cast<size_t>(gpu.GPU3D.RenderNumPolygons));

    const u32 targetHeight = 192u * static_cast<u32>(std::max(1, ScaleFactor));
    const float scale = static_cast<float>(std::max(1, ScaleFactor));
    const float maxTargetX = 256.0f * scale;
    const float maxTargetY = 192.0f * scale;
    const bool useHiresCoordinates = ScaleFactor > 1;
    const bool textureMapsEnabled = (gpu.GPU3D.RenderDispCnt & (1u << 0)) != 0;
    const bool disablePassiveRepeatCoverageExpand =
        (MelonDSAndroid::getVulkanDiagnosticFlags() & kVulkanDiagnosticDisablePassiveRepeatCoverageExpand) != 0u;
    const float coverageDepthBias = CoverageFixDepthBias * 16777215.0f;

    const auto resolveVertexX = [&](const Vertex* vertex) -> float {
        if (useHiresCoordinates)
        {
            const float hiresX = (static_cast<float>(vertex->HiresPosition[0]) * scale) * (1.0f / 16.0f);
            return std::clamp(hiresX, 0.0f, maxTargetX);
        }

        const int x = std::clamp(vertex->FinalPosition[0], 0, 256);
        return std::clamp(static_cast<float>(x) * scale, 0.0f, maxTargetX);
    };

    const auto resolveVertexY = [&](const Vertex* vertex) -> float {
        if (useHiresCoordinates)
        {
            const float hiresY = (static_cast<float>(vertex->HiresPosition[1]) * scale) * (1.0f / 16.0f);
            return std::clamp(hiresY, 0.0f, maxTargetY);
        }

        const int y = std::clamp(vertex->FinalPosition[1], 0, 192);
        return std::clamp(static_cast<float>(y) * scale, 0.0f, maxTargetY);
    };

    const auto to8From6 = [](u32 c6) -> u32 {
        c6 &= 0x3Fu;
        return (c6 << 2) | (c6 >> 4);
    };

    const auto to8From5 = [](u32 c5) -> u32 {
        c5 &= 0x1Fu;
        return (c5 << 3) | (c5 >> 2);
    };

    const auto packRgba8 = [](u32 r, u32 g, u32 b, u32 a) -> u32 {
        return (r & 0xFFu) | ((g & 0xFFu) << 8) | ((b & 0xFFu) << 16) | ((a & 0xFFu) << 24);
    };

    for (u32 i = 0; i < gpu.GPU3D.RenderNumPolygons; i++)
    {
        Polygon* polygon = gpu.GPU3D.RenderPolygonRAM[i];
        if (polygon == nullptr
            || polygon->Degenerate
            || (polygon->Type == 1 ? polygon->NumVertices < 2 : polygon->NumVertices < 3))
            continue;

        const u32 alpha5 = (polygon->Attr >> 16) & 0x1F;
        const bool isTranslucent = polygon->Translucent || (alpha5 != 0u && alpha5 < 0x1Fu);
        const u32 blendMode = (polygon->Attr >> 4) & 0x3u;
        const bool highlightEnabled = (gpu.GPU3D.RenderDispCnt & (1u << 1)) != 0;
        const bool wrapS = (polygon->TexParam & (1u << 16)) != 0;
        const bool wrapT = (polygon->TexParam & (1u << 17)) != 0;
        const bool isRepeat = wrapS || wrapT;
        const bool allowUserCoverageFix = CoverageFixEnabled && (CoverageFixPx > 0.0f) && (polygon->Type != 1);
        const bool applyPassiveCoverageFix =
            !disablePassiveRepeatCoverageExpand
            && (polygon->Type != 1)
            && isRepeat
            && (PassiveCoverageFixRepeatPx > 0.0f);
        bool applyUserCoverageFix = false;
        if (allowUserCoverageFix)
            applyUserCoverageFix = isRepeat ? CoverageFixApplyRepeat : CoverageFixApplyClamp;

        const float effectiveCoverageFixPx =
            (applyPassiveCoverageFix ? PassiveCoverageFixRepeatPx : 0.0f)
            + (applyUserCoverageFix ? CoverageFixPx : 0.0f);
        const float effectiveCoverageDepthBias =
            applyUserCoverageFix ? coverageDepthBias : 0.0f;
        const bool applyCoverageFix = effectiveCoverageFixPx > 0.0f;

        std::array<float, 10> expandedVertexX{};
        std::array<float, 10> expandedVertexY{};
        if (applyCoverageFix)
        {
            float centerX = 0.0f;
            float centerY = 0.0f;
            u32 centerVertexCount = 0;
            std::array<float, 10> baseVertexX{};
            std::array<float, 10> baseVertexY{};

            for (u32 vertexIndex = 0; vertexIndex < polygon->NumVertices; vertexIndex++)
            {
                const Vertex* vertex = polygon->Vertices[vertexIndex];
                if (vertex == nullptr)
                    continue;

                const float x = resolveVertexX(vertex);
                const float y = resolveVertexY(vertex);
                baseVertexX[vertexIndex] = x;
                baseVertexY[vertexIndex] = y;
                centerX += x;
                centerY += y;
                centerVertexCount++;
            }

            if (centerVertexCount > 0u)
            {
                const float vertexCount = static_cast<float>(centerVertexCount);
                centerX /= vertexCount;
                centerY /= vertexCount;
            }

            for (u32 vertexIndex = 0; vertexIndex < polygon->NumVertices; vertexIndex++)
            {
                if (polygon->Vertices[vertexIndex] == nullptr)
                    continue;

                const float baseX = baseVertexX[vertexIndex];
                const float baseY = baseVertexY[vertexIndex];

                float outX = baseX;
                float outY = baseY;

                const float dx = baseX - centerX;
                const float dy = baseY - centerY;
                const float lengthSquared = (dx * dx) + (dy * dy);
                if (lengthSquared > 0.000001f)
                {
                    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
                    outX = baseX + (dx * inverseLength * effectiveCoverageFixPx);
                    outY = baseY + (dy * inverseLength * effectiveCoverageFixPx);
                }

                expandedVertexX[vertexIndex] = std::clamp(outX, 0.0f, maxTargetX);
                expandedVertexY[vertexIndex] = std::clamp(outY, 0.0f, maxTargetY);
            }
        }

        bool polygonTextured = textureMapsEnabled && (((polygon->TexParam >> 26) & 0x7u) != 0u);
        TexcacheVulkanLoader::TextureHandle textureHandle = 0;
        u32 textureLayer = 0;
        u32* helper = nullptr;
        u32 textureDescriptorIndex = FallbackTextureDescriptorIndex;
        bool textureFallbackUsed = false;
        u32 texWidth = 0;
        u32 texHeight = 0;
        if (polygonTextured)
        {
            Texcache.GetTexture(
                gpu,
                polygon->TexParam,
                polygon->TexPalette,
                textureHandle,
                textureLayer,
                helper
            );

            texWidth = TextureWidth(polygon->TexParam);
            texHeight = TextureHeight(polygon->TexParam);
            if (texWidth == 0 || texHeight == 0)
            {
                polygonTextured = false;
            }
            else
            {
                const TextureLookupKey textureKey{
                    textureHandle,
                };
                auto textureIt = textureLookup.find(textureKey);
                if (textureIt == textureLookup.end())
                {
                    VkDescriptorImageInfo textureDescriptorInfo{};
                    if (Texcache.GetLoader().GetTextureDescriptor(textureHandle, &textureDescriptorInfo)
                        && ActiveTextureDescriptorCount < MaxActiveTextureDescriptors)
                    {
                        textureDescriptorIndex = ActiveTextureDescriptorCount;
                        ActiveTextureDescriptors[textureDescriptorIndex] = textureDescriptorInfo;
                        ActiveTextureDescriptorCount++;
                        textureLookup.emplace(
                            textureKey,
                            TextureFrameData{
                                textureDescriptorIndex,
                            });
                    }
                    else
                    {
                        textureDescriptorIndex = FallbackTextureDescriptorIndex;
                        textureLayer = 0;
                        texWidth = 1;
                        texHeight = 1;
                        textureFallbackUsed = true;
                    }
                }
                else
                {
                    textureDescriptorIndex = textureIt->second.DescriptorIndex;
                }
            }
        }

        const auto makeX = [&](const Vertex* vertex, u32 vertexIndex) -> float {
            if (applyCoverageFix && vertexIndex < polygon->NumVertices)
                return expandedVertexX[vertexIndex];

            return resolveVertexX(vertex);
        };

        const auto makeY = [&](const Vertex* vertex, u32 vertexIndex) -> float {
            if (applyCoverageFix && vertexIndex < polygon->NumVertices)
                return expandedVertexY[vertexIndex];

            return resolveVertexY(vertex);
        };

        const auto makeColor = [&](const Vertex* vertex) -> u32 {
            // FinalColor is produced in the same expanded range the software/OpenGL paths
            // consume. Quantizing it as if it were already 0..255 makes Vulkan too bright.
            const u32 vr = static_cast<u32>(std::clamp(vertex->FinalColor[0], 0, 511)) >> 3;
            const u32 vg = static_cast<u32>(std::clamp(vertex->FinalColor[1], 0, 511)) >> 3;
            const u32 vb = static_cast<u32>(std::clamp(vertex->FinalColor[2], 0, 511)) >> 3;
            const u32 polyAlpha = alpha5;

            return packRgba8(
                to8From6(vr),
                to8From6(vg),
                to8From6(vb),
                to8From5(std::min<u32>(31, polyAlpha))
            );
        };

        struct TriangleVertexData
        {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            float w = 1.0f;
            float u = 0.0f;
            float v = 0.0f;
            u32 colorRgba8 = 0;
            u32 wRaw = 1;
        };

        const auto makeTriangleVertex = [&](const Vertex* vertex, u32 vertexIndex, float x, float y) -> TriangleVertexData {
            TriangleVertexData triangleVertex{};
            triangleVertex.x = x;
            triangleVertex.y = y;
            triangleVertex.z = static_cast<float>(polygon->FinalZ[vertexIndex]);
            triangleVertex.wRaw = static_cast<u32>(std::max<s32>(1, polygon->FinalW[vertexIndex]));
            triangleVertex.w = static_cast<float>(triangleVertex.wRaw);
            if (applyCoverageFix && effectiveCoverageDepthBias > 0.0f)
                triangleVertex.z = std::max(0.0f, triangleVertex.z - effectiveCoverageDepthBias);

            triangleVertex.u = static_cast<float>(vertex->TexCoords[0]);
            triangleVertex.v = static_cast<float>(vertex->TexCoords[1]);
            triangleVertex.colorRgba8 = makeColor(vertex);
            return triangleVertex;
        };

        const auto makeCenterTriangleVertex = [&]() -> std::optional<TriangleVertexData> {
            if (!BetterPolygons || polygon->NumVertices <= 3u)
                return std::nullopt;

            u32 centerXFixed = 0u;
            u32 centerYFixed = 0u;
            float centerZ = 0.0f;
            float centerReciprocalW = 0.0f;
            float centerR = 0.0f;
            float centerG = 0.0f;
            float centerB = 0.0f;
            float centerU = 0.0f;
            float centerV = 0.0f;
            u32 validVertexCount = 0;

            for (u32 vertexIndex = 0; vertexIndex < polygon->NumVertices; vertexIndex++)
            {
                const Vertex* vertex = polygon->Vertices[vertexIndex];
                if (vertex == nullptr)
                    return std::nullopt;

                centerXFixed += static_cast<u32>(std::max<s32>(0, vertex->HiresPosition[0]));
                centerYFixed += static_cast<u32>(std::max<s32>(0, vertex->HiresPosition[1]));

                const float fw = static_cast<float>(std::max<s32>(1, polygon->FinalW[vertexIndex]))
                    * static_cast<float>(polygon->NumVertices);
                centerReciprocalW += 1.0f / fw;

                if (polygon->WBuffer)
                    centerZ += static_cast<float>(polygon->FinalZ[vertexIndex]) / fw;
                else
                    centerZ += static_cast<float>(polygon->FinalZ[vertexIndex]);

                // Keep the same expanded FinalColor range the software/OpenGL paths use,
                // then quantize once when emitting the synthesized center vertex.
                centerR += static_cast<float>(std::clamp(vertex->FinalColor[0], 0, 511)) / fw;
                centerG += static_cast<float>(std::clamp(vertex->FinalColor[1], 0, 511)) / fw;
                centerB += static_cast<float>(std::clamp(vertex->FinalColor[2], 0, 511)) / fw;
                centerU += static_cast<float>(vertex->TexCoords[0]) / fw;
                centerV += static_cast<float>(vertex->TexCoords[1]) / fw;
                validVertexCount++;
            }

            if (validVertexCount == 0u || centerReciprocalW <= 0.0f)
                return std::nullopt;

            TriangleVertexData centerVertex{};
            centerXFixed /= validVertexCount;
            centerYFixed /= validVertexCount;
            centerVertex.x = std::clamp((static_cast<float>(centerXFixed) * scale) * (1.0f / 16.0f), 0.0f, maxTargetX);
            centerVertex.y = std::clamp((static_cast<float>(centerYFixed) * scale) * (1.0f / 16.0f), 0.0f, maxTargetY);
            centerVertex.w = 1.0f / centerReciprocalW;
            centerVertex.wRaw = std::max<u32>(1u, static_cast<u32>(centerVertex.w));

            if (polygon->WBuffer)
                centerVertex.z = centerZ * centerVertex.w;
            else
                centerVertex.z = centerZ / static_cast<float>(validVertexCount);

            if (applyCoverageFix && effectiveCoverageDepthBias > 0.0f)
                centerVertex.z = std::max(0.0f, centerVertex.z - effectiveCoverageDepthBias);

            centerR *= centerVertex.w;
            centerG *= centerVertex.w;
            centerB *= centerVertex.w;
            centerVertex.u = static_cast<float>(static_cast<s32>(centerU * centerVertex.w));
            centerVertex.v = static_cast<float>(static_cast<s32>(centerV * centerVertex.w));
            const u32 centerR6 = static_cast<u32>(std::clamp(static_cast<int>(centerR), 0, 511)) >> 3;
            const u32 centerG6 = static_cast<u32>(std::clamp(static_cast<int>(centerG), 0, 511)) >> 3;
            const u32 centerB6 = static_cast<u32>(std::clamp(static_cast<int>(centerB), 0, 511)) >> 3;
            centerVertex.colorRgba8 = packRgba8(
                to8From6(std::min<u32>(63u, centerR6)),
                to8From6(std::min<u32>(63u, centerG6)),
                to8From6(std::min<u32>(63u, centerB6)),
                to8From5(std::min<u32>(31u, alpha5)));
            return centerVertex;
        };

        constexpr u32 kTriangleFlagTranslucent = 1u << 0u;
        constexpr u32 kTriangleFlagTextured = 1u << 1u;
        constexpr u32 kTriangleFlagDecal = 1u << 2u;
        constexpr u32 kTriangleFlagCoverageFix = 1u << 3u;
        constexpr u32 kTriangleFlagWBuffer = 1u << 4u;
        constexpr u32 kTriangleFlagShadowMask = 1u << 5u;
        constexpr u32 kTriangleFlagLinear = 1u << 6u;
        constexpr u32 kTriangleFlagBoundaryEdge0 = 1u << 7u;
        constexpr u32 kTriangleFlagBoundaryEdge1 = 1u << 8u;
        constexpr u32 kTriangleFlagBoundaryEdge2 = 1u << 9u;
        constexpr u32 kTriangleFlagFrontFacing = 1u << 10u;
        constexpr u32 kTriangleFlagTopLeftEdge0 = 1u << 11u;
        constexpr u32 kTriangleFlagTopLeftEdge1 = 1u << 12u;
        constexpr u32 kTriangleFlagTopLeftEdge2 = 1u << 13u;
        constexpr u32 kVariantFlagTextured = 1u << 0u;
        constexpr u32 kVariantFlagDecal = 1u << 1u;
        constexpr u32 kVariantFlagModulate = 1u << 2u;
        constexpr u32 kVariantFlagToon = 1u << 3u;
        constexpr u32 kVariantFlagHighlight = 1u << 4u;
        constexpr u32 kVariantFlagShadowMask = 1u << 5u;
        constexpr u32 kVariantFlagWBuffer = 1u << 6u;
        constexpr u32 kVariantFlagTranslucent = 1u << 7u;
        constexpr u32 kVariantFlagCoverageFix = 1u << 8u;

        auto packYBounds = [&](const float* yValues, size_t yValueCount) -> std::optional<u32> {
            u32 polygonYTop = targetHeight;
            u32 polygonYBot = 0;
            bool hasPolygonYBounds = false;
            for (size_t yIndex = 0; yIndex < yValueCount; yIndex++)
            {
                const float clampedY = std::clamp(yValues[yIndex], 0.0f, static_cast<float>(targetHeight));
                const u32 yTopLine = static_cast<u32>(std::floor(clampedY));
                const u32 yBottomLine = std::min<u32>(targetHeight, static_cast<u32>(std::ceil(clampedY)));
                polygonYTop = std::min(polygonYTop, yTopLine);
                polygonYBot = std::max(polygonYBot, yBottomLine);
                hasPolygonYBounds = true;
            }

            if (!hasPolygonYBounds)
                return std::nullopt;

            if (polygonYBot <= polygonYTop)
                polygonYBot = std::min<u32>(targetHeight, polygonYTop + 1u);

            return (polygonYTop & 0xFFFFu) | ((polygonYBot & 0xFFFFu) << 16u);
        };

        auto packPolygonYBounds = [&]() -> std::optional<u32> {
            if (!useHiresCoordinates)
            {
                const s32 rawTop = std::clamp(polygon->YTop, 0, static_cast<s32>(targetHeight));
                const s32 rawBottom = std::clamp(polygon->YBottom, 0, static_cast<s32>(targetHeight));
                u32 polygonYTop = static_cast<u32>(rawTop);
                u32 polygonYBot = static_cast<u32>(rawBottom);
                if (polygonYBot <= polygonYTop)
                    polygonYBot = std::min<u32>(targetHeight, polygonYTop + 1u);
                return (polygonYTop & 0xFFFFu) | ((polygonYBot & 0xFFFFu) << 16u);
            }

            return std::nullopt;
        };

        const bool hasTexture = polygonTextured && texWidth > 0 && texHeight > 0;
        const auto appendTriangle = [&](
            const TriangleVertexData& vertex0,
            const TriangleVertexData& vertex1,
            const TriangleVertexData& vertex2,
            u32 boundaryFlags,
            u32 packedYBounds)
        {
            TriangleGpu triangle{};
            triangle.x0 = vertex0.x;
            triangle.y0 = vertex0.y;
            triangle.z0 = vertex0.z;
            triangle.w0 = vertex0.w;
            triangle.x1 = vertex1.x;
            triangle.y1 = vertex1.y;
            triangle.z1 = vertex1.z;
            triangle.w1 = vertex1.w;
            triangle.x2 = vertex2.x;
            triangle.y2 = vertex2.y;
            triangle.z2 = vertex2.z;
            triangle.w2 = vertex2.w;

            triangle.u0 = vertex0.u;
            triangle.v0 = vertex0.v;
            triangle.u1 = vertex1.u;
            triangle.v1 = vertex1.v;
            triangle.u2 = vertex2.u;
            triangle.v2 = vertex2.v;
            triangle.yBounds = packedYBounds;
            triangle.texLayer = textureLayer;

            triangle.color0Rgba8 = vertex0.colorRgba8;
            triangle.color1Rgba8 = vertex1.colorRgba8;
            triangle.color2Rgba8 = vertex2.colorRgba8;
            const u32 a0 = (triangle.color0Rgba8 >> 24) & 0xFFu;
            const u32 a1 = (triangle.color1Rgba8 >> 24) & 0xFFu;
            const u32 a2 = (triangle.color2Rgba8 >> 24) & 0xFFu;
            const bool alphaTranslucent = (a0 < 255u) || (a1 < 255u) || (a2 < 255u);

            triangle.flags = boundaryFlags;
            if (isTranslucent || alphaTranslucent)
                triangle.flags |= kTriangleFlagTranslucent;
            if (hasTexture)
            {
                triangle.flags |= kTriangleFlagTextured;
                if ((blendMode & 0x1u) != 0u && !textureFallbackUsed)
                    triangle.flags |= kTriangleFlagDecal;
                triangle.texArrayIndex = textureDescriptorIndex;
                triangle.texWidth = texWidth;
                triangle.texHeight = texHeight;
                triangle.texParam = polygon->TexParam;
            }
            if (applyCoverageFix)
                triangle.flags |= kTriangleFlagCoverageFix;
            if (polygon->WBuffer)
                triangle.flags |= kTriangleFlagWBuffer;
            if (polygon->IsShadowMask)
                triangle.flags |= kTriangleFlagShadowMask;
            if (vertex0.wRaw == vertex1.wRaw && vertex1.wRaw == vertex2.wRaw && (vertex0.wRaw & 0x7Fu) == 0u)
                triangle.flags |= kTriangleFlagLinear;
            if (polygon->FacingView)
                triangle.flags |= kTriangleFlagFrontFacing;

            auto isTopLeftEdge = [](const TriangleVertexData& start, const TriangleVertexData& end) -> bool {
                const float deltaY = end.y - start.y;
                if (std::fabs(deltaY) < 0.000001f)
                    return (end.x - start.x) > 0.0f;
                return deltaY < 0.0f;
            };
            const float signedArea = (vertex2.x - vertex0.x) * (vertex1.y - vertex0.y)
                - (vertex2.y - vertex0.y) * (vertex1.x - vertex0.x);
            const bool positiveArea = signedArea > 0.0f;
            if ((triangle.flags & kTriangleFlagBoundaryEdge0) == 0u
                && (positiveArea ? isTopLeftEdge(vertex1, vertex2) : isTopLeftEdge(vertex2, vertex1)))
            {
                triangle.flags |= kTriangleFlagTopLeftEdge0;
            }
            if ((triangle.flags & kTriangleFlagBoundaryEdge1) == 0u
                && (positiveArea ? isTopLeftEdge(vertex2, vertex0) : isTopLeftEdge(vertex0, vertex2)))
            {
                triangle.flags |= kTriangleFlagTopLeftEdge1;
            }
            if ((triangle.flags & kTriangleFlagBoundaryEdge2) == 0u
                && (positiveArea ? isTopLeftEdge(vertex0, vertex1) : isTopLeftEdge(vertex1, vertex0)))
            {
                triangle.flags |= kTriangleFlagTopLeftEdge2;
            }

            triangle.polyAttr = polygon->Attr;
            triangle.variantKey = 0;
            if (hasTexture)
                triangle.variantKey |= kVariantFlagTextured;

            if (blendMode == 2u)
            {
                triangle.variantKey |= highlightEnabled ? kVariantFlagHighlight : kVariantFlagToon;
            }
            else if (hasTexture && (blendMode & 0x1u) != 0u && !textureFallbackUsed)
            {
                triangle.variantKey |= kVariantFlagDecal;
            }
            else
            {
                triangle.variantKey |= kVariantFlagModulate;
            }
            if (polygon->IsShadowMask)
                triangle.variantKey |= kVariantFlagShadowMask;
            if (polygon->WBuffer)
                triangle.variantKey |= kVariantFlagWBuffer;
            if (isTranslucent || alphaTranslucent)
                triangle.variantKey |= kVariantFlagTranslucent;
            if (applyCoverageFix)
                triangle.variantKey |= kVariantFlagCoverageFix;

            Triangles.push_back(triangle);
        };

        if (polygon->Type == 1)
        {
            const Vertex* lineVertices[2]{};
            u32 lineVertexIndices[2]{};
            u32 lineVertexCount = 0;
            s32 lastLineX = 0;
            s32 lastLineY = 0;
            bool haveLastLineVertex = false;
            for (u32 vertexIndex = 0; vertexIndex < polygon->NumVertices && lineVertexCount < 2u; vertexIndex++)
            {
                const Vertex* vertex = polygon->Vertices[vertexIndex];
                if (vertex == nullptr)
                    continue;

                if (haveLastLineVertex
                    && lastLineX == vertex->FinalPosition[0]
                    && lastLineY == vertex->FinalPosition[1])
                {
                    continue;
                }

                lineVertices[lineVertexCount] = vertex;
                lineVertexIndices[lineVertexCount] = vertexIndex;
                lineVertexCount++;
                lastLineX = vertex->FinalPosition[0];
                lastLineY = vertex->FinalPosition[1];
                haveLastLineVertex = true;
            }

            if (lineVertexCount < 2u)
                continue;

            const float lineX0 = makeX(lineVertices[0], lineVertexIndices[0]);
            const float lineY0 = makeY(lineVertices[0], lineVertexIndices[0]);
            const float lineX1 = makeX(lineVertices[1], lineVertexIndices[1]);
            const float lineY1 = makeY(lineVertices[1], lineVertexIndices[1]);

            const float deltaX = lineX1 - lineX0;
            const float deltaY = lineY1 - lineY0;
            const float lineLengthSquared = (deltaX * deltaX) + (deltaY * deltaY);
            if (lineLengthSquared <= 0.000001f)
                continue;

            const float inverseLineLength = 1.0f / std::sqrt(lineLengthSquared);
            const float halfLineWidth = 0.5f;
            const float perpX = -deltaY * inverseLineLength * halfLineWidth;
            const float perpY = deltaX * inverseLineLength * halfLineWidth;

            const float quadPositionsX[4] = {
                std::clamp(lineX0 + perpX, 0.0f, maxTargetX),
                std::clamp(lineX0 - perpX, 0.0f, maxTargetX),
                std::clamp(lineX1 - perpX, 0.0f, maxTargetX),
                std::clamp(lineX1 + perpX, 0.0f, maxTargetX),
            };
            const float quadPositionsY[4] = {
                std::clamp(lineY0 + perpY, 0.0f, maxTargetY),
                std::clamp(lineY0 - perpY, 0.0f, maxTargetY),
                std::clamp(lineY1 - perpY, 0.0f, maxTargetY),
                std::clamp(lineY1 + perpY, 0.0f, maxTargetY),
            };

            const std::optional<u32> packedLineYBounds = packYBounds(quadPositionsY, 4);
            if (!packedLineYBounds.has_value())
                continue;

            appendTriangle(
                makeTriangleVertex(lineVertices[0], lineVertexIndices[0], quadPositionsX[0], quadPositionsY[0]),
                makeTriangleVertex(lineVertices[0], lineVertexIndices[0], quadPositionsX[1], quadPositionsY[1]),
                makeTriangleVertex(lineVertices[1], lineVertexIndices[1], quadPositionsX[2], quadPositionsY[2]),
                kTriangleFlagBoundaryEdge0 | kTriangleFlagBoundaryEdge2,
                *packedLineYBounds);
            appendTriangle(
                makeTriangleVertex(lineVertices[0], lineVertexIndices[0], quadPositionsX[0], quadPositionsY[0]),
                makeTriangleVertex(lineVertices[1], lineVertexIndices[1], quadPositionsX[2], quadPositionsY[2]),
                makeTriangleVertex(lineVertices[1], lineVertexIndices[1], quadPositionsX[3], quadPositionsY[3]),
                kTriangleFlagBoundaryEdge0 | kTriangleFlagBoundaryEdge1,
                *packedLineYBounds);
            continue;
        }

        const std::optional<u32> packedPolygonYBounds = packPolygonYBounds();
        u32 polygonYTop = targetHeight;
        u32 polygonYBot = 0;
        bool hasPolygonYBounds = false;
        if (packedPolygonYBounds.has_value())
        {
            polygonYTop = *packedPolygonYBounds & 0xFFFFu;
            polygonYBot = (*packedPolygonYBounds >> 16u) & 0xFFFFu;
            hasPolygonYBounds = true;
        }
        else
        {
            for (u32 vertexIndex = 0; vertexIndex < polygon->NumVertices; vertexIndex++)
            {
                const Vertex* vertex = polygon->Vertices[vertexIndex];
                if (vertex == nullptr)
                    continue;

                const float y = makeY(vertex, vertexIndex);
                const float clampedY = std::clamp(y, 0.0f, static_cast<float>(targetHeight));
                const u32 yTopLine = static_cast<u32>(std::floor(clampedY));
                const u32 yBottomLine = std::min<u32>(targetHeight, static_cast<u32>(std::ceil(clampedY)));
                polygonYTop = std::min(polygonYTop, yTopLine);
                polygonYBot = std::max(polygonYBot, yBottomLine);
                hasPolygonYBounds = true;
            }
            if (!hasPolygonYBounds)
                continue;
            if (polygonYBot <= polygonYTop)
                polygonYBot = std::min<u32>(targetHeight, polygonYTop + 1u);
        }
        const u32 packedYBounds = (polygonYTop & 0xFFFFu) | ((polygonYBot & 0xFFFFu) << 16u);

        const std::optional<TriangleVertexData> centerVertex = makeCenterTriangleVertex();
        if (centerVertex.has_value())
        {
            u32 firstOuterVertexIndex = polygon->NumVertices;
            u32 previousOuterVertexIndex = polygon->NumVertices;

            for (u32 vertexIndex = 0; vertexIndex < polygon->NumVertices; vertexIndex++)
            {
                Vertex* outerVertex = polygon->Vertices[vertexIndex];
                if (outerVertex == nullptr)
                {
                    firstOuterVertexIndex = polygon->NumVertices;
                    break;
                }

                if (firstOuterVertexIndex == polygon->NumVertices)
                {
                    firstOuterVertexIndex = vertexIndex;
                    previousOuterVertexIndex = vertexIndex;
                    continue;
                }

                Vertex* previousOuterVertex = polygon->Vertices[previousOuterVertexIndex];
                appendTriangle(
                    *centerVertex,
                    makeTriangleVertex(
                        previousOuterVertex,
                        previousOuterVertexIndex,
                        makeX(previousOuterVertex, previousOuterVertexIndex),
                        makeY(previousOuterVertex, previousOuterVertexIndex)),
                    makeTriangleVertex(
                        outerVertex,
                        vertexIndex,
                        makeX(outerVertex, vertexIndex),
                        makeY(outerVertex, vertexIndex)),
                    kTriangleFlagBoundaryEdge0,
                    packedYBounds);
                previousOuterVertexIndex = vertexIndex;
            }

            if (firstOuterVertexIndex < polygon->NumVertices
                && previousOuterVertexIndex < polygon->NumVertices
                && previousOuterVertexIndex != firstOuterVertexIndex)
            {
                Vertex* lastOuterVertex = polygon->Vertices[previousOuterVertexIndex];
                Vertex* firstOuterVertex = polygon->Vertices[firstOuterVertexIndex];
                appendTriangle(
                    *centerVertex,
                    makeTriangleVertex(
                        lastOuterVertex,
                        previousOuterVertexIndex,
                        makeX(lastOuterVertex, previousOuterVertexIndex),
                        makeY(lastOuterVertex, previousOuterVertexIndex)),
                    makeTriangleVertex(
                        firstOuterVertex,
                        firstOuterVertexIndex,
                        makeX(firstOuterVertex, firstOuterVertexIndex),
                        makeY(firstOuterVertex, firstOuterVertexIndex)),
                    kTriangleFlagBoundaryEdge0,
                    packedYBounds);
                continue;
            }
        }

        for (u32 vertexIdx = 1; vertexIdx + 1 < polygon->NumVertices; vertexIdx++)
        {
            Vertex* v0 = polygon->Vertices[0];
            Vertex* v1 = polygon->Vertices[vertexIdx];
            Vertex* v2 = polygon->Vertices[vertexIdx + 1];
            if (v0 == nullptr || v1 == nullptr || v2 == nullptr)
                continue;

            u32 boundaryFlags = kTriangleFlagBoundaryEdge0;
            if (vertexIdx + 1 == polygon->NumVertices - 1)
                boundaryFlags |= kTriangleFlagBoundaryEdge1;
            if (vertexIdx == 1)
                boundaryFlags |= kTriangleFlagBoundaryEdge2;

            appendTriangle(
                makeTriangleVertex(v0, 0, makeX(v0, 0), makeY(v0, 0)),
                makeTriangleVertex(v1, vertexIdx, makeX(v1, vertexIdx), makeY(v1, vertexIdx)),
                makeTriangleVertex(v2, vertexIdx + 1, makeX(v2, vertexIdx + 1), makeY(v2, vertexIdx + 1)),
                boundaryFlags,
                packedYBounds);
        }
    }
}

void VulkanRenderer3D::convertReadbackToLineCache()
{
    if (!HasCpuFrame || RawReadbackWidth == 0 || RawReadbackHeight == 0 || RawReadbackRgba.empty())
    {
        clearLineCache();
        return;
    }

    const bool readbackIsNativeDs = RawReadbackWidth == 256u && RawReadbackHeight == 192u;

    if (readbackIsNativeDs)
    {
        const size_t pixelCount = 256u * 192u;
        for (size_t i = 0; i < pixelCount; i++)
        {
            const u32 sourcePixel = RawReadbackRgba[i];

            const u32 r = sourcePixel & 0xFF;
            const u32 g = (sourcePixel >> 8) & 0xFF;
            const u32 b = (sourcePixel >> 16) & 0xFF;
            const u32 a = (sourcePixel >> 24) & 0xFF;

            LineCache[i] =
                (r >> 2)
                | ((g >> 2) << 8)
                | ((b >> 2) << 16)
                | ((a >> 3) << 24);
        }

        return;
    }

    const u32 sampleScale = static_cast<u32>(std::max(1, ScaleFactor));
    for (u32 y = 0; y < 192; y++)
    {
        const u32 sourceY = std::min(RawReadbackHeight - 1, y * sampleScale);
        for (u32 x = 0; x < 256; x++)
        {
            const u32 sourceX = std::min(RawReadbackWidth - 1, x * sampleScale);
            const u32 sourcePixel = RawReadbackRgba[static_cast<size_t>(sourceY) * static_cast<size_t>(RawReadbackWidth) + sourceX];

            const u32 r = sourcePixel & 0xFF;
            const u32 g = (sourcePixel >> 8) & 0xFF;
            const u32 b = (sourcePixel >> 16) & 0xFF;
            const u32 a = (sourcePixel >> 24) & 0xFF;

            LineCache[static_cast<size_t>(y) * 256u + x] =
                (r >> 2)
                | ((g >> 2) << 8)
                | ((b >> 2) << 16)
                | ((a >> 3) << 24);
        }
    }
}

u32 VulkanRenderer3D::buildClearColorRgba8(const GPU& gpu) const
{
    const u32 clearAttr1 = gpu.GPU3D.RenderClearAttr1;

    u32 r = (clearAttr1 << 1) & 0x3E;
    if (r)
        r++;

    u32 g = (clearAttr1 >> 4) & 0x3E;
    if (g)
        g++;

    u32 b = (clearAttr1 >> 9) & 0x3E;
    if (b)
        b++;

    const u32 a = (clearAttr1 >> 16) & 0x1F;

    const u32 r8 = (r << 2) | (r >> 4);
    const u32 g8 = (g << 2) | (g >> 4);
    const u32 b8 = (b << 2) | (b >> 4);
    const u32 a8 = (a << 3) | (a >> 2);

    return r8 | (g8 << 8) | (b8 << 16) | (a8 << 24);
}

void VulkanRenderer3D::clearLineCache()
{
    std::fill(LineCache.begin(), LineCache.end(), 0);
}

void VulkanRenderer3D::WarmTextureCache(GPU& gpu)
{
    const bool enableTextureMaps = gpu.GPU3D.RenderDispCnt & (1 << 0);
    if (!enableTextureMaps)
        return;

    TexcacheVulkanLoader::TextureHandle textureHandle = 0;
    u32 textureLayer = 0;
    u32* helper = nullptr;

    for (u32 i = 0; i < gpu.GPU3D.RenderNumPolygons; i++)
    {
        Polygon* polygon = gpu.GPU3D.RenderPolygonRAM[i];
        if (polygon == nullptr)
            continue;

        if (((polygon->TexParam >> 26) & 0x7) == 0)
            continue;

        Texcache.GetTexture(
            gpu,
            polygon->TexParam,
            polygon->TexPalette,
            textureHandle,
            textureLayer,
            helper
        );
    }
}

}
