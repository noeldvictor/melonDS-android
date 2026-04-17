#include <ctime>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <filesystem>
#include <GLES3/gl3.h>
#include "Args.h"
#include "GPU3D_Compute.h"
#include "GPU2D_Soft.h"
#include "Configuration.h"
#include "DSi.h"
#include "DSiSupport.h"
#include "DSi_I2C.h"
#include "GPU3D_OpenGL.h"
#include "GPU3D_Soft.h"
#include "GPU3D_Vulkan.h"
#include "MelonDS.h"
#include "MelonInstance.h"
#include "NDS.h"
#include "NDSCart.h"
#include "VulkanContext.h"
#include "net/Net_Slirp.h"
#include "Platform.h"
#include "SDCardArgsBuilder.h"

using namespace std;
using namespace melonDS;
using namespace melonDS::Platform;

namespace MelonDSAndroid
{

const int kRewindBufferSize = 1024 * 1024 * 20; // Use 20MB per savestate
const int kRewindScreenshotSize = 256 * 384 * 4;
const int kScreenshotScreenWidth = 256;
const int kScreenshotScreenHeight = 192;
const int kCompositedScreenGapPx = 2;
const int kVulkanFastForwardHighResolutionScaleCap = 4;
const int kVulkanCompileStageInitRenderer = 1;
const int kVulkanCompileStageBuildPipelines = 2;
const int kVulkanCompileStageInitOutput = 3;
const int kVulkanCompileStageWarmupSubmission = 4;
const u64 kVulkanHighResolutionRealtimePresenterBudgetFloorNs = 4'000'000ull;

u32 expandPackedColor6ToRgba8(u32 packedColor)
{
    const u32 r6 = packedColor & 0xFFu;
    const u32 g6 = (packedColor >> 8u) & 0xFFu;
    const u32 b6 = (packedColor >> 16u) & 0xFFu;
    const u32 r8 = ((r6 & 0x3Fu) << 2u) | ((r6 & 0x3Fu) >> 4u);
    const u32 g8 = ((g6 & 0x3Fu) << 2u) | ((g6 & 0x3Fu) >> 4u);
    const u32 b8 = ((b6 & 0x3Fu) << 2u) | ((b6 & 0x3Fu) >> 4u);
    return r8 | (g8 << 8u) | (b8 << 16u) | 0xFF000000u;
}

class ScopedDebugOpenGlContext
{
public:
    ScopedDebugOpenGlContext()
    {
        if (!ensureOpenGlContext() || openGlContext == nullptr)
            return;

        Active = openGlContext->Use();
    }

    ~ScopedDebugOpenGlContext()
    {
        if (Active && openGlContext != nullptr)
            openGlContext->Release();
    }

    [[nodiscard]] bool IsReady() const noexcept
    {
        return Active;
    }

private:
    bool Active = false;
};

int getConfiguredVulkanScale(const VulkanRenderSettings& renderSettings)
{
    return std::max(1, renderSettings.scale);
}

int getEffectiveVulkanRenderScale(const VulkanRenderSettings& renderSettings)
{
    const int configuredScale = getConfiguredVulkanScale(renderSettings);
    if (!isFastForwardActive() || configuredScale <= kVulkanFastForwardHighResolutionScaleCap)
        return configuredScale;

    return kVulkanFastForwardHighResolutionScaleCap;
}

FrameQueuePolicy makeLegacyFrameQueuePolicy()
{
    FrameQueuePolicy policy{};
    policy.MaxBacklogDepth = FRAME_QUEUE_SIZE - 1;
    policy.AllowStealPending = true;
    policy.AllowPreviousFrameReuse = true;
    policy.AllowDropForDeadline = false;
    return policy;
}

FrameQueuePolicy makeVulkanRealtimeFrameQueuePolicy(int renderScale)
{
    FrameQueuePolicy policy{};
    policy.MaxBacklogDepth = renderScale > 4 ? 5 : (renderScale > 1 ? 2 : 1);
    policy.AllowStealPending = false;
    policy.AllowPreviousFrameReuse = true;
    policy.AllowDropForDeadline = false;
    policy.PreferOldestFrame = false;
    policy.PreserveBacklogOnPresent = false;
    return policy;
}

FrameQueuePolicy makeVulkanLateRealtimeFrameQueuePolicy(int renderScale)
{
    FrameQueuePolicy policy{};
    policy.MaxBacklogDepth = renderScale > 4 ? 5 : (renderScale > 1 ? 2 : 1);
    policy.AllowStealPending = false;
    policy.AllowPreviousFrameReuse = true;
    policy.AllowDropForDeadline = true;
    policy.PreferOldestFrame = false;
    policy.PreserveBacklogOnPresent = false;
    return policy;
}

FrameQueuePolicy makeVulkanFastForwardFrameQueuePolicy(int renderScale)
{
    FrameQueuePolicy policy{};
    const bool highResolutionFastForward = renderScale > 1;
    policy.MaxBacklogDepth = highResolutionFastForward ? 4 : 2;
    policy.AllowStealPending = true;
    policy.AllowPreviousFrameReuse = highResolutionFastForward;
    policy.AllowDropForDeadline = true;
    policy.PreferOldestFrame = false;
    policy.PreserveBacklogOnPresent = false;
    return policy;
}

bool isPresentationDeadlineExpired(const std::optional<std::chrono::time_point<std::chrono::steady_clock>>& deadline)
{
    return deadline.has_value() && std::chrono::steady_clock::now() >= *deadline;
}

FrameQueuePolicy makeFrameQueuePolicy(Renderer renderer, int vulkanRenderScale = 1)
{
    if (renderer == Renderer::Vulkan)
        return isFastForwardActive()
            ? makeVulkanFastForwardFrameQueuePolicy(std::max(vulkanRenderScale, 1))
            : makeVulkanRealtimeFrameQueuePolicy(std::max(vulkanRenderScale, 1));
    return makeLegacyFrameQueuePolicy();
}

void prepareRenderFrame(Frame* renderFrame)
{
    if (renderFrame == nullptr)
        return;

    EGLDisplay currentDisplay = eglGetCurrentDisplay();
    if (renderFrame->renderFence)
    {
        if (currentDisplay != EGL_NO_DISPLAY)
            eglDestroySyncKHR(currentDisplay, renderFrame->renderFence);
        renderFrame->renderFence = 0;
    }

    if (renderFrame->presentFence)
    {
        if (currentDisplay != EGL_NO_DISPLAY)
        {
            eglWaitSyncKHR(currentDisplay, renderFrame->presentFence, 0);
            eglDestroySyncKHR(currentDisplay, renderFrame->presentFence);
        }
        renderFrame->presentFence = 0;
    }
}

bool CopyCompositedFrameToScreenshot(
    const u32* sourcePixels,
    int sourceWidth,
    int sourceHeight,
    int scale,
    u32* destinationPixels,
    size_t destinationPixelCount
)
{
    if (sourcePixels == nullptr || destinationPixels == nullptr)
        return false;

    if (scale < 1)
        return false;

    const size_t requiredDestinationPixels = static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight) * 2;
    if (destinationPixelCount < requiredDestinationPixels)
        return false;

    if (sourceWidth < kScreenshotScreenWidth * scale)
        return false;

    const int bottomYOffset = (kScreenshotScreenHeight + kCompositedScreenGapPx) * scale;
    if (sourceHeight < bottomYOffset + (kScreenshotScreenHeight * scale))
        return false;

    for (int y = 0; y < kScreenshotScreenHeight; y++)
    {
        const u32* sourceTopLine = sourcePixels + static_cast<size_t>(y * scale) * static_cast<size_t>(sourceWidth);
        const u32* sourceBottomLine = sourcePixels + static_cast<size_t>(bottomYOffset + (y * scale)) * static_cast<size_t>(sourceWidth);
        u32* destinationTopLine = destinationPixels + static_cast<size_t>(y) * static_cast<size_t>(kScreenshotScreenWidth);
        u32* destinationBottomLine = destinationPixels
            + static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight)
            + static_cast<size_t>(y) * static_cast<size_t>(kScreenshotScreenWidth);

        for (int x = 0; x < kScreenshotScreenWidth; x++)
        {
            destinationTopLine[x] = sourceTopLine[x * scale];
            destinationBottomLine[x] = sourceBottomLine[x * scale];
        }
    }

    return true;
}

MelonInstance::MelonInstance(int instanceId, std::shared_ptr<EmulatorConfiguration> configuration, std::unique_ptr<melonDS::NDSArgs> args, std::shared_ptr<Net> net, ScreenshotRenderer screenshotRenderer, int consoleType) :
    instanceId(instanceId),
    currentConfiguration(configuration),
    net(net),
    lastCompletedVulkanFrame(nullptr),
    lastCompletedVulkanScale(1),
    screenshotRenderer(screenshotRenderer),
    consoleType(consoleType),
    rewindManager(configuration->rewindEnabled, configuration->rewindLengthSeconds, configuration->rewindCaptureSpacingSeconds, kRewindBufferSize, kRewindScreenshotSize),
    vulkanRuntimeConfigLogged(false),
    vulkanRuntimeFailureHandled(false),
    vulkanPrepareFailureCount(0)
{
    // Software renderer is always used during initialisation. Actual renderer will be set of first frame run
    currentRenderer = Renderer::Software;
    isRenderConfigurationDirty = true;
    inputMask = 0xFFF;
    frame = 0;

    net->RegisterInstance(instanceId);

    if (consoleType == 1)
    {
        melonDS::DSiArgs &dsiArgs = static_cast<melonDS::DSiArgs &>(*args);
        nds = new DSi(std::move(dsiArgs), this);
    }
    else
    {
        nds = new NDS(std::move(*args), this);
    }

    if (configuration->userInternalFirmwareAndBios)
    {
        std::filesystem::path firmwarePath = MelonDSAndroid::internalFilesDir;
        firmwarePath /= "wfcsettings.bin";
        firmwareSave = std::make_unique<SaveManager>(firmwarePath);
    }
    else
    {
        std::string firmwarePathString;
        if (consoleType == 1)
            firmwarePathString = configuration->dsiFirmwarePath;
        else
            firmwarePathString = configuration->dsFirmwarePath;

        firmwareSave = std::make_unique<SaveManager>(firmwarePathString);
    }

    // All instances have a RetroAchievements manager, but only the first instance will actually load achievements
    retroAchievementsManager = std::make_unique<RetroAchievements::RetroAchievementsManager>(nds);

    nds->Reset();
    setBatteryLevels();
    setDateTime();
}

MelonInstance::~MelonInstance()
{
    vulkanOutput = nullptr;
    net->UnregisterInstance(instanceId);
    delete nds;
}

bool MelonInstance::loadRom(std::string romPath, std::string sramPath)
{
    unique_ptr<u8[]> romData;
    unique_ptr<u8[]> sramData;
    u32 romFileLength = 0;
    u32 sramFileLength = 0;

    // ROM file loading
    Platform::FileHandle* romFile = Platform::OpenFile(romPath, FileMode::Read);
    if (!romFile)
        return false;

    u64 length = Platform::FileLength(romFile);
    if (length > 0x40000000)
    {
        Platform::CloseFile(romFile);
        return false;
    }

    romFileLength = (u32) length;
    Platform::FileRewind(romFile);
    romData = make_unique<u8[]>(romFileLength);
    size_t nread = Platform::FileRead(romData.get(), (size_t) romFileLength, 1, romFile);
    Platform::CloseFile(romFile);
    if (nread != 1)
    {
        return false;
    }

    // SRAM file loading
    FileHandle* sramFile = Platform::OpenFile(sramPath, FileMode::Read);
    if (!sramFile)
    {
        return false;
    }
    else if (!Platform::CheckFileWritable(sramPath))
    {
        return false;
    }

    sramFileLength = (u32) Platform::FileLength(sramFile);

    FileRewind(sramFile);
    sramData = std::make_unique<u8[]>(sramFileLength);
    FileRead(sramData.get(), sramFileLength, 1, sramFile);
    CloseFile(sramFile);

    NDSCart::NDSCartArgs cartargs{
        // Don't load the SD card itself yet, because we don't know if
        // the ROM is homebrew or not.
        // So this is the card we *would* load if the ROM were homebrew.
        .SDCard = std::nullopt, // getSDCardArgs("DLDI"), // TODO: Re-enable this
        .SRAM = std::move(sramData),
        .SRAMLength = sramFileLength,
    };

    auto cart = NDSCart::ParseROM(std::move(romData), romFileLength, this, std::move(cartargs));
    if (!cart)
    {
        return false;
    }

    nds->SetNDSCart(std::move(cart));
    ndsSave = std::make_unique<SaveManager>(sramPath);

    return true;
}

bool MelonInstance::loadGbaRom(std::string romPath, std::string sramPath)
{
    unique_ptr<u8[]> romData;
    unique_ptr<u8[]> sramData = nullptr;
    u32 romFileLength = 0;
    u32 sramFileLength = 0;

    // ROM file loading
    Platform::FileHandle* romFile = Platform::OpenFile(romPath, FileMode::Read);
    if (!romFile)
        return false;

    u64 length = Platform::FileLength(romFile);
    if (length > 0x40000000)
    {
        Platform::CloseFile(romFile);
        return false;
    }

    romFileLength = length;
    Platform::FileRewind(romFile);
    romData = make_unique<u8[]>(romFileLength);
    size_t nread = Platform::FileRead(romData.get(), (size_t) romFileLength, 1, romFile);
    Platform::CloseFile(romFile);
    if (nread != 1)
    {
        return false;
    }

    FileHandle* saveFile = Platform::OpenFile(sramPath, FileMode::Read);
    if (!saveFile)
    {
        return false;
    }
    else if (!Platform::CheckFileWritable(sramPath))
    {
        return false;
    }

    sramFileLength = (u32) FileLength(saveFile);

    if (sramFileLength > 0)
    {
        FileRewind(saveFile);
        sramData = std::make_unique<u8[]>(sramFileLength);
        FileRead(sramData.get(), sramFileLength, 1, saveFile);
    }
    CloseFile(saveFile);

    auto cart = GBACart::ParseROM(std::move(romData), romFileLength, std::move(sramData), sramFileLength, this);
    if (!cart)
    {
        return false;
    }

    nds->SetGBACart(std::move(cart));
    gbaSave = std::make_unique<SaveManager>(sramPath);

    return true;
}

void MelonInstance::loadRumblePak()
{
    auto rumblePakCart = GBACart::LoadAddon(GBAAddon_RumblePak, this);
    nds->SetGBACart(std::move(rumblePakCart));
}

void MelonInstance::loadGbaMemoryExpansion()
{
    auto memoryExpansionCart = GBACart::LoadAddon(GBAAddon_RAMExpansion, this);
    nds->SetGBACart(std::move(memoryExpansionCart));
}

void MelonInstance::loadGbaAnalogInput()
{
    auto analogInputCart = GBACart::LoadAddon(GBAAddon_Analog, this);
    nds->SetGBACart(std::move(analogInputCart));
}

void MelonInstance::loadGbaRumblePak()
{
    auto rumbleCart = GBACart::LoadAddon(GBAAddon_RumblePak, this);
    nds->SetGBACart(std::move(rumbleCart));
}

bool MelonInstance::bootFirmware()
{
    if (nds->NeedsDirectBoot())
        return false;

    return true;
}

bool MelonInstance::precompileVulkanPipelines()
{
    if (currentConfiguration->renderer != Renderer::Vulkan)
        return true;

    constexpr int kTotalCompileStages = 4;
    auto emitProgress = [&](int stageId, int current) {
        if (eventMessenger != nullptr)
            eventMessenger->onVulkanCompileProgress(stageId, current, kTotalCompileStages);
    };

    auto failPrecompile = [&](const char* reason) -> bool {
        Platform::Log(
            Platform::LogLevel::Error,
            "Vulkan precompile failed (%s)",
            reason != nullptr ? reason : "unknown"
        );
        if (eventMessenger != nullptr)
            eventMessenger->onRendererInitFailed(Renderer::Vulkan);
        return false;
    };

    emitProgress(kVulkanCompileStageInitRenderer, 0);
    if (isRenderConfigurationDirty || currentRenderer != Renderer::Vulkan)
    {
        updateRenderer();
        isRenderConfigurationDirty = false;
    }

    if (currentRenderer != Renderer::Vulkan)
        return failPrecompile("renderer switch");
    if (!vulkanOutput)
        return failPrecompile("missing output");

    auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
    auto vulkanRenderSettings = static_cast<VulkanRenderSettings&>(*currentConfiguration->renderSettings);
    const int vulkanScale = std::max(1, vulkanRenderSettings.scale);
    const u32 validationWidth = static_cast<u32>(256 * vulkanScale);
    const u32 validationHeight = static_cast<u32>((192 + 1) * 2 * vulkanScale);

    emitProgress(kVulkanCompileStageBuildPipelines, 1);
    if (!renderer3D.EnsureVulkanReadyForValidation())
        return failPrecompile("renderer pipelines");

    emitProgress(kVulkanCompileStageInitOutput, 2);
    if (!vulkanOutput->isInitialized() && !vulkanOutput->init())
        return failPrecompile("output init");

    emitProgress(kVulkanCompileStageWarmupSubmission, 3);
    if (!vulkanOutput->validateRuntimePath(validationWidth, validationHeight, renderer3D, vulkanScale))
        return failPrecompile("output warm-up");

    emitProgress(kVulkanCompileStageWarmupSubmission, 4);
    return true;
}

void MelonInstance::start()
{
    auto cart = nds->NDSCartSlot.GetCart();
    if (nds->ConsoleType == 1 && cart != nullptr && cart->GetHeader().IsDSiWare() && !currentConfiguration->showBootScreen)
    {
        auto dsi = (DSi*) nds;
        DSiSupport::SetupDSiDirectBoot(dsi);
    }
    else if (!currentConfiguration->showBootScreen || nds->NeedsDirectBoot())
    {
        // This seems to be unused, but it's required
        std::string romName;
        nds->SetupDirectBoot(romName);
    }
    nds->Start();

    vulkanRuntimeFailureHandled = false;
    vulkanPrepareFailureCount = 0;
    if (currentConfiguration->renderer != Renderer::Vulkan)
        screenshotRenderer.init();
}

void MelonInstance::reset()
{
    nds->Reset();
    setBatteryLevels();
    setDateTime();

    // If there is a cart inserted, check if direct boot is required
    if (nds->GetNDSCart())
    {
        if (!currentConfiguration->showBootScreen || nds->NeedsDirectBoot())
        {
            // This seems to be unused, but it's required
            std::string romName;
            nds->SetupDirectBoot(romName);
        }
    }

    rewindManager.Reset();
    retroAchievementsManager->Reset();
    nds->Start();
    vulkanRuntimeFailureHandled = false;
    vulkanPrepareFailureCount = 0;
}

u32 MelonInstance::runFrame()
{
    const bool measuringVulkan = currentConfiguration->renderer == Renderer::Vulkan;
    const u64 runFrameStartNs = measuringVulkan ? PerfNowNs() : 0;
    u64 ndsRunStartNs = 0;
    u64 ndsRunEndNs = 0;

    if (isRenderConfigurationDirty)
    {
        updateRenderer();
        isRenderConfigurationDirty = false;
    }

    if (currentRenderer == Renderer::Vulkan)
        updateVulkanFastForwardRenderScale();

    if (!nds->IsRunning())
        return 0;

    nds->GBACartSlot.SetInput(GBACart::Input_AnalogX, slot2AnalogX.load(std::memory_order_relaxed));
    nds->GBACartSlot.SetInput(GBACart::Input_AnalogY, slot2AnalogY.load(std::memory_order_relaxed));

    int screenWidth;
    int screenHeight;
    int vulkanRenderScale = 1;
    if (currentRenderer == Renderer::OpenGl)
    {
        int scale = static_cast<GLRenderer &>(nds->GPU.GetRenderer3D()).GetScaleFactor();
        screenWidth = 256 * scale;
        screenHeight = (192 + 1) * scale;
    }
    else if (currentRenderer == Renderer::Compute)
    {
        auto computeRenderSettings = static_cast<ComputeRenderSettings&>(*currentConfiguration->renderSettings);
        int scale = computeRenderSettings.scale;
        screenWidth = 256 * scale;
        screenHeight = (192 + 1) * scale;
    }
    else if (currentRenderer == Renderer::Vulkan)
    {
        auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
        vulkanRenderScale = std::max(renderer3D.GetScaleFactor(), 1);
        screenWidth = 256 * vulkanRenderScale;
        screenHeight = (192 + 1) * vulkanRenderScale;
    }
    else
    {
        screenWidth = 256;
        screenHeight = 192 + 1;
    }

    const FrameBackend frameBackend = (currentRenderer == Renderer::Vulkan) ? FrameBackend::VulkanImage : FrameBackend::OpenGlTexture;
    const FrameQueuePolicy frameQueuePolicy = makeFrameQueuePolicy(currentRenderer, vulkanRenderScale);
    Frame* renderFrame = nullptr;

    renderFrame = frameQueue.getRenderFrame(frameQueuePolicy);
    prepareRenderFrame(renderFrame);
    if (renderFrame != nullptr)
        frameQueue.validateRenderFrame(renderFrame, screenWidth, screenHeight * 2, frameBackend);

    if (currentRenderer == Renderer::Vulkan)
    {
        if (renderFrame != nullptr)
            renderFrame->renderTimelineValue = 0;

        if (renderFrame != nullptr && vulkanOutput != nullptr)
        {
            auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
            if (vulkanOutput->ensureFrameResources(renderFrame, screenWidth, screenHeight * 2))
                (void)vulkanOutput->captureRenderer3dSnapshot(renderFrame, renderer3D);
        }
    }

    [[unlikely]] if (nds->GPU.GetRenderer3D().NeedsShaderCompile())
    {
        // Compile all required shaders at once
        do
        {
            int currentShader;
            int shadersCount;
            nds->GPU.GetRenderer3D().ShaderCompileStep(currentShader, shadersCount);
        }
        while (nds->GPU.GetRenderer3D().NeedsShaderCompile());
    }

    bool isRendererAccelerated = nds->GPU.GetRenderer3D().Accelerated;
    if (isRendererAccelerated && frameBackend == FrameBackend::OpenGlTexture && renderFrame != nullptr)
    {
        int backBuffer = nds->GPU.FrontBuffer ? 0 : 1;
        nds->GPU.GetRenderer3D().SetOutputTexture(backBuffer, renderFrame->frameTexture);
    }

    if (measuringVulkan)
    {
        ndsRunStartNs = PerfNowNs();
        vulkanSetupCpuWindow.Add(ndsRunStartNs - runFrameStartNs);
    }

    u32 nLines = nds->RunFrame();
    if (measuringVulkan)
    {
        ndsRunEndNs = PerfNowNs();
        vulkanNdsRunCpuWindow.Add(ndsRunEndNs - ndsRunStartNs);
    }
    retroAchievementsManager->FrameUpdate();

    bool hasValidFrame = false;
    int frontbuf = nds->GPU.FrontBuffer;
    if (currentRenderer == Renderer::Vulkan)
    {
        if (vulkanOutput)
        {
            const auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
            const u64 composeStartNs = PerfNowNs();
            const bool isFrameUploaded = renderFrame != nullptr
                && vulkanOutput->prepareFrameForPresentation(renderFrame, nds->GPU, frontbuf, renderer3D);
            vulkanComposeCpuWindow.Add(PerfNowNs() - composeStartNs);
            if (renderFrame != nullptr && !isFrameUploaded)
            {
                vulkanPrepareFailureCount++;
                Platform::Log(
                    Platform::LogLevel::Warn,
                    "VulkanOutput: prepare/present failed, requesting resync (%d/4)",
                    vulkanPrepareFailureCount
                );
                requestVulkanPresentationResync();
                if (vulkanPrepareFailureCount >= 4)
                    handleVulkanRuntimeFailure("prepare/present");
            }
            else if (isFrameUploaded)
            {
                vulkanPrepareFailureCount = 0;
            }
            hasValidFrame = isFrameUploaded;
        }
        else
        {
            handleVulkanRuntimeFailure("missing VulkanOutput");
        }
    }
    else if (!isRendererAccelerated)
    {
        if (nds->GPU.Framebuffer[frontbuf][0] && nds->GPU.Framebuffer[frontbuf][1])
        {
            glBindTexture(GL_TEXTURE_2D, renderFrame->frameTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, nds->GPU.Framebuffer[frontbuf][0].get());
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 192 + 2, 256, 192, GL_RGBA, GL_UNSIGNED_BYTE, nds->GPU.Framebuffer[frontbuf][1].get());
            glBindTexture(GL_TEXTURE_2D, 0);
            hasValidFrame = true;
        }
    }
    else
    {
        // Do nothing. Emulator already renders into the texture, which was set-up above
        hasValidFrame = true;
    }

    if (currentRenderer == Renderer::Vulkan)
    {
        if (hasValidFrame)
        {
            lastCompletedVulkanFrame = renderFrame;
            lastCompletedVulkanScale = std::max(vulkanRenderScale, 1);
        }
    }
    else
    {
        lastCompletedVulkanFrame = nullptr;
        lastCompletedVulkanScale = 1;
        screenshotRenderer.renderScreenshot(&nds->GPU, currentRenderer, renderFrame);
    }

    const int nextFrame = frame + 1;
    const bool shouldCaptureRewindState = rewindManager.ShouldCaptureState(nextFrame);
    if (currentRenderer == Renderer::Vulkan && shouldCaptureRewindState)
        (void)updateVulkanScreenshot(hasValidFrame ? renderFrame : lastCompletedVulkanFrame, hasValidFrame ? std::max(vulkanRenderScale, 1) : lastCompletedVulkanScale, true);

    bool isSleeping = nds->CPUStop & CPUStop_Sleep;

    if (!isSleeping && hasValidFrame) [[likely]]
    {
        EGLDisplay currentDisplay = eglGetCurrentDisplay();
        if (frameBackend == FrameBackend::OpenGlTexture)
        {
            renderFrame->renderFence = eglCreateSyncKHR(currentDisplay, EGL_SYNC_FENCE_KHR, nullptr);
            glFlush();
        }
        else
        {
            renderFrame->renderFence = 0;
        }
        frameQueue.pushRenderedFrame(renderFrame, frameQueuePolicy);
    }
    else if (renderFrame != nullptr)
    {
        frameQueue.discardRenderedFrame(renderFrame);
    }

    if (ndsSave)
        ndsSave->CheckFlush();

    if (gbaSave)
        gbaSave->CheckFlush();

    if (firmwareSave)
        firmwareSave->CheckFlush();

    frame = nextFrame;
    if (shouldCaptureRewindState)
    {
        auto nextRewindState = rewindManager.GetNextRewindSaveState(frame);
        saveRewindState(nextRewindState);
    }

    if (currentRenderer == Renderer::Vulkan)
    {
        const u64 runFrameEndNs = PerfNowNs();
        if (ndsRunEndNs > 0 && runFrameEndNs >= ndsRunEndNs)
            vulkanPostRunCpuWindow.Add(runFrameEndNs - ndsRunEndNs);
        vulkanRunFrameCpuWindow.Add(runFrameEndNs - runFrameStartNs);
        logVulkanPerformanceIfNeeded();
    }

    return nLines;
}

void MelonInstance::handleVulkanRuntimeFailure(const char* reason)
{
    if (vulkanRuntimeFailureHandled)
        return;

    vulkanRuntimeFailureHandled = true;

    Platform::Log(
        Platform::LogLevel::Error,
        "Vulkan renderer runtime failure (%s)",
        reason != nullptr ? reason : "unknown"
    );

    if (eventMessenger)
        eventMessenger->onRendererInitFailed(Renderer::Vulkan);

    nds->Stop(Platform::StopReason::BadExceptionRegion);
}

void MelonInstance::stop()
{
    retroAchievementsManager = nullptr;
    vulkanOutput = nullptr;
    vulkanSurfacePresenter = nullptr;
    vulkanReadbackFrame.clear();
    lastCompletedVulkanFrame = nullptr;
    lastCompletedVulkanScale = 1;
    frameQueue.clear();
    screenshotRenderer.cleanup();
    vulkanRuntimeFailureHandled = false;
    vulkanPrepareFailureCount = 0;
}

void MelonInstance::touchScreen(u16 x, u16 y)
{
    nds->TouchScreen(x, y);
}

void MelonInstance::releaseScreen()
{
    nds->ReleaseScreen();
}

void MelonInstance::pressKey(u32 key)
{
    // Special handling for Lid input
    if (key == 16 + 7)
    {
        nds->SetLidClosed(true);
    }
    else
    {
        inputMask &= ~(1 << key);
        nds->SetKeyMask(inputMask);
    }
}

void MelonInstance::releaseKey(u32 key)
{
    // Special handling for Lid input
    if (key == 16 + 7)
    {
        nds->SetLidClosed(false);
    }
    else
    {
        inputMask |= (1 << key);
        nds->SetKeyMask(inputMask);
    }
}

void MelonInstance::setSlot2AnalogInput(float x, float y)
{
    slot2AnalogX.store(std::clamp(x, -1.0f, 1.0f), std::memory_order_relaxed);
    slot2AnalogY.store(std::clamp(y, -1.0f, 1.0f), std::memory_order_relaxed);
}

int MelonInstance::readAudioOutput(s16* buffer, int length)
{
    return nds->SPU.ReadOutput(buffer, length);
}

void MelonInstance::setAudioOutputSkew(double skew)
{
    nds->SPU.SetOutputSkew(skew);
}

void MelonInstance::loadCheats(std::list<Cheat> cheats)
{
    std::vector<ARCode> codeList;

    for (auto cheat : cheats)
    {
        ARCode arCode {
            .Enabled = true,
            .Code = cheat.code,
        };
        codeList.push_back(arCode);
    }

    nds->AREngine.Cheats = codeList;
}

int MelonInstance::sendNetPacket(u8* data, int length)
{
    return net->SendPacket(data, length, instanceId);
}

int MelonInstance::receiveNetPacket(u8* data)
{
    return net->RecvPacket(data, instanceId);
}

Frame* MelonInstance::getPresentationFrame(std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline)
{
    int vulkanRenderScale = 1;
    if (currentRenderer == Renderer::Vulkan)
        vulkanRenderScale = std::max(static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D()).GetScaleFactor(), 1);
    return frameQueue.getPresentFrame(makeFrameQueuePolicy(currentRenderer, vulkanRenderScale), deadline);
}

bool MelonInstance::waitForPresentationFrame(Frame* frame, u64 timeoutNs)
{
    if (frame == nullptr)
        return false;

    if (frame->backend != FrameBackend::VulkanImage)
        return true;

    if (!vulkanOutput)
        return false;

    return vulkanOutput->waitForFrame(frame, timeoutNs);
}

int MelonInstance::attachVulkanSurface(ANativeWindow* window, u32 width, u32 height)
{
    if (window == nullptr)
        return 0;

    if (!vulkanSurfacePresenter)
        vulkanSurfacePresenter = std::make_unique<VulkanSurfacePresenter>();

    if (!vulkanSurfacePresenter->init())
    {
        ANativeWindow_release(window);
        return 0;
    }

    return vulkanSurfacePresenter->attachSurface(window, width, height);
}

bool MelonInstance::resizeVulkanSurface(int surfaceId, u32 width, u32 height)
{
    if (!vulkanSurfacePresenter)
        return false;

    return vulkanSurfacePresenter->resizeSurface(surfaceId, width, height);
}

bool MelonInstance::configureVulkanSurface(
    int surfaceId,
    const VulkanSurfaceConfig& config,
    const VulkanBackgroundImage& backgroundImage)
{
    if (!vulkanSurfacePresenter)
        return false;

    return vulkanSurfacePresenter->configureSurface(surfaceId, config, backgroundImage);
}

void MelonInstance::detachVulkanSurface(int surfaceId)
{
    if (!vulkanSurfacePresenter)
        return;

    vulkanSurfacePresenter->detachSurface(surfaceId);
}

bool MelonInstance::presentVulkanFrame(
    std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline,
    std::optional<std::chrono::time_point<std::chrono::steady_clock>> budgetDeadline)
{
    if (currentRenderer != Renderer::Vulkan || !vulkanOutput || !vulkanSurfacePresenter)
        return false;

    auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
    const int renderScale = std::max(renderer3D.GetScaleFactor(), 1);
    const bool fastForwardActive = isFastForwardActive();
    const bool lateRealtimePresentation = !fastForwardActive && isPresentationDeadlineExpired(deadline);
    const std::optional<std::chrono::time_point<std::chrono::steady_clock>> effectiveBudgetDeadline = [&]() -> std::optional<std::chrono::time_point<std::chrono::steady_clock>> {
        if (fastForwardActive)
            return std::nullopt;
        if (budgetDeadline.has_value() && deadline.has_value())
            return std::min(*budgetDeadline, *deadline);
        if (budgetDeadline.has_value())
            return budgetDeadline;
        return deadline;
    }();
    const FrameQueuePolicy frameQueuePolicy = lateRealtimePresentation
        ? makeVulkanLateRealtimeFrameQueuePolicy(renderScale)
        : makeFrameQueuePolicy(Renderer::Vulkan, renderScale);
    const FrameQueuePolicy deferFrameQueuePolicy = [&]() -> FrameQueuePolicy {
        if (!fastForwardActive)
            return frameQueuePolicy;

        FrameQueuePolicy policy = frameQueuePolicy;
        policy.AllowDropForDeadline = false;
        return policy;
    }();
    const bool shouldProbeRealtimeBacklog = !frameQueuePolicy.AllowDropForDeadline
        && frameQueuePolicy.MaxBacklogDepth > 1;
    const bool shouldAllowBlockingHighResolutionRealtimePresentation = !frameQueuePolicy.AllowDropForDeadline
        && frameQueuePolicy.MaxBacklogDepth > 2;
    const FrameQueuePolicy candidateQueuePolicy = [&]() -> FrameQueuePolicy {
        FrameQueuePolicy policy = frameQueuePolicy;
        if (shouldProbeRealtimeBacklog)
        {
            policy.PreserveBacklogOnPresent = true;
            policy.PreferOldestFrame = false;
        }
        return policy;
    }();
    const int maxPresentAttempts = [&]() -> int {
        if (shouldProbeRealtimeBacklog)
            return static_cast<int>(std::max<u64>(1u, frameQueuePolicy.MaxBacklogDepth));
        if (frameQueuePolicy.AllowDropForDeadline)
            return static_cast<int>(std::max<u64>(1u, frameQueuePolicy.MaxBacklogDepth + 1));
        return 1;
    }();

    for (int attempt = 0; attempt < maxPresentAttempts; attempt++)
    {
        Frame* frame = frameQueue.getPresentCandidate(candidateQueuePolicy, effectiveBudgetDeadline);
        if (frame == nullptr)
            return false;

        const bool shouldContinueRealtimeProbe = shouldProbeRealtimeBacklog
            && attempt + 1 < maxPresentAttempts
            && !vulkanOutput->isFrameReady(frame);
        if (shouldContinueRealtimeProbe)
        {
            frameQueue.deferPresentedFrame(frame, candidateQueuePolicy);
            continue;
        }

        if (frameQueuePolicy.AllowDropForDeadline && !vulkanOutput->isFrameReady(frame))
        {
            frameQueue.deferPresentedFrame(frame, deferFrameQueuePolicy);
            if (frameQueuePolicy.PreferOldestFrame)
                break;
            continue;
        }

        u64 waitTimeoutNs = UINT64_MAX;
        if (effectiveBudgetDeadline.has_value())
        {
            const auto now = std::chrono::steady_clock::now();
            if (*effectiveBudgetDeadline <= now)
                waitTimeoutNs = 0;
            else
                waitTimeoutNs = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(*effectiveBudgetDeadline - now).count());
        }

        if (frameQueuePolicy.AllowDropForDeadline)
            waitTimeoutNs = 0;

        VulkanCompositionInputs compositionInputs{};
        if (!vulkanOutput->buildCompositionInputs(
                frame,
                renderer3D,
                renderScale,
                false,
                false,
                false,
                compositionInputs))
        {
            frameQueue.deferPresentedFrame(frame, deferFrameQueuePolicy);
            if (!frameQueuePolicy.AllowDropForDeadline || frameQueuePolicy.PreferOldestFrame)
                return false;
            continue;
        }

        const u64 presenterTimeoutNs = [&]() -> u64 {
            if (shouldAllowBlockingHighResolutionRealtimePresentation)
                return UINT64_MAX;

            if (!shouldProbeRealtimeBacklog || waitTimeoutNs == UINT64_MAX)
                return waitTimeoutNs;

            return std::max(waitTimeoutNs, kVulkanHighResolutionRealtimePresenterBudgetFloorNs);
        }();

        const bool presented = vulkanSurfacePresenter->presentFrame(frame, *vulkanOutput, compositionInputs, presenterTimeoutNs);
        if (presented)
        {
            frameQueue.commitPresentedFrame(frame, shouldProbeRealtimeBacklog ? candidateQueuePolicy : frameQueuePolicy);
            return true;
        }

        frameQueue.deferPresentedFrame(frame, deferFrameQueuePolicy);
        if (!frameQueuePolicy.AllowDropForDeadline || frameQueuePolicy.PreferOldestFrame)
            return false;
    }

    return false;
}

void MelonInstance::requestVulkanPresentationResync()
{
    if (currentRenderer != Renderer::Vulkan)
        return;

    frameQueue.requestPresentationResync();
    static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D()).requestPostFastForwardDrain();
    lastCompletedVulkanFrame = nullptr;
    lastCompletedVulkanScale = 1;
    vulkanReadbackFrame.clear();
}

std::vector<u32> MelonInstance::captureCurrentFrameForDebug()
{
    if (!areRendererDebugToolsEnabled())
        return {};

    constexpr size_t kScreenshotPixelCount =
        static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight) * 2u;

    if (currentRenderer == Renderer::Vulkan)
        (void)updateVulkanScreenshot(lastCompletedVulkanFrame, lastCompletedVulkanScale, true);

    const u32* screenshot = screenshotRenderer.getScreenshot();
    if (screenshot == nullptr)
        return {};

    std::vector<u32> pixels(kScreenshotPixelCount);
    std::memcpy(pixels.data(), screenshot, kScreenshotPixelCount * sizeof(u32));
    return pixels;
}

std::vector<u32> MelonInstance::captureCurrent3dDimensionsForDebug()
{
    if (!areRendererDebugToolsEnabled())
        return {};

    if (nds == nullptr)
        return {};

    if (currentRenderer == Renderer::Software)
    {
        const auto& renderer3D = static_cast<const SoftRenderer&>(nds->GPU.GetRenderer3D());
        return {renderer3D.GetColorTargetWidth(), renderer3D.GetColorTargetHeight()};
    }

    if (currentRenderer == Renderer::OpenGl)
    {
        const auto& renderer3D = static_cast<const GLRenderer&>(nds->GPU.GetRenderer3D());
        const u32 width = renderer3D.GetColorTargetWidth();
        const u32 height = renderer3D.GetColorTargetHeight();
        if (width == 0 || height == 0)
            return {};
        return {width, height};
    }

    if (currentRenderer != Renderer::Vulkan)
        return {static_cast<u32>(kScreenshotScreenWidth), static_cast<u32>(kScreenshotScreenHeight)};

    if (lastCompletedVulkanFrame != nullptr && vulkanOutput != nullptr)
    {
        u32 width = 0;
        u32 height = 0;
        if (vulkanOutput->getPreparedRenderer3dDimensions(lastCompletedVulkanFrame, width, height))
            return {width, height};
    }

    const auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
    const u32 width = renderer3D.GetColorTargetWidth();
    const u32 height = renderer3D.GetColorTargetHeight();
    if (width == 0 || height == 0)
        return {};

    return {width, height};
}

std::vector<u32> MelonInstance::captureCurrentPackedTopPrimaryForDebug()
{
    return captureCurrentPackedPrimaryForDebug(true);
}

std::vector<u32> MelonInstance::captureCurrentPackedBottomPrimaryForDebug()
{
    return captureCurrentPackedPrimaryForDebug(false);
}

std::vector<u32> MelonInstance::captureCurrentPackedPrimaryForDebug(bool topScreen)
{
    if (!areRendererDebugToolsEnabled())
        return {};

    if (nds == nullptr)
        return {};

    const int frontbuf = nds->GPU.FrontBuffer;
    const u32* topPacked = nullptr;
    const u32* bottomPacked = nullptr;
    u32 packedStride = 256 * 3 + 1;
    u32 packedHeight = 192;
    bool packedScreenSwap = (nds->PowerControl9 & (1u << 15)) != 0;

    const bool usingPreparedPackedBuffers = lastCompletedVulkanFrame != nullptr
        && vulkanOutput != nullptr
        && vulkanOutput->getPreparedPackedBuffers(
            lastCompletedVulkanFrame,
            topPacked,
            bottomPacked,
            packedStride,
            packedHeight,
            packedScreenSwap);

    if (!usingPreparedPackedBuffers)
    {
        if (nds->GPU.Framebuffer[frontbuf][0] != nullptr)
            topPacked = nds->GPU.Framebuffer[frontbuf][0].get();
        if (nds->GPU.Framebuffer[frontbuf][1] != nullptr)
            bottomPacked = nds->GPU.Framebuffer[frontbuf][1].get();
    }

    const u32* packed = topScreen ? topPacked : bottomPacked;
    if (packed == nullptr || packedStride < 256 || packedHeight < 192)
        return {};

    std::vector<u32> pixels(static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight));
    for (u32 y = 0; y < static_cast<u32>(kScreenshotScreenHeight); y++)
    {
        const u32 lineBase = y * packedStride;
        for (u32 x = 0; x < static_cast<u32>(kScreenshotScreenWidth); x++)
            pixels[static_cast<size_t>(y) * static_cast<size_t>(kScreenshotScreenWidth) + static_cast<size_t>(x)] =
                expandPackedColor6ToRgba8(packed[lineBase + x]);
    }

    return pixels;
}

std::vector<u32> MelonInstance::captureCurrent3dFrameForDebug()
{
    if (!areRendererDebugToolsEnabled())
        return {};

    if (nds == nullptr)
        return {};

    auto& renderer3DBase = nds->GPU.GetRenderer3D();
    if (currentRenderer == Renderer::Vulkan)
    {
        if (lastCompletedVulkanFrame != nullptr && vulkanOutput != nullptr)
        {
            u32 width = 0;
            u32 height = 0;
            if (vulkanOutput->getPreparedRenderer3dDimensions(lastCompletedVulkanFrame, width, height)
                && width > 0
                && height > 0)
            {
                std::vector<u32> pixels(static_cast<size_t>(width) * static_cast<size_t>(height));
                if (vulkanOutput->readPreparedRenderer3dPixels(
                        lastCompletedVulkanFrame,
                        pixels.data(),
                        pixels.size(),
                        width,
                        height))
                {
                    return pixels;
                }
            }
        }

        auto& renderer3D = static_cast<VulkanRenderer3D&>(renderer3DBase);
        return renderer3D.CaptureColorTargetForDebug();
    }

    if (currentRenderer == Renderer::Software)
        return static_cast<SoftRenderer&>(renderer3DBase).CaptureColorTargetForDebug();

    if (currentRenderer == Renderer::OpenGl)
    {
        ScopedDebugOpenGlContext contextBinding;
        if (!contextBinding.IsReady())
            return {};
        return static_cast<GLRenderer&>(renderer3DBase).CaptureColorTargetForDebug();
    }

    renderer3DBase.PrepareCaptureFrame();
    std::vector<u32> pixels(static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight));
    for (int line = 0; line < kScreenshotScreenHeight; line++)
    {
        const u32* linePixels = renderer3DBase.GetLine(line);
        if (linePixels == nullptr)
            return {};
        std::memcpy(
            pixels.data() + static_cast<size_t>(line) * static_cast<size_t>(kScreenshotScreenWidth),
            linePixels,
            static_cast<size_t>(kScreenshotScreenWidth) * sizeof(u32)
        );
    }
    return pixels;
}

std::vector<u32> MelonInstance::captureCurrent3dCaptureFrameForDebug()
{
    if (!areRendererDebugToolsEnabled())
        return {};

    if (nds == nullptr)
        return {};

    if (const auto* renderer2D = dynamic_cast<const GPU2D::SoftRenderer*>(&nds->GPU.GetRenderer2D()))
    {
        if (const u32* capture3dSource = renderer2D->GetDebugCapture3dSource())
        {
            std::vector<u32> pixels(static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight));
            for (size_t i = 0; i < pixels.size(); i++)
                pixels[i] = expandPackedColor6ToRgba8(capture3dSource[i]);
            return pixels;
        }
    }

    auto& renderer3DBase = nds->GPU.GetRenderer3D();
    const auto captureLines = [&renderer3DBase]() -> std::vector<u32> {
        renderer3DBase.PrepareCaptureFrame();
        std::vector<u32> pixels(static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight));
        for (int line = 0; line < kScreenshotScreenHeight; line++)
        {
            const u32* linePixels = renderer3DBase.GetLine(line);
            if (linePixels == nullptr)
                return {};
            std::memcpy(
                pixels.data() + static_cast<size_t>(line) * static_cast<size_t>(kScreenshotScreenWidth),
                linePixels,
                static_cast<size_t>(kScreenshotScreenWidth) * sizeof(u32)
            );
        }
        return pixels;
    };

    if (currentRenderer == Renderer::OpenGl)
    {
        ScopedDebugOpenGlContext contextBinding;
        if (!contextBinding.IsReady())
            return {};
        return captureLines();
    }

    return captureLines();
}

std::vector<u32> MelonInstance::captureCurrent3dDepthForDebug()
{
    if (!areRendererDebugToolsEnabled() || nds == nullptr)
        return {};

    auto& renderer3DBase = nds->GPU.GetRenderer3D();
    if (currentRenderer == Renderer::Vulkan)
        return static_cast<VulkanRenderer3D&>(renderer3DBase).CaptureTopDepthForDebug();
    if (currentRenderer == Renderer::Software)
        return static_cast<SoftRenderer&>(renderer3DBase).CaptureTopDepthForDebug();
    if (currentRenderer == Renderer::OpenGl)
    {
        ScopedDebugOpenGlContext contextBinding;
        if (!contextBinding.IsReady())
            return {};
        return static_cast<GLRenderer&>(renderer3DBase).CaptureTopDepthForDebug();
    }

    return {};
}

std::vector<u32> MelonInstance::captureCurrent3dAttrForDebug()
{
    if (!areRendererDebugToolsEnabled() || nds == nullptr)
        return {};

    auto& renderer3DBase = nds->GPU.GetRenderer3D();
    if (currentRenderer == Renderer::Vulkan)
        return static_cast<VulkanRenderer3D&>(renderer3DBase).CaptureTopAttrForDebug();
    if (currentRenderer == Renderer::Software)
        return static_cast<SoftRenderer&>(renderer3DBase).CaptureTopAttrForDebug();
    if (currentRenderer == Renderer::OpenGl)
    {
        ScopedDebugOpenGlContext contextBinding;
        if (!contextBinding.IsReady())
            return {};
        return static_cast<GLRenderer&>(renderer3DBase).CaptureTopAttrForDebug();
    }

    return {};
}

std::vector<u32> MelonInstance::captureCurrent3dCoverageForDebug()
{
    if (!areRendererDebugToolsEnabled() || nds == nullptr)
        return {};

    auto& renderer3DBase = nds->GPU.GetRenderer3D();
    if (currentRenderer == Renderer::Vulkan)
        return static_cast<VulkanRenderer3D&>(renderer3DBase).CaptureTopCoverageForDebug();
    if (currentRenderer == Renderer::Software)
        return static_cast<SoftRenderer&>(renderer3DBase).CaptureTopCoverageForDebug();
    if (currentRenderer == Renderer::OpenGl)
    {
        ScopedDebugOpenGlContext contextBinding;
        if (!contextBinding.IsReady())
            return {};
        return static_cast<GLRenderer&>(renderer3DBase).CaptureTopCoverageForDebug();
    }

    return {};
}

void MelonInstance::dumpDebugSnapshot()
{
    if (!areRendererDebugToolsEnabled())
        return;

    const auto rendererName = [](Renderer renderer) -> const char* {
        switch (renderer)
        {
            case Renderer::Software: return "software";
            case Renderer::OpenGl: return "opengl";
            case Renderer::Vulkan: return "vulkan";
            case Renderer::Compute: return "compute";
        }

        return "unknown";
    };

    const FrameQueueStats queueStats = frameQueue.takeStatsSnapshotAndReset();
    if (currentRenderer != Renderer::Vulkan || nds == nullptr)
    {
        Platform::Log(
            Platform::LogLevel::Warn,
            "RendererDebug[Snapshot]: renderer=%s backlog=%llu/%llu queued=%llu discarded=%llu presented=%llu staleDropped=%llu reusedPrev=%llu stolen=%llu renderDropped=%llu presentDropped=%llu",
            rendererName(currentRenderer),
            static_cast<unsigned long long>(queueStats.CurrentBacklogDepth),
            static_cast<unsigned long long>(queueStats.MaxBacklogDepth),
            static_cast<unsigned long long>(queueStats.RenderFramesQueued),
            static_cast<unsigned long long>(queueStats.RenderFramesDiscarded),
            static_cast<unsigned long long>(queueStats.PresentFramesReturned),
            static_cast<unsigned long long>(queueStats.StaleFramesDropped),
            static_cast<unsigned long long>(queueStats.PreviousFrameReused),
            static_cast<unsigned long long>(queueStats.PendingFramesStolenForRender),
            static_cast<unsigned long long>(queueStats.RenderFramesDroppedByPolicy),
            static_cast<unsigned long long>(queueStats.PresentFramesDroppedByPolicy)
        );
        return;
    }

    const auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
    const Frame* debugFrame = lastCompletedVulkanFrame;
    const VulkanPresenterPacingStats presenterStats = vulkanSurfacePresenter
        ? vulkanSurfacePresenter->takePacingStatsSnapshotAndReset()
        : VulkanPresenterPacingStats{};
    const auto& deviceProfile = VulkanContext::Get().GetDeviceProfile();
    struct PackedScreenStats
    {
        std::array<u32, 4> DisplayModeCounts{};
        std::array<u32, 8> CompModeCounts{};
        int MinXOffset = 0;
        int MaxXOffset = 0;
        bool HasOffsets = false;
    };

    const auto collectPackedScreenStats = [&](const u32* packed, u32 packedStride, u32 packedHeight) -> PackedScreenStats {
        PackedScreenStats stats{};
        constexpr int kMetaIndex = 256 * 3;

        if (packed == nullptr || packedStride < kMetaIndex + 1 || packedHeight < 192)
            return stats;

        for (int y = 0; y < 192; y++)
        {
            const u32 lineBase = static_cast<u32>(y) * packedStride;
            const u32 meta = packed[lineBase + kMetaIndex];
            const u32 displayMode = (meta >> 16) & 0x3u;
            stats.DisplayModeCounts[displayMode]++;

            const int xOffset = static_cast<int>((meta >> 24) & 0xFFu)
                - ((((meta >> 16) & 0x80u) != 0u) ? 256 : 0);
            if (!stats.HasOffsets)
            {
                stats.MinXOffset = xOffset;
                stats.MaxXOffset = xOffset;
                stats.HasOffsets = true;
            }
            else
            {
                stats.MinXOffset = std::min(stats.MinXOffset, xOffset);
                stats.MaxXOffset = std::max(stats.MaxXOffset, xOffset);
            }

            if (displayMode != 1u)
                continue;

            for (int x = 0; x < 256; x++)
            {
                const u32 control = packed[lineBase + 512u + static_cast<u32>(x)];
                const u32 compMode = (control >> 24) & 0xFu;
                if (compMode < stats.CompModeCounts.size())
                    stats.CompModeCounts[compMode]++;
            }
        }

        return stats;
    };
    const int frontbuf = nds->GPU.FrontBuffer;
    const u32* topPacked = nullptr;
    const u32* bottomPacked = nullptr;
    u32 packedStride = 256 * 3 + 1;
    u32 packedHeight = 192;
    bool packedScreenSwap = (nds->PowerControl9 & (1u << 15)) != 0;
    const bool usingPreparedPackedBuffers = debugFrame != nullptr
        && vulkanOutput != nullptr
        && vulkanOutput->getPreparedPackedBuffers(debugFrame, topPacked, bottomPacked, packedStride, packedHeight, packedScreenSwap);

    if (!usingPreparedPackedBuffers)
    {
        if (nds->GPU.Framebuffer[frontbuf][0])
            topPacked = nds->GPU.Framebuffer[frontbuf][0].get();
        if (nds->GPU.Framebuffer[frontbuf][1])
            bottomPacked = nds->GPU.Framebuffer[frontbuf][1].get();
    }

    const PackedScreenStats topPackedStats = collectPackedScreenStats(topPacked, packedStride, packedHeight);
    const PackedScreenStats bottomPackedStats = collectPackedScreenStats(bottomPacked, packedStride, packedHeight);
    if (const auto* renderer2D = dynamic_cast<const GPU2D::SoftRenderer*>(&nds->GPU.GetRenderer2D()))
    {
        const auto& captureStats = renderer2D->GetDebugCaptureStats();
        Platform::Log(
            Platform::LogLevel::Warn,
            "RendererDebug[Capture]: lines=%u width=%u mode=%u bit24=%u direct3dLines=%u compositeLines=%u opaque3dPixels=%u backdrop3dPixels=%u comp=%u/%u/%u/%u/%u/%u/%u/%u",
            captureStats.CaptureLines,
            captureStats.CaptureWidth,
            captureStats.CaptureMode,
            captureStats.CaptureBit24,
            captureStats.Direct3DLines,
            captureStats.SourceACompositeLines,
            captureStats.Opaque3DSourcePixels,
            captureStats.Opaque3DBackdropPixels,
            captureStats.CompModeCounts[0],
            captureStats.CompModeCounts[1],
            captureStats.CompModeCounts[2],
            captureStats.CompModeCounts[3],
            captureStats.CompModeCounts[4],
            captureStats.CompModeCounts[5],
            captureStats.CompModeCounts[6],
            captureStats.CompModeCounts[7]
        );

        const u32* capture3dSource = renderer2D->GetDebugCapture3dSource();
        if (capture3dSource != nullptr && bottomPacked != nullptr && packedStride >= 256u)
        {
            struct CaptureSamplePoint
            {
                const char* label;
                u32 x;
                u32 y;
            };

            constexpr CaptureSamplePoint kCaptureSamplePoints[] = {
                {"seamA", 85u, 14u},
                {"goodA", 84u, 14u},
                {"seamB", 75u, 58u},
                {"goodB", 74u, 58u},
                {"seamC", 150u, 81u},
                {"goodC", 149u, 81u},
            };

            std::string sampleLog;
            for (const CaptureSamplePoint& sample : kCaptureSamplePoints)
            {
                if (!sampleLog.empty())
                    sampleLog += ' ';

                const size_t sourceOffset = static_cast<size_t>(sample.y) * 256u + static_cast<size_t>(sample.x);
                const size_t packedOffset = static_cast<size_t>(sample.y) * static_cast<size_t>(packedStride) + static_cast<size_t>(sample.x);
                const u32 sourceRaw = capture3dSource[sourceOffset];
                const u32 packedRaw = bottomPacked[packedOffset];

                char entry[96];
                std::snprintf(
                    entry,
                    sizeof(entry),
                    "%s(%u,%u)=src:%08X packed:%08X",
                    sample.label,
                    sample.x,
                    sample.y,
                    sourceRaw,
                    packedRaw
                );
                sampleLog += entry;
            }

            Platform::Log(
                Platform::LogLevel::Warn,
                "RendererDebug[CaptureSamples]: %s",
                sampleLog.c_str()
            );
        }
    }

    Platform::Log(
        Platform::LogLevel::Warn,
        "VulkanDebug[Snapshot]: device='%s' vendor=%#x deviceId=%#x adreno=%d mali=%d g52=%d threaded=%d betterPolygons=%d renderScale=%d coverageFix=%d coveragePx=%.3f passiveRepeatPx=%.3f coverageBias=%.5f lastFrame=%ux%u backend=%u queue backlog=%llu/%llu queued=%llu discarded=%llu presented=%llu staleDropped=%llu reusedPrev=%llu stolen=%llu renderDropped=%llu presentDropped=%llu pacing presented=%llu direct=%llu fallback=%llu acquireTimeouts=%llu surfaceWaitTimeouts=%llu deadlineSkipped=%llu recoveries=%llu presentMode=%d",
        deviceProfile.DeviceName.c_str(),
        deviceProfile.VendorId,
        deviceProfile.DeviceId,
        deviceProfile.IsAdreno ? 1 : 0,
        deviceProfile.IsArmMali ? 1 : 0,
        deviceProfile.IsMaliG52Class ? 1 : 0,
        renderer3D.IsThreaded() ? 1 : 0,
        renderer3D.UsesBetterPolygons() ? 1 : 0,
        renderer3D.GetScaleFactor(),
        renderer3D.IsCoverageFixEnabled() ? 1 : 0,
        renderer3D.GetCoverageFixPx(),
        renderer3D.GetPassiveCoverageFixRepeatPx(),
        renderer3D.GetCoverageFixDepthBias(),
        debugFrame != nullptr ? debugFrame->width : 0u,
        debugFrame != nullptr ? debugFrame->height : 0u,
        debugFrame != nullptr ? static_cast<unsigned>(debugFrame->backend) : static_cast<unsigned>(FrameBackend::VulkanImage),
        static_cast<unsigned long long>(queueStats.CurrentBacklogDepth),
        static_cast<unsigned long long>(queueStats.MaxBacklogDepth),
        static_cast<unsigned long long>(queueStats.RenderFramesQueued),
        static_cast<unsigned long long>(queueStats.RenderFramesDiscarded),
        static_cast<unsigned long long>(queueStats.PresentFramesReturned),
        static_cast<unsigned long long>(queueStats.StaleFramesDropped),
        static_cast<unsigned long long>(queueStats.PreviousFrameReused),
        static_cast<unsigned long long>(queueStats.PendingFramesStolenForRender),
        static_cast<unsigned long long>(queueStats.RenderFramesDroppedByPolicy),
        static_cast<unsigned long long>(queueStats.PresentFramesDroppedByPolicy),
        static_cast<unsigned long long>(presenterStats.PresentedFrames),
        static_cast<unsigned long long>(presenterStats.DirectPresentedFrames),
        static_cast<unsigned long long>(presenterStats.FallbackPresentedFrames),
        static_cast<unsigned long long>(presenterStats.AcquireTimeouts),
        static_cast<unsigned long long>(presenterStats.SurfaceWaitTimeouts),
        static_cast<unsigned long long>(presenterStats.PresentSkippedForDeadline),
        static_cast<unsigned long long>(presenterStats.SwapchainRecoveries),
        static_cast<int>(presenterStats.PresentMode)
    );

    Platform::Log(
        Platform::LogLevel::Warn,
        "VulkanDebug[Packed]: prepared=%d screenSwap=%d topDM=%u/%u/%u/%u topComp=%u/%u/%u/%u/%u/%u/%u/%u topX=%d..%d bottomDM=%u/%u/%u/%u bottomComp=%u/%u/%u/%u/%u/%u/%u/%u bottomX=%d..%d",
        usingPreparedPackedBuffers ? 1 : 0,
        packedScreenSwap ? 1 : 0,
        topPackedStats.DisplayModeCounts[0],
        topPackedStats.DisplayModeCounts[1],
        topPackedStats.DisplayModeCounts[2],
        topPackedStats.DisplayModeCounts[3],
        topPackedStats.CompModeCounts[0],
        topPackedStats.CompModeCounts[1],
        topPackedStats.CompModeCounts[2],
        topPackedStats.CompModeCounts[3],
        topPackedStats.CompModeCounts[4],
        topPackedStats.CompModeCounts[5],
        topPackedStats.CompModeCounts[6],
        topPackedStats.CompModeCounts[7],
        topPackedStats.MinXOffset,
        topPackedStats.MaxXOffset,
        bottomPackedStats.DisplayModeCounts[0],
        bottomPackedStats.DisplayModeCounts[1],
        bottomPackedStats.DisplayModeCounts[2],
        bottomPackedStats.DisplayModeCounts[3],
        bottomPackedStats.CompModeCounts[0],
        bottomPackedStats.CompModeCounts[1],
        bottomPackedStats.CompModeCounts[2],
        bottomPackedStats.CompModeCounts[3],
        bottomPackedStats.CompModeCounts[4],
        bottomPackedStats.CompModeCounts[5],
        bottomPackedStats.CompModeCounts[6],
        bottomPackedStats.CompModeCounts[7],
        bottomPackedStats.MinXOffset,
        bottomPackedStats.MaxXOffset
    );
}

void MelonInstance::updateVulkanFastForwardRenderScale()
{
    if (currentRenderer != Renderer::Vulkan)
        return;

    auto& vulkanRenderSettings = static_cast<VulkanRenderSettings&>(*currentConfiguration->renderSettings);
    auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
    const int desiredScale = getEffectiveVulkanRenderScale(vulkanRenderSettings);
    if (renderer3D.GetScaleFactor() == desiredScale)
        return;

    renderer3D.SetRenderSettings(
        vulkanRenderSettings.threadedRendering,
        vulkanRenderSettings.betterPolygons,
        desiredScale,
        vulkanRenderSettings.conservativeCoverageEnabled,
        vulkanRenderSettings.conservativeCoveragePx,
        vulkanRenderSettings.conservativeCoverageDepthBias,
        vulkanRenderSettings.conservativeCoverageApplyRepeat,
        vulkanRenderSettings.conservativeCoverageApplyClamp,
        vulkanRenderSettings.debug3dClearMagenta,
        nds->GPU);
    requestVulkanPresentationResync();
}

void MelonInstance::updateConfiguration(std::shared_ptr<EmulatorConfiguration> newConfiguration)
{
    if (nds)
    {
        nds->SPU.SetInterpolation(static_cast<AudioInterpolation>(newConfiguration->audioSettings.audioInterpolation));
        nds->SPU.SetDegrade10Bit(static_cast<AudioBitDepth>(newConfiguration->audioSettings.audioBitrate));
    }

    rewindManager.UpdateRewindSettings(newConfiguration->rewindEnabled, newConfiguration->rewindLengthSeconds, newConfiguration->rewindCaptureSpacingSeconds);

    currentConfiguration = newConfiguration;
    isRenderConfigurationDirty = true;
}

void MelonInstance::requestNdsSaveWrite(const u8* saveData, u32 saveLength, u32 writeOffset, u32 writeLength)
{
    if (ndsSave)
        ndsSave->RequestFlush(saveData, saveLength, writeOffset, writeLength);
}

void MelonInstance::requestGbaSaveWrite(const u8* saveData, u32 saveLength, u32 writeOffset, u32 writeLength)
{
    if (gbaSave)
        gbaSave->RequestFlush(saveData, saveLength, writeOffset, writeLength);
}

void MelonInstance::requestFirmwareSaveWrite(const u8* saveData, u32 saveLength, u32 writeOffset, u32 writeLength)
{
    if (firmwareSave)
        firmwareSave->RequestFlush(saveData, saveLength, writeOffset, writeLength);
}

bool MelonInstance::areSaveStatesAllowed()
{
    if (!retroAchievementsManager)
        return true;

    return retroAchievementsManager->AreSaveStatesAllowed();
}

bool MelonInstance::saveState(Savestate* state, bool refreshScreenshot)
{
    if (refreshScreenshot && currentRenderer == Renderer::Vulkan)
        (void)updateVulkanScreenshot(lastCompletedVulkanFrame, lastCompletedVulkanScale, true);

    if (!retroAchievementsManager->DoSavestate(state))
        return false;

    return nds->DoSavestate(state);
}

bool MelonInstance::loadState(Savestate* state)
{
    if (!retroAchievementsManager->DoSavestate(state))
        return false;

    return nds->DoSavestate(state);
}

RewindWindow MelonInstance::getRewindWindow()
{
    return RewindWindow {
        .currentFrame = frame,
        .rewindStates = rewindManager.GetRewindWindow(),
    };
}

bool MelonInstance::loadRewindState(RewindSaveState rewindSaveState)
{
    Savestate* savestate = new Savestate(rewindSaveState.buffer, rewindSaveState.bufferContentSize, false);
    if (savestate->Error)
    {
        delete savestate;
        return false;
    }

    bool result = loadState(savestate);
    if (result)
    {
        frame = rewindSaveState.frame;
        rewindManager.OnRewindFromState(rewindSaveState);
    }

    delete savestate;

    return result;
}

void MelonInstance::setupAchievements(
    std::list<RetroAchievements::RAAchievement> achievements,
    std::list<RetroAchievements::RALeaderboard> leaderboards,
    std::optional<std::string> richPresenceScript,
    std::optional<RetroAchievements::RARuntimeBridgeConfig> runtimeBridgeConfig
)
{
    if (instanceId == 0)
    {
        retroAchievementsManager->ConfigureRuntimeBridge(std::move(runtimeBridgeConfig));
        retroAchievementsManager->LoadAchievements(achievements);
        retroAchievementsManager->LoadLeaderboards(leaderboards);
        if (richPresenceScript)
            retroAchievementsManager->SetupRichPresence(*richPresenceScript);
        retroAchievementsManager->ActivatePreferredRuntime();
    }
}

void MelonInstance::unloadRetroAchievementsData()
{
    retroAchievementsManager->UnloadEverything();
}

std::string MelonInstance::getRichPresenceStatus()
{
    if (instanceId == 0 && retroAchievementsManager)
        return retroAchievementsManager->GetRichPresenceStatus();
    else
        return "";
}

std::vector<RetroAchievements::RARuntimeAchievement> MelonInstance::getRuntimeAchievements()
{
    if (instanceId == 0 && retroAchievementsManager)
        return retroAchievementsManager->GetRuntimeAchievements();
    else
        return { };
}

std::vector<RetroAchievements::RARuntimeAchievementBucketEntry> MelonInstance::getRuntimeAchievementBuckets()
{
    if (instanceId == 0 && retroAchievementsManager)
        return retroAchievementsManager->GetRuntimeAchievementBuckets();
    else
        return { };
}

std::vector<long> MelonInstance::getRuntimeSubsetIds()
{
    if (instanceId == 0 && retroAchievementsManager)
        return retroAchievementsManager->GetRuntimeSubsetIds();
    else
        return { };
}

void MelonInstance::updateRenderer()
{
    Renderer newRenderer = currentConfiguration->renderer;

    if (newRenderer != currentRenderer)
    {
        vulkanRuntimeFailureHandled = false;

        if (newRenderer == Renderer::Vulkan)
        {
            if (!vulkanOutput)
                vulkanOutput = std::make_unique<VulkanOutput>();

            if (!vulkanOutput->isInitialized() && !vulkanOutput->init())
            {
                Platform::Log(Platform::LogLevel::Error, "Failed to initialize Vulkan renderer backend");
                if (eventMessenger)
                    eventMessenger->onRendererInitFailed(Renderer::Vulkan);

                if (frame == 0)
                {
                    Platform::Log(Platform::LogLevel::Error, "Aborting launch after Vulkan renderer initialization failure");
                    nds->Stop(Platform::StopReason::BadExceptionRegion);
                }

                currentConfiguration->renderer = currentRenderer;
                return;
            }

            vulkanRuntimeFailureHandled = false;
        }
        else if (vulkanOutput)
        {
            vulkanOutput = nullptr;
            vulkanSurfacePresenter = nullptr;
        }

        std::unique_ptr<Renderer3D> nextRenderer = nullptr;
        switch (newRenderer)
        {
            case Renderer::Software:
                nextRenderer = std::make_unique<SoftRenderer>();
                break;
            case Renderer::OpenGl:
                nextRenderer = GLRenderer::New();
                break;
            case Renderer::Vulkan:
            {
                auto vulkanRenderer = VulkanRenderer3D::New();
                auto vulkanRenderSettings = static_cast<VulkanRenderSettings&>(*currentConfiguration->renderSettings);

                if (vulkanRenderer)
                {
                    vulkanRenderer->SetRenderSettings(
                        vulkanRenderSettings.threadedRendering,
                        vulkanRenderSettings.betterPolygons,
                        getEffectiveVulkanRenderScale(vulkanRenderSettings),
                        vulkanRenderSettings.conservativeCoverageEnabled,
                        vulkanRenderSettings.conservativeCoveragePx,
                        vulkanRenderSettings.conservativeCoverageDepthBias,
                        vulkanRenderSettings.conservativeCoverageApplyRepeat,
                        vulkanRenderSettings.conservativeCoverageApplyClamp,
                        vulkanRenderSettings.debug3dClearMagenta,
                        nds->GPU
                    );
                }

                if (!vulkanRenderer)
                {
                    Platform::Log(Platform::LogLevel::Error, "Failed to create Vulkan renderer backend");
                    if (eventMessenger)
                        eventMessenger->onRendererInitFailed(Renderer::Vulkan);

                    if (frame == 0)
                    {
                        Platform::Log(Platform::LogLevel::Error, "Aborting launch after Vulkan renderer validation failure");
                        nds->Stop(Platform::StopReason::BadExceptionRegion);
                    }

                    if (currentRenderer != Renderer::Vulkan)
                        vulkanOutput = nullptr;

                    currentConfiguration->renderer = currentRenderer;
                    return;
                }

                nextRenderer = std::move(vulkanRenderer);
                break;
            }
            case Renderer::Compute:
                nextRenderer = ComputeRenderer::New();
                break;
            default: __builtin_unreachable();
        }

        if (!nextRenderer)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to create requested renderer backend");
            currentConfiguration->renderer = currentRenderer;
            return;
        }

        nds->GPU.SetRenderer3D(std::move(nextRenderer));
        currentRenderer = newRenderer;
    }

    switch (newRenderer)
    {
        case Renderer::Software:
        {
            auto softwareRenderSettings = static_cast<SoftwareRenderSettings&>(*currentConfiguration->renderSettings);
            static_cast<SoftRenderer&>(nds->GPU.GetRenderer3D()).SetThreaded(softwareRenderSettings.threadedRendering, nds->GPU);
            break;
        }
        case Renderer::OpenGl:
        {
            auto glRenderSettings = static_cast<OpenGlRenderSettings&>(*currentConfiguration->renderSettings);
            auto& renderer3d = static_cast<GLRenderer&>(nds->GPU.GetRenderer3D());
            renderer3d.SetRenderSettings(glRenderSettings.betterPolygons, glRenderSettings.scale);
            renderer3d.SetCoverageFixSettings(
                glRenderSettings.conservativeCoverageEnabled,
                glRenderSettings.conservativeCoveragePx,
                glRenderSettings.conservativeCoverageDepthBias,
                glRenderSettings.conservativeCoverageApplyRepeat,
                glRenderSettings.conservativeCoverageApplyClamp,
                glRenderSettings.debug3dClearMagenta);
            break;
        }
        case Renderer::Vulkan:
        {
            auto vulkanRenderSettings = static_cast<VulkanRenderSettings&>(*currentConfiguration->renderSettings);
            auto& renderer3d = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
            renderer3d.SetRenderSettings(
                vulkanRenderSettings.threadedRendering,
                vulkanRenderSettings.betterPolygons,
                getEffectiveVulkanRenderScale(vulkanRenderSettings),
                vulkanRenderSettings.conservativeCoverageEnabled,
                vulkanRenderSettings.conservativeCoveragePx,
                vulkanRenderSettings.conservativeCoverageDepthBias,
                vulkanRenderSettings.conservativeCoverageApplyRepeat,
                vulkanRenderSettings.conservativeCoverageApplyClamp,
                vulkanRenderSettings.debug3dClearMagenta,
                nds->GPU);

            if (!vulkanRuntimeConfigLogged)
            {
                Platform::Log(
                    Platform::LogLevel::Warn,
                    "VulkanRuntime[Renderer]: threaded=%d ringContexts=%llu readbackWaitScope=%s renderScale=%d outputScale=%d betterPolygons=%d diagFlags=0x%08X",
                    vulkanRenderSettings.threadedRendering ? 1 : 0,
                    static_cast<unsigned long long>(renderer3d.GetAsyncRenderContextCount()),
                    renderer3d.WaitsForReadbackSourceOnly() ? "readback-only" : "hot-path",
                    std::max(renderer3d.GetScaleFactor(), 1),
                    getConfiguredVulkanScale(vulkanRenderSettings),
                    vulkanRenderSettings.betterPolygons ? 1 : 0,
                    static_cast<unsigned>(MelonDSAndroid::getVulkanDiagnosticFlags())
                );
                vulkanRuntimeConfigLogged = true;
            }
            break;
        }
        case Renderer::Compute:
        {
            auto computeRenderSettings = static_cast<ComputeRenderSettings&>(*currentConfiguration->renderSettings);
            static_cast<ComputeRenderer&>(nds->GPU.GetRenderer3D()).SetRenderSettings(computeRenderSettings.scale,computeRenderSettings.highResCoordinates);
            break;
        }
        default: __builtin_unreachable();
    }
}

void MelonInstance::setBatteryLevels()
{
    if (consoleType == 1)
    {
        auto dsi = static_cast<DSi*>(nds);
        dsi->I2C.GetBPTWL()->SetBatteryLevel(DSi_BPTWL::batteryLevel_Full);
        dsi->I2C.GetBPTWL()->SetBatteryCharging(false);
    }
    else
    {
        nds->SPI.GetPowerMan()->SetBatteryLevelOkay(true);
    }
}

void MelonInstance::setDateTime()
{
    std::time_t t = std::time(0);
    std::tm* now = std::localtime(&t);

    nds->RTC.SetDateTime(now->tm_year + 1900, now->tm_mon + 1, now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
}

bool MelonInstance::updateVulkanScreenshot(Frame* frame, int scale, bool clearOnFailure)
{
    const size_t screenshotPixelCount = static_cast<size_t>(kScreenshotScreenWidth) * static_cast<size_t>(kScreenshotScreenHeight) * 2;
    auto clearScreenshot = [&]() {
        if (clearOnFailure)
            std::fill_n(screenshotRenderer.getScreenshot(), screenshotPixelCount, 0u);
    };

    if (currentRenderer != Renderer::Vulkan || vulkanOutput == nullptr || frame == nullptr || scale < 1)
    {
        clearScreenshot();
        return false;
    }

    const size_t readbackPixels = static_cast<size_t>(frame->width) * static_cast<size_t>(frame->height);
    if (readbackPixels == 0)
    {
        clearScreenshot();
        return false;
    }

    vulkanReadbackFrame.resize(readbackPixels);
    auto& renderer3D = static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D());
    VulkanCompositionInputs compositionInputs{};
    if (!vulkanOutput->buildCompositionInputs(
            frame,
            renderer3D,
            scale,
            true,
            false,
            false,
            compositionInputs)
        || !vulkanOutput->composeAndSubmitFrame(frame, compositionInputs)
        || !vulkanOutput->readFramePixels(frame, vulkanReadbackFrame.data(), vulkanReadbackFrame.size()))
    {
        clearScreenshot();
        Platform::Log(Platform::LogLevel::Error, "Failed to readback Vulkan composited frame for screenshot");
        return false;
    }

    const bool copied = CopyCompositedFrameToScreenshot(
        vulkanReadbackFrame.data(),
        static_cast<int>(frame->width),
        static_cast<int>(frame->height),
        scale,
        screenshotRenderer.getScreenshot(),
        screenshotPixelCount
    );
    if (!copied)
    {
        clearScreenshot();
        Platform::Log(Platform::LogLevel::Error, "Failed to downscale Vulkan composited frame for screenshot");
        return false;
    }

    return true;
}

void MelonInstance::logVulkanPerformanceIfNeeded()
{
    if (!areRendererDebugToolsEnabled())
        return;

    if (!vulkanRunFrameCpuWindow.Ready())
        return;

    const PerfSampleWindow<120>::Summary runFrameSummary = vulkanRunFrameCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary setupSummary = vulkanSetupCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary ndsRunSummary = vulkanNdsRunCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary postRunSummary = vulkanPostRunCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary composeSummary = vulkanComposeCpuWindow.SummarizeAndReset();
    const FrameQueueStats queueStats = frameQueue.takeStatsSnapshotAndReset();
    const VulkanPresenterPacingStats presenterStats = vulkanSurfacePresenter
        ? vulkanSurfacePresenter->takePacingStatsSnapshotAndReset()
        : VulkanPresenterPacingStats{};
    int vulkanOutputScale = 1;
    int vulkanRenderScale = 1;
    if (currentRenderer == Renderer::Vulkan)
    {
        auto& vulkanRenderSettings = static_cast<VulkanRenderSettings&>(*currentConfiguration->renderSettings);
        vulkanOutputScale = getConfiguredVulkanScale(vulkanRenderSettings);
        vulkanRenderScale = std::max(static_cast<VulkanRenderer3D&>(nds->GPU.GetRenderer3D()).GetScaleFactor(), 1);
    }
    const double presentedFrameAgeAvgMs = queueStats.PresentedFrameAgeSamples > 0
        ? PerfNsToMs(queueStats.PresentedFrameAgeTotalNs / queueStats.PresentedFrameAgeSamples)
        : 0.0;
    const double droppedFrameAgeAvgMs = queueStats.DroppedFrameAgeSamples > 0
        ? PerfNsToMs(queueStats.DroppedFrameAgeTotalNs / queueStats.DroppedFrameAgeSamples)
        : 0.0;

    Platform::Log(
        Platform::LogLevel::Warn,
        "VulkanPerf[Instance]: run cpu avg=%.3fms p95=%.3fms max=%.3fms compose avg=%.3fms p95=%.3fms max=%.3fms queue queued=%llu discarded=%llu presented=%llu staleDropped=%llu reusedPrev=%llu stolen=%llu renderDropped=%llu presentDropped=%llu backlog=%llu/%llu dropCause(stale=%llu steal=%llu deadline=%llu backlogTrim=%llu deferred=%llu) ageMs(present avg=%.3f max=%.3f drop avg=%.3f max=%.3f)",
        PerfNsToMs(runFrameSummary.MeanNs),
        PerfNsToMs(runFrameSummary.P95Ns),
        PerfNsToMs(runFrameSummary.MaxNs),
        PerfNsToMs(composeSummary.MeanNs),
        PerfNsToMs(composeSummary.P95Ns),
        PerfNsToMs(composeSummary.MaxNs),
        static_cast<unsigned long long>(queueStats.RenderFramesQueued),
        static_cast<unsigned long long>(queueStats.RenderFramesDiscarded),
        static_cast<unsigned long long>(queueStats.PresentFramesReturned),
        static_cast<unsigned long long>(queueStats.StaleFramesDropped),
        static_cast<unsigned long long>(queueStats.PreviousFrameReused),
        static_cast<unsigned long long>(queueStats.PendingFramesStolenForRender),
        static_cast<unsigned long long>(queueStats.RenderFramesDroppedByPolicy),
        static_cast<unsigned long long>(queueStats.PresentFramesDroppedByPolicy),
        static_cast<unsigned long long>(queueStats.CurrentBacklogDepth),
        static_cast<unsigned long long>(queueStats.MaxBacklogDepth),
        static_cast<unsigned long long>(queueStats.PresentDroppedByStale),
        static_cast<unsigned long long>(queueStats.PresentDroppedBySteal),
        static_cast<unsigned long long>(queueStats.PresentDroppedByDeadline),
        static_cast<unsigned long long>(queueStats.PresentDroppedByBacklogTrim),
        static_cast<unsigned long long>(queueStats.PresentDeferredByDeadline),
        presentedFrameAgeAvgMs,
        PerfNsToMs(queueStats.PresentedFrameAgeMaxNs),
        droppedFrameAgeAvgMs,
        PerfNsToMs(queueStats.DroppedFrameAgeMaxNs)
    );
    Platform::Log(
        Platform::LogLevel::Warn,
        "VulkanPerf[InstancePhases]: setup cpu avg=%.3fms p95=%.3fms max=%.3fms nds cpu avg=%.3fms p95=%.3fms max=%.3fms post cpu avg=%.3fms p95=%.3fms max=%.3fms",
        PerfNsToMs(setupSummary.MeanNs),
        PerfNsToMs(setupSummary.P95Ns),
        PerfNsToMs(setupSummary.MaxNs),
        PerfNsToMs(ndsRunSummary.MeanNs),
        PerfNsToMs(ndsRunSummary.P95Ns),
        PerfNsToMs(ndsRunSummary.MaxNs),
        PerfNsToMs(postRunSummary.MeanNs),
        PerfNsToMs(postRunSummary.P95Ns),
        PerfNsToMs(postRunSummary.MaxNs)
    );
    Platform::Log(
        Platform::LogLevel::Warn,
        "VulkanPerf[Pacing]: mode=%s acquireTimeouts=%llu presentDropped=%llu renderDropped=%llu backlog=%llu/%llu reusedPrev=%llu stolen=%llu skippedWait=%llu presented=%llu direct=%llu fallback=%llu recoveries=%llu presentMode=%d swapchainImages=%u renderScale=%d outputScale=%d dropCause(stale=%llu steal=%llu deadline=%llu backlogTrim=%llu deferred=%llu) ageMs(present avg=%.3f max=%.3f drop avg=%.3f max=%.3f)",
        isFastForwardActive() ? "ff" : "realtime",
        static_cast<unsigned long long>(presenterStats.AcquireTimeouts),
        static_cast<unsigned long long>(queueStats.PresentFramesDroppedByPolicy),
        static_cast<unsigned long long>(queueStats.RenderFramesDroppedByPolicy),
        static_cast<unsigned long long>(queueStats.CurrentBacklogDepth),
        static_cast<unsigned long long>(queueStats.MaxBacklogDepth),
        static_cast<unsigned long long>(queueStats.PreviousFrameReused),
        static_cast<unsigned long long>(queueStats.PendingFramesStolenForRender),
        static_cast<unsigned long long>(presenterStats.SurfaceWaitTimeouts),
        static_cast<unsigned long long>(presenterStats.PresentedFrames),
        static_cast<unsigned long long>(presenterStats.DirectPresentedFrames),
        static_cast<unsigned long long>(presenterStats.FallbackPresentedFrames),
        static_cast<unsigned long long>(presenterStats.SwapchainRecoveries),
        static_cast<int>(presenterStats.PresentMode),
        presenterStats.SwapchainImageCount,
        vulkanRenderScale,
        vulkanOutputScale,
        static_cast<unsigned long long>(queueStats.PresentDroppedByStale),
        static_cast<unsigned long long>(queueStats.PresentDroppedBySteal),
        static_cast<unsigned long long>(queueStats.PresentDroppedByDeadline),
        static_cast<unsigned long long>(queueStats.PresentDroppedByBacklogTrim),
        static_cast<unsigned long long>(queueStats.PresentDeferredByDeadline),
        presentedFrameAgeAvgMs,
        PerfNsToMs(queueStats.PresentedFrameAgeMaxNs),
        droppedFrameAgeAvgMs,
        PerfNsToMs(queueStats.DroppedFrameAgeMaxNs)
    );
}

void MelonInstance::saveRewindState(RewindSaveState* rewindSaveState)
{
    Savestate* savestate = new Savestate(rewindSaveState->buffer, rewindSaveState->bufferSize, true);
    if (saveState(savestate, false))
    {
        rewindSaveState->bufferContentSize = savestate->Length();
        memcpy(rewindSaveState->screenshot, screenshotRenderer.getScreenshot(), rewindSaveState->screenshotSize);
    }

    delete savestate;
}

}
