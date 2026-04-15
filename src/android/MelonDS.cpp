#include <cstring>
#include <cstdlib>
#include <utility>
#include <atomic>
#include <cctype>
#include <android/asset_manager.h>
#include <sys/system_properties.h>
#include <oboe/Oboe.h>
#include "EmulatorArgsBuilder.h"
#include "MelonDS.h"
#include "MelonDSAudio.h"
#include "OboeCallback.h"
#include "MicInputOboeCallback.h"
#include "OpenGLContext.h"
#include "mic_blow.h"
#include "NDS.h"
#include "GPU.h"
#include "GPU3D.h"
#include "GBACart.h"
#include "SPU.h"
#include "Platform.h"
#include "Savestate.h"
#include "MelonInstance.h"
#include "RewindManager.h"
#include "ROMManager.h"
#include "MPInterface.h"
#include "AndroidCameraHandler.h"
#include "renderer/ScreenshotRenderer.h"
#include "renderer/FrameQueue.h"
#include "retroachievements/RetroAchievementsManager.h"
#include "net/Net.h"
#include "net/Net_Slirp.h"
#include <fstream>

namespace MelonDSAndroid
{
    namespace
    {
        bool fastForwardActive = false;
        std::atomic_bool rendererDebugToolsEnabled = false;
        std::atomic_uint vulkanDiagnosticFlags = 0;

        bool EqualsIgnoreCase(const char* lhs, const char* rhs)
        {
            if (lhs == nullptr || rhs == nullptr)
                return false;

            while (*lhs != '\0' && *rhs != '\0')
            {
                if (std::tolower(static_cast<unsigned char>(*lhs)) != std::tolower(static_cast<unsigned char>(*rhs)))
                    return false;
                lhs++;
                rhs++;
            }

            return *lhs == '\0' && *rhs == '\0';
        }

        bool ReadBooleanSystemProperty(const char* key, bool defaultValue = false)
        {
            char value[PROP_VALUE_MAX]{};
            const int length = __system_property_get(key, value);
            if (length <= 0)
                return defaultValue;

            if (EqualsIgnoreCase(value, "1")
                || EqualsIgnoreCase(value, "true")
                || EqualsIgnoreCase(value, "y")
                || EqualsIgnoreCase(value, "yes")
                || EqualsIgnoreCase(value, "on"))
            {
                return true;
            }

            if (EqualsIgnoreCase(value, "0")
                || EqualsIgnoreCase(value, "false")
                || EqualsIgnoreCase(value, "n")
                || EqualsIgnoreCase(value, "no")
                || EqualsIgnoreCase(value, "off"))
            {
                return false;
            }

            return defaultValue;
        }

        u32 ResolveVulkanDiagnosticFlags()
        {
            u32 flags = 0;

            if (ReadBooleanSystemProperty("debug.melonds.vulkan.disable_passive_repeat_expand"))
                flags |= VulkanDiagnosticDisablePassiveRepeatCoverageExpand;
            if (ReadBooleanSystemProperty("debug.melonds.vulkan.legacy_compat_fill_depth"))
                flags |= VulkanDiagnosticLegacyCompatFillDepth;
            if (ReadBooleanSystemProperty("debug.melonds.vulkan.legacy_final_aa_mask"))
                flags |= VulkanDiagnosticLegacyFinalAaMask;

            return flags;
        }

        bool ResolveRendererDebugToolsEnabled(const EmulatorConfiguration& configuration)
        {
            if (!configuration.renderSettings)
                return false;

            switch (configuration.renderer)
            {
                case Renderer::Software:
                    return static_cast<const SoftwareRenderSettings&>(*configuration.renderSettings).rendererDebugToolsEnabled;
                case Renderer::OpenGl:
                    return static_cast<const OpenGlRenderSettings&>(*configuration.renderSettings).rendererDebugToolsEnabled;
                case Renderer::Vulkan:
                    return static_cast<const VulkanRenderSettings&>(*configuration.renderSettings).rendererDebugToolsEnabled;
                case Renderer::Compute:
                    return false;
            }

            return false;
        }

        void LogEffectiveJitConfiguration(const EmulatorConfiguration& configuration, const NDSArgs& args)
        {
            if (configuration.renderer != Renderer::Vulkan)
                return;

#ifdef JIT_ENABLED
            JITArgs effectiveJit{};
            effectiveJit.MaxBlockSize = 32;
            effectiveJit.LiteralOptimizations = true;
            effectiveJit.BranchOptimizations = true;
            effectiveJit.FastMemory = true;
            effectiveJit.HgEngineFix = configuration.hgEngineFixEnabled;

            const bool jitEnabled = args.JIT.has_value();
            if (jitEnabled)
                effectiveJit = *args.JIT;

            Platform::Log(
                Platform::LogLevel::Warn,
                "VulkanRuntime[JIT]: enabled=%d maxBlockSize=%u literalOpt=%d branchOpt=%d fastMemory=%d hgEngineFix=%d",
                jitEnabled ? 1 : 0,
                effectiveJit.MaxBlockSize,
                effectiveJit.LiteralOptimizations ? 1 : 0,
                effectiveJit.BranchOptimizations ? 1 : 0,
                effectiveJit.FastMemory ? 1 : 0,
                effectiveJit.HgEngineFix ? 1 : 0
            );
#else
            Platform::Log(
                Platform::LogLevel::Warn,
                "VulkanRuntime[JIT]: enabled=0 compiled=0 maxBlockSize=32 literalOpt=1 branchOpt=1 fastMemory=1 hgEngineFix=%d",
                configuration.hgEngineFixEnabled ? 1 : 0
            );
#endif
        }

        void ReleaseConfigurationPath(char*& path)
        {
            if (path != nullptr)
            {
                std::free(path);
                path = nullptr;
            }
        }

        void DestroyEmulatorConfiguration(MelonDSAndroid::EmulatorConfiguration* configuration)
        {
            if (configuration == nullptr)
                return;

            ReleaseConfigurationPath(configuration->dsBios7Path);
            ReleaseConfigurationPath(configuration->dsBios9Path);
            ReleaseConfigurationPath(configuration->dsFirmwarePath);
            ReleaseConfigurationPath(configuration->dsiBios7Path);
            ReleaseConfigurationPath(configuration->dsiBios9Path);
            ReleaseConfigurationPath(configuration->dsiFirmwarePath);
            ReleaseConfigurationPath(configuration->dsiNandPath);
            ReleaseConfigurationPath(configuration->internalFilesDir);
            ReleaseConfigurationPath(configuration->dsiSdCardSettings.imagePath);
            ReleaseConfigurationPath(configuration->dsiSdCardSettings.folderPath);
            ReleaseConfigurationPath(configuration->dldiSdCardSettings.imagePath);
            ReleaseConfigurationPath(configuration->dldiSdCardSettings.folderPath);
            delete configuration;
        }

        std::shared_ptr<MelonDSAndroid::EmulatorConfiguration> ShareConfiguration(MelonDSAndroid::EmulatorConfiguration configuration)
        {
            return std::shared_ptr<MelonDSAndroid::EmulatorConfiguration>(
                new MelonDSAndroid::EmulatorConfiguration(std::move(configuration)),
                [](MelonDSAndroid::EmulatorConfiguration* config) {
                    DestroyEmulatorConfiguration(config);
                }
            );
        }

        std::shared_ptr<MelonDSAndroid::EmulatorConfiguration> ShareConfiguration(std::unique_ptr<MelonDSAndroid::EmulatorConfiguration> configuration)
        {
            MelonDSAndroid::EmulatorConfiguration* rawConfiguration = configuration.release();
            return std::shared_ptr<MelonDSAndroid::EmulatorConfiguration>(
                rawConfiguration,
                [](MelonDSAndroid::EmulatorConfiguration* config) {
                    DestroyEmulatorConfiguration(config);
                }
            );
        }
    }
    OpenGLContext *openGlContext;
    AndroidFileHandler* fileHandler;
    AndroidCameraHandler* cameraHandler;
    std::string internalFilesDir;
    std::shared_ptr<MelonEventMessenger> eventMessenger;
    std::shared_ptr<EmulatorConfiguration> currentConfiguration;
    std::shared_ptr<Net> net;

    std::shared_ptr<MelonInstance> instance;

    bool ensureOpenGlContext()
    {
        if (openGlContext != nullptr)
            return true;

        auto* context = new OpenGLContext();
        if (!context->InitContext(0))
        {
            delete context;
            Platform::Log(Platform::LogLevel::Error, "Failed to initialize OpenGL context");
            return false;
        }

        openGlContext = context;
        return true;
    }

    bool setupOpenGlContext();
    void cleanupOpenGlContext();

    /**
     * Used to set the emulator's initial configuration, before boot. To update the configuration during runtime, use @updateEmulatorConfiguration.
     *
     * @param emulatorConfiguration The emulator configuration during the next emulator run
     */
    void setConfiguration(EmulatorConfiguration emulatorConfiguration) {
        currentConfiguration = ShareConfiguration(std::move(emulatorConfiguration));
        internalFilesDir = currentConfiguration->internalFilesDir;
        rendererDebugToolsEnabled.store(ResolveRendererDebugToolsEnabled(*currentConfiguration), std::memory_order_relaxed);
        vulkanDiagnosticFlags.store(ResolveVulkanDiagnosticFlags(), std::memory_order_relaxed);

        net = std::make_shared<Net>();
        net->SetDriver(std::make_unique<Net_Slirp>([](const u8* data, int len) {
            net->RXEnqueue(data, len);
        }));
    }

    void setup(AndroidCameraHandler* androidCameraHandler, std::shared_ptr<MelonEventMessenger> androidEventMessenger, u32* screenshotBufferPointer, int instanceId)
    {
        cameraHandler = androidCameraHandler;
        eventMessenger = androidEventMessenger;
        RetroAchievements::RetroAchievementsManager::EventMessenger = androidEventMessenger;

        auto instanceArgs = BuildArgsFromConfiguration(*currentConfiguration, instanceId);
        if (!instanceArgs.has_value())
        {
            // TODO: Handle this somehow?
            instance = nullptr;
            return;
        }

        auto args = std::move(instanceArgs.value());
        LogEffectiveJitConfiguration(*currentConfiguration, *args);
        instance = std::make_shared<MelonInstance>(
            instanceId,
            currentConfiguration,
            std::move(args),
            net,
            ScreenshotRenderer(screenshotBufferPointer),
            currentConfiguration->consoleType
        );

        setupAudio(currentConfiguration->audioSettings);
        setAudioActiveInstance(instance);
    }

    void setCodeList(std::list<Cheat> cheats)
    {
        if (instance == nullptr)
            return;
        instance->loadCheats(std::move(cheats));
    }

    void setupAchievements(
        std::list<RetroAchievements::RAAchievement> achievements,
        std::list<RetroAchievements::RALeaderboard> leaderboards,
        std::optional<std::string> richPresenceScript,
        std::optional<RetroAchievements::RARuntimeBridgeConfig> runtimeBridgeConfig
    )
    {
        if (instance == nullptr)
            return;
        instance->setupAchievements(
            std::move(achievements),
            std::move(leaderboards),
            std::move(richPresenceScript),
            std::move(runtimeBridgeConfig)
        );
    }

    void unloadRetroAchievementsData()
    {
        if (instance == nullptr)
            return;
        instance->unloadRetroAchievementsData();
    }

    std::string getRichPresenceStatus()
    {
        if (instance == nullptr)
            return "";
        return instance->getRichPresenceStatus();
    }

    std::vector<RetroAchievements::RARuntimeAchievement> getRuntimeAchievements()
    {
        if (instance == nullptr)
            return { };
        return instance->getRuntimeAchievements();
    }

    std::vector<RetroAchievements::RARuntimeAchievementBucketEntry> getRuntimeAchievementBuckets()
    {
        if (instance == nullptr)
            return { };
        return instance->getRuntimeAchievementBuckets();
    }

    std::vector<long> getRuntimeSubsetIds()
    {
        if (instance == nullptr)
            return { };
        return instance->getRuntimeSubsetIds();
    }

    Renderer getCurrentRenderer()
    {
        if (instance != nullptr)
            return instance->getCurrentRenderer();

        if (currentConfiguration != nullptr)
            return currentConfiguration->renderer;

        return Renderer::Software;
    }

    /**
     * Used to update the emulator's configuration during runtime. Will only update the configurations that can actually change during runtime without causing issues,
     *
     * @param emulatorConfiguration The new emulator configuration
     */
    void updateEmulatorConfiguration(std::unique_ptr<EmulatorConfiguration> emulatorConfiguration) {
        std::shared_ptr<EmulatorConfiguration> sharedConfig = ShareConfiguration(std::move(emulatorConfiguration));
        instance->updateConfiguration(sharedConfig);
        updateAudioSettings(sharedConfig->audioSettings);

        currentConfiguration = sharedConfig;
        rendererDebugToolsEnabled.store(ResolveRendererDebugToolsEnabled(*sharedConfig), std::memory_order_relaxed);
        vulkanDiagnosticFlags.store(ResolveVulkanDiagnosticFlags(), std::memory_order_relaxed);
    }

    int loadRom(std::string romPath, std::string sramPath, RomGbaSlotConfig* gbaSlotConfig)
    {
        if (!instance->loadRom(std::move(romPath), std::move(sramPath)))
            return 2;

        if (gbaSlotConfig->type == GBA_ROM)
        {
            RomGbaSlotConfigGbaRom* gbaRomConfig = (RomGbaSlotConfigGbaRom*) gbaSlotConfig;
            if (!instance->loadGbaRom(gbaRomConfig->romPath, gbaRomConfig->savePath))
                return 1;
        }
        else if (gbaSlotConfig->type == RUMBLE_PAK)
        {
            instance->loadRumblePak();
        }
        else if (gbaSlotConfig->type == MEMORY_EXPANSION)
        {
            instance->loadGbaMemoryExpansion();
        }
        else if (gbaSlotConfig->type == ANALOG_INPUT)
        {
            Platform::Log(Platform::LogLevel::Warn, "MelonDSAndroid: enabling Slot-2 analog input addon\n");
            instance->loadGbaAnalogInput();
        }

        return 0;
    }

    int bootFirmware()
    {
        // TODO: Maybe validate BIOS and firmware?
        if (instance->bootFirmware())
            return ROMManager::SUCCESS;
        else
            return ROMManager::FIRMWARE_NOT_BOOTABLE;
    }

    bool precompileVulkanPipelines()
    {
        if (!instance)
            return false;

        return instance->precompileVulkanPipelines();
    }

    void touchScreen(u16 x, u16 y)
    {
        if (instance)
            instance->touchScreen(x, y);
    }

    void releaseScreen()
    {
        if (instance)
            instance->releaseScreen();
    }

    void pressKey(u32 key)
    {
        if (instance)
            instance->pressKey(key);
    }

    void releaseKey(u32 key)
    {
        if (instance)
            instance->releaseKey(key);
    }

    void setSlot2AnalogInput(float x, float y)
    {
        if (instance)
            instance->setSlot2AnalogInput(x, y);
    }

    void start()
    {
        startAudio();
        if (currentConfiguration->renderer != Renderer::Vulkan)
            setupOpenGlContext();

        instance->start();
    }

    u32 loop()
    {
        MPInterface::Get().Process();
        if (currentConfiguration != nullptr && currentConfiguration->renderer != Renderer::Vulkan)
            setupOpenGlContext();
        return instance->runFrame();
    }

    Frame* getPresentationFrame(std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline)
    {
        if (!instance)
            return nullptr;

        return instance->getPresentationFrame(deadline);
    }

    bool waitForPresentationFrame(Frame* frame, u64 timeoutNs)
    {
        if (!instance)
            return false;

        return instance->waitForPresentationFrame(frame, timeoutNs);
    }

    int attachVulkanSurface(ANativeWindow* window, u32 width, u32 height)
    {
        if (!instance)
        {
            if (window != nullptr)
                ANativeWindow_release(window);
            return 0;
        }

        return instance->attachVulkanSurface(window, width, height);
    }

    bool resizeVulkanSurface(int surfaceId, u32 width, u32 height)
    {
        if (!instance)
            return false;

        return instance->resizeVulkanSurface(surfaceId, width, height);
    }

    bool configureVulkanSurface(int surfaceId, const VulkanSurfaceConfig& config, const VulkanBackgroundImage& backgroundImage)
    {
        if (!instance)
            return false;

        return instance->configureVulkanSurface(surfaceId, config, backgroundImage);
    }

    void detachVulkanSurface(int surfaceId)
    {
        if (instance)
            instance->detachVulkanSurface(surfaceId);
    }

    bool presentVulkanFrame(
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline,
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> budgetDeadline)
    {
        if (!instance)
            return false;

        return instance->presentVulkanFrame(deadline, budgetDeadline);
    }

    void requestVulkanPresentationResync()
    {
        if (instance)
            instance->requestVulkanPresentationResync();
    }

    bool areRendererDebugToolsEnabled()
    {
        return rendererDebugToolsEnabled.load(std::memory_order_relaxed);
    }

    u32 getVulkanDiagnosticFlags()
    {
        return vulkanDiagnosticFlags.load(std::memory_order_relaxed);
    }

    bool hasVulkanDiagnosticFlag(VulkanDiagnosticFlag flag)
    {
        return (getVulkanDiagnosticFlags() & static_cast<u32>(flag)) != 0u;
    }

    std::vector<u32> captureCurrentFrameForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrentFrameForDebug();
    }

    std::vector<u32> captureCurrentPackedTopPrimaryForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrentPackedTopPrimaryForDebug();
    }

    std::vector<u32> captureCurrentPackedBottomPrimaryForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrentPackedBottomPrimaryForDebug();
    }

    std::vector<u32> captureCurrent3dDimensionsForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrent3dDimensionsForDebug();
    }

    std::vector<u32> captureCurrent3dFrameForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrent3dFrameForDebug();
    }

    std::vector<u32> captureCurrent3dCaptureFrameForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrent3dCaptureFrameForDebug();
    }

    std::vector<u32> captureCurrent3dDepthForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrent3dDepthForDebug();
    }

    std::vector<u32> captureCurrent3dAttrForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrent3dAttrForDebug();
    }

    std::vector<u32> captureCurrent3dCoverageForDebug()
    {
        if (!instance)
            return {};

        return instance->captureCurrent3dCoverageForDebug();
    }

    void dumpCurrentRendererDebugSnapshot()
    {
        if (instance)
            instance->dumpDebugSnapshot();
    }

    void setFastForwardActive(bool enabled)
    {
        fastForwardActive = enabled;
    }

    bool isFastForwardActive()
    {
        return fastForwardActive;
    }

    void pause()
    {
        pauseAudio();
    }

    void resume()
    {
        startAudio();
    }

    void reset()
    {
        instance->reset();
    }

    bool saveState(const char* path)
    {
        Platform::FileHandle* saveStateFile = Platform::OpenFile(path, Platform::FileMode::Write);

        if (!saveStateFile)
            return false;

        Savestate state;
        if (state.Error)
        {
            Platform::CloseFile(saveStateFile);
            return false;
        }

        instance->saveState(&state, true);

        if (state.Error)
        {
            Platform::CloseFile(saveStateFile);
            return false;
        }

        if (Platform::FileWrite(state.Buffer(), state.Length(), 1, saveStateFile) == 0)
        {
            Platform::Log(Platform::Error, "Failed to write %d-byte savestate to %s\n", state.Length(), path);
            Platform::CloseFile(saveStateFile);
            return false;
        }

        Platform::CloseFile(saveStateFile);
        return true;
    }

    bool loadState(const char* path)
    {
        if (!instance->areSaveStatesAllowed())
        {
            Platform::Log(Platform::LogLevel::Warn, "Savestate load denied: RetroAchievements hardcore session active\n");
            return false;
        }

        auto saveStateFile = Platform::OpenFile(path, Platform::FileMode::Read);
        if (!saveStateFile)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to open state file \"%s\"\n", path);
            return false;
        }

        std::unique_ptr<Savestate> backup = std::make_unique<Savestate>(Savestate::DEFAULT_SIZE);
        if (backup->Error)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to allocate memory for state backup\n");
            Platform::CloseFile(saveStateFile);
            return false;
        }

        if (!instance->saveState(backup.get(), false) || backup->Error)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to back up state, aborting load (from \"%s\")\n", path);
            Platform::CloseFile(saveStateFile);
            return false;
        }

        size_t size = Platform::FileLength(saveStateFile);

        // Allocate exactly as much memory as we need for the savestate
        std::vector<u8> buffer(size);
        if (Platform::FileRead(buffer.data(), size, 1, saveStateFile) == 0)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to read %u-byte state file \"%s\"\n", size, path);
            Platform::CloseFile(saveStateFile);
            return false;
        }
        Platform::CloseFile(saveStateFile);

        std::unique_ptr<Savestate> state = std::make_unique<Savestate>(buffer.data(), size, false);

        if (!instance->loadState(state.get()) || state->Error)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to load state file \"%s\" into emulator\n", path);
            // Restore backup
            if (!instance->loadState(backup.get()) || state->Error)
                Platform::Log(Platform::LogLevel::Error, "Failed to load backup state\n", path);
            else
                Platform::Log(Platform::LogLevel::Info, "Backup state loaded\n", path);

            return false;
        }
        return true;
    }

    bool loadRewindState(melonDS::RewindSaveState rewindSaveState)
    {
        if (!instance->areSaveStatesAllowed())
        {
            Platform::Log(Platform::LogLevel::Warn, "Rewind load denied: RetroAchievements hardcore session active\n");
            return false;
        }

        std::unique_ptr<Savestate> backup = std::make_unique<Savestate>(Savestate::DEFAULT_SIZE);
        if (backup->Error)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to allocate memory for state backup");
            return false;
        }

        if (!instance->saveState(backup.get(), false) || backup->Error)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to back up state, aborting rewind state load");
            return false;
        }

        bool result = instance->loadRewindState(rewindSaveState);
        if (!result)
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to load rewind state");
            // Restore backup
            if (!instance->loadState(backup.get()) || backup->Error)
                Platform::Log(Platform::LogLevel::Error, "Failed to load backup state");
            else
                Platform::Log(Platform::LogLevel::Info, "Backup state loaded");
        }

        return result;
    }

    RewindWindow getRewindWindow()
    {
        return instance->getRewindWindow();
    }

    void stop()
    {
        instance->stop();
        cleanupOpenGlContext();
    }

    void cleanup()
    {
        cleanupAudio();

        instance = nullptr;
        eventMessenger = nullptr;
    }

    bool setupOpenGlContext()
    {
        if (!ensureOpenGlContext())
            return false;

        if (!openGlContext->Use())
        {
            Platform::Log(Platform::LogLevel::Error, "Failed to use OpenGL context");
            return false;
        }

        return true;
    }

    void cleanupOpenGlContext()
    {
        if (openGlContext == nullptr)
            return;

        openGlContext->Release();
    }
}
