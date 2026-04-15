#ifndef MELONDS_MELONDS_H
#define MELONDS_MELONDS_H

#include <list>
#include <vector>
#include <optional>
#include <android/native_window.h>
#include "AndroidFileHandler.h"
#include "AndroidCameraHandler.h"
#include "Configuration.h"
#include "MelonEventMessenger.h"
#include "RewindManager.h"
#include "RomGbaSlotConfig.h"
#include "retroachievements/RAAchievement.h"
#include "retroachievements/RALeaderboard.h"
#include "retroachievements/RARuntimeBridgeConfig.h"
#include "renderer/FrameQueue.h"
#include "types.h"
#include "../GPU.h"
#include "renderer/VulkanSurfacePresenter.h"
#include <android/asset_manager.h>

using namespace melonDS;

namespace MelonDSAndroid {
    typedef struct {
        std::vector<u32> code;
    } Cheat;

    enum VulkanDiagnosticFlag : u32 {
        VulkanDiagnosticDisablePassiveRepeatCoverageExpand = 1u << 0u,
        VulkanDiagnosticLegacyCompatFillDepth = 1u << 1u,
        VulkanDiagnosticLegacyFinalAaMask = 1u << 2u,
    };

    typedef enum {
        ROM,
        FIRMWARE
    } RunMode;

    extern OpenGLContext *openGlContext;
    extern AndroidFileHandler* fileHandler;
    extern AndroidCameraHandler* cameraHandler;
    extern std::string internalFilesDir;
    extern std::shared_ptr<MelonEventMessenger> eventMessenger;
    extern bool ensureOpenGlContext();

    extern void setConfiguration(EmulatorConfiguration emulatorConfiguration);
    extern void setup(AndroidCameraHandler* androidCameraHandler, std::shared_ptr<MelonEventMessenger> androidEventMessenger, u32* screenshotBufferPointer, int instanceId);
    extern void setCodeList(std::list<Cheat> cheats);
    extern void setupAchievements(
        std::list<RetroAchievements::RAAchievement> achievements,
        std::list<RetroAchievements::RALeaderboard> leaderboards,
        std::optional<std::string> richPresenceScript,
        std::optional<RetroAchievements::RARuntimeBridgeConfig> runtimeBridgeConfig
    );
    extern void unloadRetroAchievementsData();
    extern std::string getRichPresenceStatus();
    extern std::vector<RetroAchievements::RARuntimeAchievement> getRuntimeAchievements();
    extern std::vector<RetroAchievements::RARuntimeAchievementBucketEntry> getRuntimeAchievementBuckets();
    extern std::vector<long> getRuntimeSubsetIds();
    extern Renderer getCurrentRenderer();
    extern void updateEmulatorConfiguration(std::unique_ptr<EmulatorConfiguration> emulatorConfiguration);

    /**
     * Loads the NDS ROM and, optionally, the GBA ROM.
     *
     * @param romPath The path to the NDS rom
     * @param sramPath The path to the rom's SRAM file
     * @param gbaSlotConfig The config to be used for the GBA slot
     * @return The load result. 0 if everything was loaded successfully, 1 if the NDS ROM was loaded but the GBA ROM
     * failed to load, 2 if the NDS ROM failed to load
     */
    extern int loadRom(std::string romPath, std::string sramPath, RomGbaSlotConfig* gbaSlotConfig);
    extern int bootFirmware();
    extern bool precompileVulkanPipelines();
    extern void touchScreen(u16 x, u16 y);
    extern void releaseScreen();
    extern void pressKey(u32 key);
    extern void releaseKey(u32 key);
    extern void setSlot2AnalogInput(float x, float y);
    extern void start();
    extern u32 loop();
    extern Frame* getPresentationFrame(std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline);
    extern bool waitForPresentationFrame(Frame* frame, u64 timeoutNs);
    extern int attachVulkanSurface(ANativeWindow* window, u32 width, u32 height);
    extern bool resizeVulkanSurface(int surfaceId, u32 width, u32 height);
    extern bool configureVulkanSurface(int surfaceId, const VulkanSurfaceConfig& config, const VulkanBackgroundImage& backgroundImage);
    extern void detachVulkanSurface(int surfaceId);
    extern bool presentVulkanFrame(
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline,
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> budgetDeadline);
    extern void requestVulkanPresentationResync();
    extern bool areRendererDebugToolsEnabled();
    extern u32 getVulkanDiagnosticFlags();
    extern bool hasVulkanDiagnosticFlag(VulkanDiagnosticFlag flag);
    extern std::vector<u32> captureCurrentFrameForDebug();
    extern std::vector<u32> captureCurrentPackedTopPrimaryForDebug();
    extern std::vector<u32> captureCurrentPackedBottomPrimaryForDebug();
    extern std::vector<u32> captureCurrent3dDimensionsForDebug();
    extern std::vector<u32> captureCurrent3dFrameForDebug();
    extern std::vector<u32> captureCurrent3dCaptureFrameForDebug();
    extern std::vector<u32> captureCurrent3dDepthForDebug();
    extern std::vector<u32> captureCurrent3dAttrForDebug();
    extern std::vector<u32> captureCurrent3dCoverageForDebug();
    extern void dumpCurrentRendererDebugSnapshot();
    extern void setFastForwardActive(bool enabled);
    extern bool isFastForwardActive();
    extern void pause();
    extern void resume();
    extern void reset();
    extern bool saveState(const char* path);
    extern bool loadState(const char* path);
    extern bool loadRewindState(melonDS::RewindSaveState rewindSaveState);
    extern RewindWindow getRewindWindow();
    extern void stop();
    extern void cleanup();
}

#endif //MELONDS_MELONDS_H
