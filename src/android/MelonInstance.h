#ifndef MELONINSTANCE_H
#define MELONINSTANCE_H

#include <string>
#include <atomic>
#include "Args.h"
#include "Configuration.h"
#include "NDS.h"
#include "MelonDS.h"
#include "SaveManager.h"
#include "VulkanPerfStats.h"
#include "frontend/RewindManager.h"
#include "renderer/FrameQueue.h"
#include "renderer/Renderer.h"
#include "renderer/ScreenshotRenderer.h"
#include "renderer/VulkanOutput.h"
#include "renderer/VulkanSurfacePresenter.h"
#include "retroachievements/RetroAchievementsManager.h"
#include "net/Net.h"

using namespace melonDS;

namespace MelonDSAndroid
{

class MelonInstance
{

public:
    MelonInstance(int instanceId, std::shared_ptr<EmulatorConfiguration> configuration, std::unique_ptr<melonDS::NDSArgs> args, std::shared_ptr<Net> net, ScreenshotRenderer screenshotRenderer, int consoleType);
    ~MelonInstance();

    int getInstanceId() { return instanceId; };
    Renderer getCurrentRenderer() const { return currentRenderer; }

    bool loadRom(std::string romPath, std::string sramPath);
    bool loadGbaRom(std::string romPath, std::string sramPath);
    void loadRumblePak();
    void loadGbaMemoryExpansion();
    void loadGbaAnalogInput();
    void loadGbaRumblePak();
    bool bootFirmware();
    bool precompileVulkanPipelines();
    void start();
    void reset();
    melonDS::u32 runFrame();
    void stop();

    void touchScreen(u16 x, u16 y);
    void releaseScreen();
    void pressKey(u32 key);
    void releaseKey(u32 key);
    void setSlot2AnalogInput(float x, float y);
    int readAudioOutput(s16* buffer, int length);
    void setAudioOutputSkew(double skew);
    void loadCheats(std::list<Cheat> cheats);
    int sendNetPacket(u8* data, int length);
    int receiveNetPacket(u8* data);

    Frame* getPresentationFrame(std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline);
    bool waitForPresentationFrame(Frame* frame, u64 timeoutNs);
    int attachVulkanSurface(ANativeWindow* window, u32 width, u32 height);
    bool resizeVulkanSurface(int surfaceId, u32 width, u32 height);
    bool configureVulkanSurface(int surfaceId, const VulkanSurfaceConfig& config, const VulkanBackgroundImage& backgroundImage);
    void detachVulkanSurface(int surfaceId);
    bool presentVulkanFrame(
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> deadline,
        std::optional<std::chrono::time_point<std::chrono::steady_clock>> budgetDeadline);
    void requestVulkanPresentationResync();
    std::vector<u32> captureCurrentFrameForDebug();
    std::vector<u32> captureCurrentPackedTopPrimaryForDebug();
    std::vector<u32> captureCurrentPackedBottomPrimaryForDebug();
    std::vector<u32> captureCurrent3dDimensionsForDebug();
    std::vector<u32> captureCurrent3dFrameForDebug();
    std::vector<u32> captureCurrent3dCaptureFrameForDebug();
    std::vector<u32> captureCurrent3dDepthForDebug();
    std::vector<u32> captureCurrent3dAttrForDebug();
    std::vector<u32> captureCurrent3dCoverageForDebug();
    void dumpDebugSnapshot();

    void updateConfiguration(std::shared_ptr<EmulatorConfiguration> newConfiguration);
    void requestNdsSaveWrite(const u8* saveData, u32 saveLength, u32 writeOffset, u32 writeLength);
    void requestGbaSaveWrite(const u8* saveData, u32 saveLength, u32 writeOffset, u32 writeLength);
    void requestFirmwareSaveWrite(const u8* saveData, u32 saveLength, u32 writeOffset, u32 writeLength);
    bool areSaveStatesAllowed();
    bool saveState(Savestate* state, bool refreshScreenshot);
    bool loadState(Savestate* state);
    RewindWindow getRewindWindow();
    bool loadRewindState(RewindSaveState rewindSaveState);
    void setupAchievements(
        std::list<RetroAchievements::RAAchievement> achievements,
        std::list<RetroAchievements::RALeaderboard> leaderboards,
        std::optional<std::string> richPresenceScript,
        std::optional<RetroAchievements::RARuntimeBridgeConfig> runtimeBridgeConfig
    );
    void unloadRetroAchievementsData();
    std::string getRichPresenceStatus();
    std::vector<RetroAchievements::RARuntimeAchievement> getRuntimeAchievements();
    std::vector<RetroAchievements::RARuntimeAchievementBucketEntry> getRuntimeAchievementBuckets();
    std::vector<long> getRuntimeSubsetIds();

private:
    void updateRenderer();
    void updateVulkanFastForwardRenderScale();
    void handleVulkanRuntimeFailure(const char* reason);
    bool updateVulkanScreenshot(Frame* frame, int scale, bool clearOnFailure);
    void logVulkanPerformanceIfNeeded();
    void setBatteryLevels();
    void setDateTime();
    void saveRewindState(RewindSaveState* rewindSaveState);
    std::vector<u32> captureCurrentPackedPrimaryForDebug(bool topScreen);

private:
    int instanceId;
    int consoleType;
    NDS* nds;
    std::shared_ptr<Net> net;

    std::unique_ptr<RetroAchievements::RetroAchievementsManager> retroAchievementsManager;
    std::unique_ptr<SaveManager> ndsSave;
    std::unique_ptr<SaveManager> gbaSave;
    std::unique_ptr<SaveManager> firmwareSave;
    u32 inputMask;
    std::atomic<float> slot2AnalogX = 0.0f;
    std::atomic<float> slot2AnalogY = 0.0f;

    std::shared_ptr<EmulatorConfiguration> currentConfiguration;
    FrameQueue frameQueue;
    std::unique_ptr<VulkanOutput> vulkanOutput;
    std::unique_ptr<VulkanSurfacePresenter> vulkanSurfacePresenter;
    std::vector<u32> vulkanReadbackFrame;
    Frame* lastCompletedVulkanFrame;
    int lastCompletedVulkanScale;
    ScreenshotRenderer screenshotRenderer;
    RewindManager rewindManager;
    Renderer currentRenderer;
    bool isRenderConfigurationDirty;
    bool vulkanRuntimeConfigLogged;
    bool vulkanRuntimeFailureHandled;
    int vulkanPrepareFailureCount;
    int frame;
    PerfSampleWindow<120> vulkanRunFrameCpuWindow;
    PerfSampleWindow<120> vulkanSetupCpuWindow;
    PerfSampleWindow<120> vulkanNdsRunCpuWindow;
    PerfSampleWindow<120> vulkanPostRunCpuWindow;
    PerfSampleWindow<120> vulkanComposeCpuWindow;
};

}

#endif
