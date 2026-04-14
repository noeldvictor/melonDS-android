#include "VulkanOutput.h"

#include <array>
#include <cstring>
#include <mutex>
#include <vector>

#include "GPU.h"
#include "GPU3D_Vulkan.h"
#include "NDS.h"
#include "Platform.h"
#include "VulkanContext.h"
#include "VulkanCompositorShaderData.h"

namespace MelonDSAndroid
{
bool areRendererDebugToolsEnabled();

namespace
{
constexpr int kScreenWidth = 256;
constexpr int kScreenHeight = 192;
constexpr int kAcceleratedStride = kScreenWidth * 3 + 1;
constexpr VkDeviceSize kPackedBufferSize = static_cast<VkDeviceSize>(kScreenHeight) * static_cast<VkDeviceSize>(kAcceleratedStride) * sizeof(melonDS::u32);
constexpr u64 kValidationWaitTimeoutNs = 2'000'000'000ull;

bool hasFramebufferData(const melonDS::GPU& gpu, int frontBuffer)
{
    if (frontBuffer < 0 || frontBuffer > 1)
        return false;

    return gpu.Framebuffer[frontBuffer][0] && gpu.Framebuffer[frontBuffer][1];
}

VkWriteDescriptorSet makeImageDescriptorWrite(
    VkDescriptorSet descriptorSet,
    melonDS::u32 binding,
    const VkDescriptorImageInfo* imageInfo)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = imageInfo;
    return write;
}

VkWriteDescriptorSet makeBufferDescriptorWrite(
    VkDescriptorSet descriptorSet,
    melonDS::u32 binding,
    const VkDescriptorBufferInfo* bufferInfo)
{
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = bufferInfo;
    return write;
}

}

VulkanOutput::~VulkanOutput()
{
    shutdown();
}

bool VulkanOutput::init()
{
    shutdown();

    if (!melonDS::VulkanContext::Get().Acquire())
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to acquire shared Vulkan context");
        return false;
    }

    contextAcquired = true;
    instance = melonDS::VulkanContext::Get().GetInstance();
    physicalDevice = melonDS::VulkanContext::Get().GetPhysicalDevice();
    device = melonDS::VulkanContext::Get().GetDevice();
    queue = melonDS::VulkanContext::Get().GetQueue();
    queueFamilyIndex = melonDS::VulkanContext::Get().GetQueueFamilyIndex();
    useTimelineSemaphores = melonDS::VulkanContext::Get().SupportsTimelineSemaphores();
    waitSemaphores = useTimelineSemaphores ? melonDS::VulkanContext::Get().GetWaitSemaphores() : nullptr;
    getSemaphoreCounterValue = useTimelineSemaphores ? melonDS::VulkanContext::Get().GetSemaphoreCounterValue() : nullptr;
    resetQueryPool = melonDS::VulkanContext::Get().GetResetQueryPool();
    timestampPeriodNs = melonDS::VulkanContext::Get().GetTimestampPeriod();
    timestampQueriesSupported = melonDS::VulkanContext::Get().SupportsTimestamps();

    if (useTimelineSemaphores && (waitSemaphores == nullptr || getSemaphoreCounterValue == nullptr))
    {
        melonDS::Platform::Log(
            melonDS::Platform::LogLevel::Warn,
            "VulkanOutput: timeline semaphore support reported but required functions are unavailable; using fence-based fallback"
        );
        useTimelineSemaphores = false;
        waitSemaphores = nullptr;
        getSemaphoreCounterValue = nullptr;
    }

    if (device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: shared context is incomplete");
        shutdown();
        return false;
    }

    if (!createSyncObjects() || !createCommandObjects() || !createCompositorResources())
    {
        shutdown();
        return false;
    }

    initialized = true;
    timelineValue = 0;
    melonDS::Platform::Log(
        melonDS::Platform::LogLevel::Warn,
        "VulkanOutput: sync path initialized (timeline=%d)",
        useTimelineSemaphores ? 1 : 0
    );
    return true;
}

void VulkanOutput::shutdown()
{
    if (device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(device);

    destroyFrameResources();
    destroyCompositorResources();

    if (timelineSemaphore != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(device, timelineSemaphore, nullptr);
        timelineSemaphore = VK_NULL_HANDLE;
    }

    if (commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    if (contextAcquired)
    {
        melonDS::VulkanContext::Get().Release();
        contextAcquired = false;
    }

    instance = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
    queueFamilyIndex = 0;
    waitSemaphores = nullptr;
    getSemaphoreCounterValue = nullptr;
    resetQueryPool = nullptr;
    timestampPeriodNs = 0.0f;
    timestampQueriesSupported = false;
    timelineValue = 0;
    useTimelineSemaphores = false;
    initialized = false;
}

bool VulkanOutput::createSyncObjects()
{
    if (!useTimelineSemaphores)
        return true;

    VkSemaphoreTypeCreateInfo semaphoreTypeInfo{};
    semaphoreTypeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphoreTypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphoreTypeInfo.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = &semaphoreTypeInfo;

    if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &timelineSemaphore) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create timeline semaphore");
        return false;
    }

    return true;
}

bool VulkanOutput::createCommandObjects()
{
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;

    if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create command pool");
        return false;
    }

    return true;
}

bool VulkanOutput::createTimestampQueryPool(VkQueryPool& queryPool)
{
    if (!timestampQueriesSupported)
        return true;

    VkQueryPoolCreateInfo queryPoolCreateInfo{};
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = 2;

    if (vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &queryPool) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Warn, "VulkanOutput: failed to create timestamp query pool");
        queryPool = VK_NULL_HANDLE;
    }

    return true;
}

void VulkanOutput::destroyTimestampQueryPool(VkQueryPool& queryPool)
{
    if (queryPool != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device, queryPool, nullptr);
        queryPool = VK_NULL_HANDLE;
    }
}

bool VulkanOutput::createCompositorResources()
{
    VkDescriptorSetLayoutBinding outputBinding{};
    outputBinding.binding = 0;
    outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    outputBinding.descriptorCount = 1;
    outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding input3dBinding{};
    input3dBinding.binding = 1;
    input3dBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    input3dBinding.descriptorCount = 1;
    input3dBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding topPackedBinding{};
    topPackedBinding.binding = 2;
    topPackedBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    topPackedBinding.descriptorCount = 1;
    topPackedBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding bottomPackedBinding{};
    bottomPackedBinding.binding = 3;
    bottomPackedBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bottomPackedBinding.descriptorCount = 1;
    bottomPackedBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding, 4> compositorBindings = {
        outputBinding,
        input3dBinding,
        topPackedBinding,
        bottomPackedBinding,
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = static_cast<u32>(compositorBindings.size());
    descriptorSetLayoutCreateInfo.pBindings = compositorBindings.data();

    if (vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &compositorDescriptorSetLayout) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create compositor descriptor set layout");
        return false;
    }

    std::array<VkDescriptorPoolSize, 2> descriptorPoolSizes{};
    descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorPoolSizes[0].descriptorCount = static_cast<u32>(FRAME_QUEUE_SIZE * 2);
    descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSizes[1].descriptorCount = static_cast<u32>(FRAME_QUEUE_SIZE * 2);

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorPoolCreateInfo.maxSets = static_cast<u32>(FRAME_QUEUE_SIZE);
    descriptorPoolCreateInfo.poolSizeCount = static_cast<u32>(descriptorPoolSizes.size());
    descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes.data();

    if (vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &compositorDescriptorPool) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create compositor descriptor pool");
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(CompositorPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &compositorDescriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compositorPipelineLayout) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create compositor pipeline layout");
        return false;
    }

    if (melonDS_android_vulkan_compositor_comp_spv_len == 0)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: compositor SPIR-V blob is empty");
        return false;
    }

    std::vector<u32> shaderWords((melonDS_android_vulkan_compositor_comp_spv_len + sizeof(u32) - 1u) / sizeof(u32));
    std::memcpy(shaderWords.data(), melonDS_android_vulkan_compositor_comp_spv, melonDS_android_vulkan_compositor_comp_spv_len);

    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = melonDS_android_vulkan_compositor_comp_spv_len;
    shaderModuleCreateInfo.pCode = shaderWords.data();

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create compositor shader module");
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo{};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = "main";

    VkComputePipelineCreateInfo computePipelineCreateInfo{};
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.stage = shaderStageCreateInfo;
    computePipelineCreateInfo.layout = compositorPipelineLayout;

    const VkResult pipelineResult = vkCreateComputePipelines(
        device,
        VK_NULL_HANDLE,
        1,
        &computePipelineCreateInfo,
        nullptr,
        &compositorPipeline
    );

    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (pipelineResult != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create compositor pipeline (%d)", static_cast<int>(pipelineResult));
        return false;
    }

    return true;
}

void VulkanOutput::destroyCompositorResources()
{
    if (compositorPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, compositorPipeline, nullptr);
        compositorPipeline = VK_NULL_HANDLE;
    }

    if (compositorPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, compositorPipelineLayout, nullptr);
        compositorPipelineLayout = VK_NULL_HANDLE;
    }

    if (compositorDescriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, compositorDescriptorPool, nullptr);
        compositorDescriptorPool = VK_NULL_HANDLE;
    }

    if (compositorDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, compositorDescriptorSetLayout, nullptr);
        compositorDescriptorSetLayout = VK_NULL_HANDLE;
    }
}

u32 VulkanOutput::findMemoryType(u32 typeBits, VkMemoryPropertyFlags properties) const
{
    return melonDS::VulkanContext::Get().FindMemoryType(typeBits, properties);
}

bool VulkanOutput::createFrameResource(Frame* frame, u32 width, u32 height)
{
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
    imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(device, &imageCreateInfo, nullptr, &image) != VK_SUCCESS)
    {
        melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create frame image");
        return false;
    }

    VkMemoryRequirements imageRequirements{};
    vkGetImageMemoryRequirements(device, image, &imageRequirements);

    VkMemoryAllocateInfo imageMemoryAllocateInfo{};
    imageMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imageMemoryAllocateInfo.allocationSize = imageRequirements.size;

    u32 imageMemoryType = findMemoryType(imageRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (imageMemoryType == UINT32_MAX)
        imageMemoryType = findMemoryType(imageRequirements.memoryTypeBits, 0);
    if (imageMemoryType == UINT32_MAX)
    {
        vkDestroyImage(device, image, nullptr);
        return false;
    }
    imageMemoryAllocateInfo.memoryTypeIndex = imageMemoryType;

    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &imageMemoryAllocateInfo, nullptr, &imageMemory) != VK_SUCCESS)
    {
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    if (vkBindImageMemory(device, image, imageMemory, 0) != VK_SUCCESS)
    {
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkImageView imageView = VK_NULL_HANDLE;
    if (vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    VkDeviceSize stagingBufferSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;

    VkBufferCreateInfo stagingBufferCreateInfo{};
    stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufferCreateInfo.size = stagingBufferSize;
    stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    stagingBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &stagingBufferCreateInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
    {
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    VkMemoryRequirements stagingBufferRequirements{};
    vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingBufferRequirements);

    VkMemoryAllocateInfo stagingMemoryAllocateInfo{};
    stagingMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingMemoryAllocateInfo.allocationSize = stagingBufferRequirements.size;
    stagingMemoryAllocateInfo.memoryTypeIndex = findMemoryType(
        stagingBufferRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    if (stagingMemoryAllocateInfo.memoryTypeIndex == UINT32_MAX)
    {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &stagingMemoryAllocateInfo, nullptr, &stagingMemory) != VK_SUCCESS)
    {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    if (vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0) != VK_SUCCESS)
    {
        vkFreeMemory(device, stagingMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS)
    {
        vkFreeMemory(device, stagingMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VkFence submitFence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fenceCreateInfo, nullptr, &submitFence) != VK_SUCCESS)
    {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkFreeMemory(device, stagingMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = compositorDescriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &compositorDescriptorSetLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet) != VK_SUCCESS)
    {
        vkDestroyFence(device, submitFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkFreeMemory(device, stagingMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    auto createPackedBuffer = [&](VkBuffer& buffer, VkDeviceMemory& memory, void*& mappedMemory, const char* label) -> bool {
        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = kPackedBufferSize;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
        {
            melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to create %s packed buffer", label);
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

        VkMemoryAllocateInfo memoryAllocateInfo{};
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex = findMemoryType(
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX
            || vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory) != VK_SUCCESS
            || vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS
            || vkMapMemory(device, memory, 0, kPackedBufferSize, 0, &mappedMemory) != VK_SUCCESS)
        {
            melonDS::Platform::Log(melonDS::Platform::LogLevel::Error, "VulkanOutput: failed to allocate %s packed buffer memory", label);
            if (mappedMemory != nullptr)
            {
                vkUnmapMemory(device, memory);
                mappedMemory = nullptr;
            }
            if (memory != VK_NULL_HANDLE)
            {
                vkFreeMemory(device, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }
            if (buffer != VK_NULL_HANDLE)
            {
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
            }
            return false;
        }

        return true;
    };

    VkBuffer topPackedBuffer = VK_NULL_HANDLE;
    VkDeviceMemory topPackedMemory = VK_NULL_HANDLE;
    void* topPackedMapped = nullptr;
    VkBuffer bottomPackedBuffer = VK_NULL_HANDLE;
    VkDeviceMemory bottomPackedMemory = VK_NULL_HANDLE;
    void* bottomPackedMapped = nullptr;

    if (!createPackedBuffer(topPackedBuffer, topPackedMemory, topPackedMapped, "top")
        || !createPackedBuffer(bottomPackedBuffer, bottomPackedMemory, bottomPackedMapped, "bottom"))
    {
        if (bottomPackedMapped != nullptr)
            vkUnmapMemory(device, bottomPackedMemory);
        if (bottomPackedMemory != VK_NULL_HANDLE)
            vkFreeMemory(device, bottomPackedMemory, nullptr);
        if (bottomPackedBuffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device, bottomPackedBuffer, nullptr);
        if (topPackedMapped != nullptr)
            vkUnmapMemory(device, topPackedMemory);
        if (topPackedMemory != VK_NULL_HANDLE)
            vkFreeMemory(device, topPackedMemory, nullptr);
        if (topPackedBuffer != VK_NULL_HANDLE)
            vkDestroyBuffer(device, topPackedBuffer, nullptr);
        vkFreeDescriptorSets(device, compositorDescriptorPool, 1, &descriptorSet);
        vkDestroyFence(device, submitFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkFreeMemory(device, stagingMemory, nullptr);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkDestroyImageView(device, imageView, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        vkDestroyImage(device, image, nullptr);
        return false;
    }

    FrameResource resource{};
    resource.image = image;
    resource.imageView = imageView;
    resource.imageMemory = imageMemory;
    resource.stagingBuffer = stagingBuffer;
    resource.stagingMemory = stagingMemory;
    resource.stagingSize = stagingBufferSize;
    resource.commandBuffer = commandBuffer;
    resource.submitFence = submitFence;
    resource.descriptorSet = descriptorSet;
    resource.topPackedBuffer = topPackedBuffer;
    resource.topPackedMemory = topPackedMemory;
    resource.topPackedMapped = topPackedMapped;
    resource.bottomPackedBuffer = bottomPackedBuffer;
    resource.bottomPackedMemory = bottomPackedMemory;
    resource.bottomPackedMapped = bottomPackedMapped;
    resource.packedBufferSize = kPackedBufferSize;
    resource.renderer3dSnapshot = VK_NULL_HANDLE;
    resource.renderer3dSnapshotView = VK_NULL_HANDLE;
    resource.renderer3dSnapshotMemory = VK_NULL_HANDLE;
    resource.snapshotWidth = 0;
    resource.snapshotHeight = 0;
    resource.submissionValue = 0;
    resource.width = width;
    resource.height = height;
    resource.hasContent = false;
    resource.hasPreparedInputs = false;
    resource.hasRenderer3dSnapshot = false;
    resource.descriptorSetReady = false;
    resource.timestampPending = false;
    resource.cachedRendererImageView = VK_NULL_HANDLE;

    (void)createTimestampQueryPool(resource.timestampQueryPool);

    resources[frame] = resource;

    frame->backend = FrameBackend::VulkanImage;
    frame->renderTimelineValue = 0;

    return true;
}

void VulkanOutput::destroyFrameResource(Frame* frame)
{
    auto iterator = resources.find(frame);
    if (iterator == resources.end())
        return;

    FrameResource& resource = iterator->second;

    if (resource.submitFence != VK_NULL_HANDLE)
        vkWaitForFences(device, 1, &resource.submitFence, VK_TRUE, UINT64_MAX);

    if (resource.descriptorSet != VK_NULL_HANDLE && compositorDescriptorPool != VK_NULL_HANDLE)
        vkFreeDescriptorSets(device, compositorDescriptorPool, 1, &resource.descriptorSet);

    destroyTimestampQueryPool(resource.timestampQueryPool);

    if (resource.submitFence != VK_NULL_HANDLE)
        vkDestroyFence(device, resource.submitFence, nullptr);

    if (resource.commandBuffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device, commandPool, 1, &resource.commandBuffer);

    if (resource.topPackedMapped != nullptr)
    {
        vkUnmapMemory(device, resource.topPackedMemory);
        resource.topPackedMapped = nullptr;
    }
    if (resource.topPackedBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, resource.topPackedBuffer, nullptr);
    if (resource.topPackedMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, resource.topPackedMemory, nullptr);

    if (resource.bottomPackedMapped != nullptr)
    {
        vkUnmapMemory(device, resource.bottomPackedMemory);
        resource.bottomPackedMapped = nullptr;
    }
    if (resource.bottomPackedBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, resource.bottomPackedBuffer, nullptr);
    if (resource.bottomPackedMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, resource.bottomPackedMemory, nullptr);

    destroyRenderer3dSnapshot(resource);

    if (resource.stagingBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(device, resource.stagingBuffer, nullptr);
    if (resource.stagingMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, resource.stagingMemory, nullptr);

    if (resource.imageView != VK_NULL_HANDLE)
        vkDestroyImageView(device, resource.imageView, nullptr);
    if (resource.image != VK_NULL_HANDLE)
        vkDestroyImage(device, resource.image, nullptr);
    if (resource.imageMemory != VK_NULL_HANDLE)
        vkFreeMemory(device, resource.imageMemory, nullptr);

    if (frame != nullptr)
    {
        frame->renderTimelineValue = 0;
    }

    resources.erase(iterator);
}

void VulkanOutput::destroyFrameResources()
{
    while (!resources.empty())
    {
        auto iterator = resources.begin();
        destroyFrameResource(iterator->first);
    }
}

bool VulkanOutput::ensureFrameResources(Frame* frame, u32 width, u32 height)
{
    if (!initialized || frame == nullptr || width == 0 || height == 0)
        return false;

    auto iterator = resources.find(frame);
    if (iterator != resources.end())
    {
        const FrameResource& resource = iterator->second;
        if (resource.width == width && resource.height == height)
        {
            frame->backend = FrameBackend::VulkanImage;
            return true;
        }

        destroyFrameResource(frame);
    }

    return createFrameResource(frame, width, height);
}

bool VulkanOutput::beginFrameCommand(FrameResource& resource, u64 waitTimeoutNs)
{
    const VkResult waitResult = vkWaitForFences(device, 1, &resource.submitFence, VK_TRUE, waitTimeoutNs);
    if (waitResult != VK_SUCCESS)
    {
        melonDS::Platform::Log(
            melonDS::Platform::LogLevel::Error,
            "VulkanOutput: beginFrameCommand fence wait failed (%d, timeoutNs=%llu)",
            static_cast<int>(waitResult),
            static_cast<unsigned long long>(waitTimeoutNs)
        );
        return false;
    }

    consumeFrameGpuTiming(resource);

    if (resource.timestampQueryPool != VK_NULL_HANDLE && resetQueryPool != nullptr)
        resetQueryPool(device, resource.timestampQueryPool, 0, 2);

    if (vkResetFences(device, 1, &resource.submitFence) != VK_SUCCESS)
        return false;

    if (vkResetCommandBuffer(resource.commandBuffer, 0) != VK_SUCCESS)
        return false;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(resource.commandBuffer, &beginInfo) == VK_SUCCESS;
}

bool VulkanOutput::submitFrameCommand(Frame* frame, FrameResource& resource, bool signalTimeline)
{
    if (vkEndCommandBuffer(resource.commandBuffer) != VK_SUCCESS)
        return false;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &resource.commandBuffer;

    u64 signalValue = resource.submissionValue;
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    const bool shouldSignalTimelineSemaphore = signalTimeline && useTimelineSemaphores && timelineSemaphore != VK_NULL_HANDLE;
    if (signalTimeline)
    {
        signalValue = ++timelineValue;
        if (shouldSignalTimelineSemaphore)
        {
            timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timelineSubmitInfo.signalSemaphoreValueCount = 1;
            timelineSubmitInfo.pSignalSemaphoreValues = &signalValue;

            submitInfo.pNext = &timelineSubmitInfo;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &timelineSemaphore;
        }
    }

    {
        std::scoped_lock queueLock(melonDS::VulkanContext::Get().GetQueueLock());
        if (vkQueueSubmit(queue, 1, &submitInfo, resource.submitFence) != VK_SUCCESS)
            return false;
    }

    if (frame != nullptr)
    {
        frame->backend = FrameBackend::VulkanImage;
        if (signalTimeline)
            frame->renderTimelineValue = signalValue;
    }

    if (signalTimeline)
        resource.submissionValue = signalValue;

    if (signalTimeline && resource.timestampQueryPool != VK_NULL_HANDLE)
        resource.timestampPending = true;

    return true;
}

bool VulkanOutput::updateCompositorPackedBuffers(FrameResource& resource, const melonDS::GPU& gpu, int frontBuffer)
{
    if (!hasFramebufferData(gpu, frontBuffer))
        return false;

    if (resource.topPackedMapped == nullptr || resource.bottomPackedMapped == nullptr || resource.packedBufferSize == 0)
        return false;

    const melonDS::u32* topPacked = gpu.Framebuffer[frontBuffer][0].get();
    const melonDS::u32* bottomPacked = gpu.Framebuffer[frontBuffer][1].get();

    std::memcpy(resource.topPackedMapped, topPacked, static_cast<size_t>(resource.packedBufferSize));
    std::memcpy(resource.bottomPackedMapped, bottomPacked, static_cast<size_t>(resource.packedBufferSize));

    return true;
}

bool VulkanOutput::prepareFrameForPresentation(
    Frame* frame,
    const melonDS::GPU& gpu,
    int frontBuffer,
    const melonDS::VulkanRenderer3D& renderer3D)
{
    if (!initialized || frame == nullptr || !renderer3D.HasColorTarget())
        return false;

    auto iterator = resources.find(frame);
    if (iterator == resources.end())
        return false;

    FrameResource& resource = iterator->second;

    const u64 packedUploadStartNs = PerfNowNs();
    if (!updateCompositorPackedBuffers(resource, gpu, frontBuffer))
        return false;
    packedUploadCpuWindow.Add(PerfNowNs() - packedUploadStartNs);
    resource.screenSwap = (gpu.NDS.PowerControl9 & (1u << 15)) != 0u;

    if (resource.hasRenderer3dSnapshot && frame->renderTimelineValue != 0)
    {
        resource.hasPreparedInputs = true;
        resource.hasContent = false;
        return true;
    }

    if (!recordDirectPresentationPrep(frame, resource, renderer3D))
        return false;

    resource.hasPreparedInputs = true;
    resource.hasContent = false;
    return true;
}

bool VulkanOutput::captureRenderer3dSnapshot(Frame* frame, const melonDS::VulkanRenderer3D& renderer3D)
{
    if (frame != nullptr)
        frame->renderTimelineValue = 0;

    if (!initialized || frame == nullptr || !renderer3D.HasColorTarget())
        return false;

    auto iterator = resources.find(frame);
    if (iterator == resources.end())
        return false;

    FrameResource& resource = iterator->second;
    resource.hasPreparedInputs = false;

    if (!beginFrameCommand(resource))
        return false;

    if (!recordRenderer3dSnapshotCopy(resource, renderer3D))
        return false;

    const bool submitted = submitFrameCommand(frame, resource, true);
    if (submitted)
        resource.timestampPending = false;
    return submitted;
}

bool VulkanOutput::composeAndSubmitFrame(
    Frame* frame,
    const VulkanCompositionInputs& inputs)
{
    if (!initialized || frame == nullptr || inputs.scale < 1 || inputs.sourceImage == VK_NULL_HANDLE || inputs.sourceImageView == VK_NULL_HANDLE)
        return false;

    auto iterator = resources.find(frame);
    if (iterator == resources.end())
        return false;

    FrameResource& resource = iterator->second;

    const u64 composeStartNs = PerfNowNs();
    const bool dispatched = dispatchCompositor(frame, resource, inputs);
    composeCpuWindow.Add(PerfNowNs() - composeStartNs);
    logPerformanceIfNeeded();
    return dispatched;
}

bool VulkanOutput::buildCompositionInputs(
    const Frame* frame,
    const melonDS::VulkanRenderer3D& renderer3D,
    int scale,
    bool needsReadback,
    bool multiSurface,
    bool validationMode,
    VulkanCompositionInputs& outInputs) const
{
    if (!initialized || frame == nullptr || scale < 1 || !renderer3D.HasColorTarget())
        return false;

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return false;

    const FrameResource& resource = iterator->second;
    if (!resource.hasPreparedInputs)
        return false;

    if (resource.hasRenderer3dSnapshot
        && resource.renderer3dSnapshot != VK_NULL_HANDLE
        && resource.renderer3dSnapshotView != VK_NULL_HANDLE)
    {
        outInputs.sourceImage = resource.renderer3dSnapshot;
        outInputs.sourceImageView = resource.renderer3dSnapshotView;
        outInputs.rendererWidth = resource.snapshotWidth;
        outInputs.rendererHeight = resource.snapshotHeight;
    }
    else
    {
        outInputs.sourceImage = renderer3D.GetColorTargetImage();
        outInputs.sourceImageView = renderer3D.GetColorTargetImageView();
        outInputs.rendererWidth = renderer3D.GetColorTargetWidth();
        outInputs.rendererHeight = renderer3D.GetColorTargetHeight();
    }
    outInputs.topPackedBuffer = resource.topPackedBuffer;
    outInputs.bottomPackedBuffer = resource.bottomPackedBuffer;
    outInputs.packedBufferSize = resource.packedBufferSize;
    outInputs.packedStride = kAcceleratedStride;
    outInputs.scale = static_cast<u32>(scale);
    outInputs.needsReadback = needsReadback;
    outInputs.multiSurface = multiSurface;
    outInputs.validationMode = validationMode;
    return outInputs.sourceImage != VK_NULL_HANDLE
        && outInputs.sourceImageView != VK_NULL_HANDLE
        && outInputs.topPackedBuffer != VK_NULL_HANDLE
        && outInputs.bottomPackedBuffer != VK_NULL_HANDLE;
}

void VulkanOutput::destroyRenderer3dSnapshot(FrameResource& resource)
{
    if (resource.renderer3dSnapshotView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device, resource.renderer3dSnapshotView, nullptr);
        resource.renderer3dSnapshotView = VK_NULL_HANDLE;
    }
    if (resource.renderer3dSnapshot != VK_NULL_HANDLE)
    {
        vkDestroyImage(device, resource.renderer3dSnapshot, nullptr);
        resource.renderer3dSnapshot = VK_NULL_HANDLE;
    }
    if (resource.renderer3dSnapshotMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(device, resource.renderer3dSnapshotMemory, nullptr);
        resource.renderer3dSnapshotMemory = VK_NULL_HANDLE;
    }

    resource.snapshotWidth = 0;
    resource.snapshotHeight = 0;
    resource.hasRenderer3dSnapshot = false;
}

bool VulkanOutput::ensureRenderer3dSnapshot(FrameResource& resource, u32 width, u32 height)
{
    if (width == 0 || height == 0)
        return false;

    if (resource.renderer3dSnapshot != VK_NULL_HANDLE
        && resource.renderer3dSnapshotView != VK_NULL_HANDLE
        && resource.snapshotWidth == width
        && resource.snapshotHeight == height)
        return true;

    destroyRenderer3dSnapshot(resource);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageCreateInfo.extent = {width, height, 1};
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &imageCreateInfo, nullptr, &resource.renderer3dSnapshot) != VK_SUCCESS)
        return false;

    VkMemoryRequirements memoryRequirements{};
    vkGetImageMemoryRequirements(device, resource.renderer3dSnapshot, &memoryRequirements);

    VkMemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (memoryAllocateInfo.memoryTypeIndex == UINT32_MAX
        || vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &resource.renderer3dSnapshotMemory) != VK_SUCCESS)
    {
        vkDestroyImage(device, resource.renderer3dSnapshot, nullptr);
        resource.renderer3dSnapshot = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindImageMemory(device, resource.renderer3dSnapshot, resource.renderer3dSnapshotMemory, 0) != VK_SUCCESS)
    {
        destroyRenderer3dSnapshot(resource);
        return false;
    }

    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = resource.renderer3dSnapshot;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &imageViewCreateInfo, nullptr, &resource.renderer3dSnapshotView) != VK_SUCCESS)
    {
        destroyRenderer3dSnapshot(resource);
        return false;
    }

    resource.snapshotWidth = width;
    resource.snapshotHeight = height;
    return true;
}

bool VulkanOutput::recordDirectPresentationPrep(Frame* frame, FrameResource& resource, const melonDS::VulkanRenderer3D& renderer3D)
{
    if (!beginFrameCommand(resource))
        return false;

    if (!recordRenderer3dSnapshotCopy(resource, renderer3D))
        return false;

    VkBufferMemoryBarrier topPackedBarrier{};
    topPackedBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    topPackedBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    topPackedBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    topPackedBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    topPackedBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    topPackedBarrier.buffer = resource.topPackedBuffer;
    topPackedBarrier.offset = 0;
    topPackedBarrier.size = resource.packedBufferSize;

    VkBufferMemoryBarrier bottomPackedBarrier{};
    bottomPackedBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bottomPackedBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bottomPackedBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bottomPackedBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bottomPackedBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bottomPackedBarrier.buffer = resource.bottomPackedBuffer;
    bottomPackedBarrier.offset = 0;
    bottomPackedBarrier.size = resource.packedBufferSize;

    std::array<VkBufferMemoryBarrier, 2> bufferBarriers = {
        topPackedBarrier,
        bottomPackedBarrier,
    };
    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        static_cast<u32>(bufferBarriers.size()),
        bufferBarriers.data(),
        0,
        nullptr
    );

    const bool submitted = submitFrameCommand(frame, resource, true);
    if (submitted)
        resource.timestampPending = false;
    return submitted;
}

bool VulkanOutput::recordRenderer3dSnapshotCopy(FrameResource& resource, const melonDS::VulkanRenderer3D& renderer3D)
{
    const u32 rendererWidth = renderer3D.GetColorTargetWidth();
    const u32 rendererHeight = renderer3D.GetColorTargetHeight();
    if (!ensureRenderer3dSnapshot(resource, rendererWidth, rendererHeight))
        return false;

    VkImageMemoryBarrier sourceToTransferBarrier{};
    sourceToTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    sourceToTransferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    sourceToTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sourceToTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    sourceToTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sourceToTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sourceToTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sourceToTransferBarrier.image = renderer3D.GetColorTargetImage();
    sourceToTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sourceToTransferBarrier.subresourceRange.baseMipLevel = 0;
    sourceToTransferBarrier.subresourceRange.levelCount = 1;
    sourceToTransferBarrier.subresourceRange.baseArrayLayer = 0;
    sourceToTransferBarrier.subresourceRange.layerCount = 1;

    VkImageMemoryBarrier snapshotToTransferBarrier{};
    snapshotToTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    snapshotToTransferBarrier.srcAccessMask = resource.hasRenderer3dSnapshot ? VK_ACCESS_SHADER_READ_BIT : 0;
    snapshotToTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    snapshotToTransferBarrier.oldLayout = resource.hasRenderer3dSnapshot ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    snapshotToTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    snapshotToTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    snapshotToTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    snapshotToTransferBarrier.image = resource.renderer3dSnapshot;
    snapshotToTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    snapshotToTransferBarrier.subresourceRange.baseMipLevel = 0;
    snapshotToTransferBarrier.subresourceRange.levelCount = 1;
    snapshotToTransferBarrier.subresourceRange.baseArrayLayer = 0;
    snapshotToTransferBarrier.subresourceRange.layerCount = 1;

    std::array<VkImageMemoryBarrier, 2> preCopyBarriers = {
        sourceToTransferBarrier,
        snapshotToTransferBarrier,
    };
    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        static_cast<u32>(preCopyBarriers.size()),
        preCopyBarriers.data()
    );

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.extent = {rendererWidth, rendererHeight, 1};
    vkCmdCopyImage(
        resource.commandBuffer,
        renderer3D.GetColorTargetImage(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        resource.renderer3dSnapshot,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copyRegion
    );

    VkImageMemoryBarrier sourceBackToGeneralBarrier{};
    sourceBackToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    sourceBackToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sourceBackToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    sourceBackToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sourceBackToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    sourceBackToGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sourceBackToGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sourceBackToGeneralBarrier.image = renderer3D.GetColorTargetImage();
    sourceBackToGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sourceBackToGeneralBarrier.subresourceRange.baseMipLevel = 0;
    sourceBackToGeneralBarrier.subresourceRange.levelCount = 1;
    sourceBackToGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    sourceBackToGeneralBarrier.subresourceRange.layerCount = 1;

    VkImageMemoryBarrier snapshotToReadableBarrier{};
    snapshotToReadableBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    snapshotToReadableBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    snapshotToReadableBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    snapshotToReadableBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    snapshotToReadableBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    snapshotToReadableBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    snapshotToReadableBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    snapshotToReadableBarrier.image = resource.renderer3dSnapshot;
    snapshotToReadableBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    snapshotToReadableBarrier.subresourceRange.baseMipLevel = 0;
    snapshotToReadableBarrier.subresourceRange.levelCount = 1;
    snapshotToReadableBarrier.subresourceRange.baseArrayLayer = 0;
    snapshotToReadableBarrier.subresourceRange.layerCount = 1;

    std::array<VkImageMemoryBarrier, 2> postCopyBarriers = {
        sourceBackToGeneralBarrier,
        snapshotToReadableBarrier,
    };
    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        static_cast<u32>(postCopyBarriers.size()),
        postCopyBarriers.data()
    );

    resource.hasRenderer3dSnapshot = true;
    return true;
}

bool VulkanOutput::dispatchCompositor(
    Frame* frame,
    FrameResource& resource,
    const VulkanCompositionInputs& inputs)
{
    if (!beginFrameCommand(resource))
        return false;

    if (resource.timestampQueryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(resource.commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, resource.timestampQueryPool, 0);

    VkImageMemoryBarrier outputToGeneralBarrier{};
    outputToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    outputToGeneralBarrier.srcAccessMask = resource.hasContent ? (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT) : 0;
    outputToGeneralBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputToGeneralBarrier.oldLayout = resource.hasContent ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    outputToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputToGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputToGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputToGeneralBarrier.image = resource.image;
    outputToGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    outputToGeneralBarrier.subresourceRange.baseMipLevel = 0;
    outputToGeneralBarrier.subresourceRange.levelCount = 1;
    outputToGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    outputToGeneralBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        resource.hasContent ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &outputToGeneralBarrier
    );

    VkImageMemoryBarrier renderer3dReadableBarrier{};
    renderer3dReadableBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    renderer3dReadableBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    renderer3dReadableBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    renderer3dReadableBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    renderer3dReadableBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    renderer3dReadableBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    renderer3dReadableBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    renderer3dReadableBarrier.image = inputs.sourceImage;
    renderer3dReadableBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    renderer3dReadableBarrier.subresourceRange.baseMipLevel = 0;
    renderer3dReadableBarrier.subresourceRange.levelCount = 1;
    renderer3dReadableBarrier.subresourceRange.baseArrayLayer = 0;
    renderer3dReadableBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &renderer3dReadableBarrier
    );

    VkBufferMemoryBarrier topPackedBarrier{};
    topPackedBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    topPackedBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    topPackedBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    topPackedBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    topPackedBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    topPackedBarrier.buffer = resource.topPackedBuffer;
    topPackedBarrier.offset = 0;
    topPackedBarrier.size = resource.packedBufferSize;

    VkBufferMemoryBarrier bottomPackedBarrier{};
    bottomPackedBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bottomPackedBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bottomPackedBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bottomPackedBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bottomPackedBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bottomPackedBarrier.buffer = resource.bottomPackedBuffer;
    bottomPackedBarrier.offset = 0;
    bottomPackedBarrier.size = resource.packedBufferSize;

    std::array<VkBufferMemoryBarrier, 2> compositorBufferBarriers = {
        topPackedBarrier,
        bottomPackedBarrier,
    };

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        static_cast<u32>(compositorBufferBarriers.size()),
        compositorBufferBarriers.data(),
        0,
        nullptr
    );

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = resource.imageView;
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo input3dImageInfo{};
    input3dImageInfo.imageView = inputs.sourceImageView;
    input3dImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo topPackedBufferInfo{};
    topPackedBufferInfo.buffer = resource.topPackedBuffer;
    topPackedBufferInfo.offset = 0;
    topPackedBufferInfo.range = resource.packedBufferSize;

    VkDescriptorBufferInfo bottomPackedBufferInfo{};
    bottomPackedBufferInfo.buffer = resource.bottomPackedBuffer;
    bottomPackedBufferInfo.offset = 0;
    bottomPackedBufferInfo.range = resource.packedBufferSize;

    if (!resource.descriptorSetReady || resource.cachedRendererImageView != inputs.sourceImageView)
    {
        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
        descriptorWrites[0] = makeImageDescriptorWrite(resource.descriptorSet, 0, &outputImageInfo);
        descriptorWrites[1] = makeImageDescriptorWrite(resource.descriptorSet, 1, &input3dImageInfo);
        descriptorWrites[2] = makeBufferDescriptorWrite(resource.descriptorSet, 2, &topPackedBufferInfo);
        descriptorWrites[3] = makeBufferDescriptorWrite(resource.descriptorSet, 3, &bottomPackedBufferInfo);

        vkUpdateDescriptorSets(device, static_cast<u32>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        resource.descriptorSetReady = true;
        resource.cachedRendererImageView = inputs.sourceImageView;
    }

    vkCmdBindPipeline(resource.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compositorPipeline);
    vkCmdBindDescriptorSets(
        resource.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        compositorPipelineLayout,
        0,
        1,
        &resource.descriptorSet,
        0,
        nullptr
    );

    CompositorPushConstants pushConstants{};
    pushConstants.outputWidth = resource.width;
    pushConstants.outputHeight = resource.height;
    pushConstants.scale = inputs.scale;
    pushConstants.rendererWidth = inputs.rendererWidth;
    pushConstants.rendererHeight = inputs.rendererHeight;
    pushConstants.packedStride = inputs.packedStride;

    vkCmdPushConstants(
        resource.commandBuffer,
        compositorPipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(pushConstants),
        &pushConstants
    );

    const u32 groupCountX = (resource.width + 7u) / 8u;
    const u32 groupCountY = (resource.height + 7u) / 8u;
    vkCmdDispatch(resource.commandBuffer, groupCountX, groupCountY, 1);

    if (resource.timestampQueryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(resource.commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, resource.timestampQueryPool, 1);

    VkImageMemoryBarrier outputReadableBarrier{};
    outputReadableBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    outputReadableBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputReadableBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    outputReadableBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputReadableBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputReadableBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputReadableBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputReadableBarrier.image = resource.image;
    outputReadableBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    outputReadableBarrier.subresourceRange.baseMipLevel = 0;
    outputReadableBarrier.subresourceRange.levelCount = 1;
    outputReadableBarrier.subresourceRange.baseArrayLayer = 0;
    outputReadableBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &outputReadableBarrier
    );

    if (!submitFrameCommand(frame, resource, true))
        return false;

    resource.hasContent = true;
    return true;
}

bool VulkanOutput::validateCompositorSubmission(Frame* frame, const melonDS::VulkanRenderer3D& renderer3D, int scale, u64 waitTimeoutNs)
{
    if (!initialized || frame == nullptr || scale < 1 || !renderer3D.HasColorTarget())
        return false;

    auto iterator = resources.find(frame);
    if (iterator == resources.end())
        return false;

    FrameResource& resource = iterator->second;
    if (resource.topPackedMapped == nullptr || resource.bottomPackedMapped == nullptr || resource.packedBufferSize == 0)
        return false;
    std::memset(resource.topPackedMapped, 0, static_cast<size_t>(resource.packedBufferSize));
    std::memset(resource.bottomPackedMapped, 0, static_cast<size_t>(resource.packedBufferSize));
    resource.hasPreparedInputs = true;

    VulkanCompositionInputs inputs{};
    if (!buildCompositionInputs(frame, renderer3D, scale, false, false, true, inputs))
        return false;

    if (!dispatchCompositor(frame, resource, inputs))
        return false;

    if (!waitForFrame(frame, waitTimeoutNs))
    {
        melonDS::Platform::Log(
            melonDS::Platform::LogLevel::Error,
            "VulkanOutput: validateCompositorSubmission timed out (timeoutNs=%llu)",
            static_cast<unsigned long long>(waitTimeoutNs)
        );
        return false;
    }

    return true;
}

bool VulkanOutput::validateFrameSubmission(Frame* frame, u64 waitTimeoutNs)
{
    if (!initialized || frame == nullptr)
        return false;

    auto iterator = resources.find(frame);
    if (iterator == resources.end())
        return false;

    FrameResource& resource = iterator->second;
    if (!beginFrameCommand(resource, waitTimeoutNs))
        return false;

    VkImageMemoryBarrier toTransferDstBarrier{};
    toTransferDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferDstBarrier.srcAccessMask = resource.hasContent ? (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT) : 0;
    toTransferDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toTransferDstBarrier.oldLayout = resource.hasContent ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDstBarrier.image = resource.image;
    toTransferDstBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferDstBarrier.subresourceRange.baseMipLevel = 0;
    toTransferDstBarrier.subresourceRange.levelCount = 1;
    toTransferDstBarrier.subresourceRange.baseArrayLayer = 0;
    toTransferDstBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        resource.hasContent ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toTransferDstBarrier
    );

    VkClearColorValue clearColor{};
    clearColor.float32[0] = 0.0f;
    clearColor.float32[1] = 0.0f;
    clearColor.float32[2] = 0.0f;
    clearColor.float32[3] = 1.0f;

    VkImageSubresourceRange clearRange{};
    clearRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clearRange.baseMipLevel = 0;
    clearRange.levelCount = 1;
    clearRange.baseArrayLayer = 0;
    clearRange.layerCount = 1;
    vkCmdClearColorImage(
        resource.commandBuffer,
        resource.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &clearColor,
        1,
        &clearRange
    );

    VkImageMemoryBarrier backToGeneralBarrier{};
    backToGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    backToGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    backToGeneralBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    backToGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    backToGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    backToGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    backToGeneralBarrier.image = resource.image;
    backToGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    backToGeneralBarrier.subresourceRange.baseMipLevel = 0;
    backToGeneralBarrier.subresourceRange.levelCount = 1;
    backToGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    backToGeneralBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &backToGeneralBarrier
    );

    if (!submitFrameCommand(frame, resource, true))
        return false;

    resource.hasContent = true;
    if (!waitForFrame(frame, waitTimeoutNs))
    {
        melonDS::Platform::Log(
            melonDS::Platform::LogLevel::Error,
            "VulkanOutput: validateFrameSubmission timed out (timeoutNs=%llu)",
            static_cast<unsigned long long>(waitTimeoutNs)
        );
        return false;
    }

    return true;
}

bool VulkanOutput::validateRuntimePath(u32 width, u32 height, const melonDS::VulkanRenderer3D& renderer3D, int scale)
{
    if (!initialized || width == 0 || height == 0 || scale < 1)
        return false;

    Frame validationFrame{};
    validationFrame.backend = FrameBackend::VulkanImage;
    if (!ensureFrameResources(&validationFrame, width, height))
        return false;

    const bool validationResult = validateCompositorSubmission(&validationFrame, renderer3D, scale, kValidationWaitTimeoutNs)
        && validateFrameSubmission(&validationFrame, kValidationWaitTimeoutNs);

    destroyFrameResource(&validationFrame);
    return validationResult;
}

bool VulkanOutput::waitForFrame(const Frame* frame, u64 timeoutNs)
{
    if (!initialized || frame == nullptr || frame->backend != FrameBackend::VulkanImage)
        return false;

    if (frame->renderTimelineValue == 0)
        return false;

    const u64 waitStartNs = PerfNowNs();
    bool waitSucceeded = false;

    if (useTimelineSemaphores && waitSemaphores != nullptr && timelineSemaphore != VK_NULL_HANDLE)
    {
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &timelineSemaphore;
        waitInfo.pValues = &frame->renderTimelineValue;
        waitSucceeded = waitSemaphores(device, &waitInfo, timeoutNs) == VK_SUCCESS;
    }
    else
    {
        auto iterator = resources.find(const_cast<Frame*>(frame));
        if (iterator == resources.end())
            return false;
        waitSucceeded = vkWaitForFences(device, 1, &iterator->second.submitFence, VK_TRUE, timeoutNs) == VK_SUCCESS;
    }

    if (!waitSucceeded)
        return false;

    waitCpuWindow.Add(PerfNowNs() - waitStartNs);

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator != resources.end())
        consumeFrameGpuTiming(iterator->second);

    logPerformanceIfNeeded();
    return true;
}

bool VulkanOutput::getPreparedRenderer3dDimensions(const Frame* frame, u32& outWidth, u32& outHeight) const
{
    outWidth = 0;
    outHeight = 0;

    if (!initialized || frame == nullptr)
        return false;

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return false;

    const FrameResource& resource = iterator->second;
    if (!resource.hasPreparedInputs || !resource.hasRenderer3dSnapshot)
        return false;

    outWidth = resource.snapshotWidth;
    outHeight = resource.snapshotHeight;
    return outWidth > 0 && outHeight > 0;
}

bool VulkanOutput::getPreparedPackedBuffers(
    const Frame* frame,
    const u32*& outTopPacked,
    const u32*& outBottomPacked,
    u32& outPackedStride,
    u32& outPackedHeight,
    bool& outScreenSwap) const
{
    outTopPacked = nullptr;
    outBottomPacked = nullptr;
    outPackedStride = 0;
    outPackedHeight = 0;
    outScreenSwap = false;

    if (!initialized || frame == nullptr)
        return false;

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return false;

    const FrameResource& resource = iterator->second;
    if (!resource.hasPreparedInputs || resource.topPackedMapped == nullptr || resource.bottomPackedMapped == nullptr)
        return false;

    outTopPacked = static_cast<const u32*>(resource.topPackedMapped);
    outBottomPacked = static_cast<const u32*>(resource.bottomPackedMapped);
    outPackedStride = kAcceleratedStride;
    outPackedHeight = kScreenHeight;
    outScreenSwap = resource.screenSwap;
    return true;
}

bool VulkanOutput::isFrameReady(const Frame* frame) const
{
    if (!initialized || frame == nullptr || frame->backend != FrameBackend::VulkanImage)
        return false;

    if (frame->renderTimelineValue == 0)
        return false;

    if (useTimelineSemaphores && getSemaphoreCounterValue != nullptr && timelineSemaphore != VK_NULL_HANDLE)
    {
        u64 completedValue = 0;
        if (getSemaphoreCounterValue(device, timelineSemaphore, &completedValue) == VK_SUCCESS)
            return completedValue >= frame->renderTimelineValue;
    }

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return false;

    if (iterator->second.submitFence != VK_NULL_HANDLE)
        return vkGetFenceStatus(device, iterator->second.submitFence) == VK_SUCCESS;

    return false;
}

void VulkanOutput::consumeFrameGpuTiming(FrameResource& resource)
{
    if (!resource.timestampPending || resource.timestampQueryPool == VK_NULL_HANDLE || timestampPeriodNs <= 0.0f)
        return;

    u64 timestamps[2]{};
    const VkResult queryResult = vkGetQueryPoolResults(
        device,
        resource.timestampQueryPool,
        0,
        2,
        sizeof(timestamps),
        timestamps,
        sizeof(u64),
        VK_QUERY_RESULT_64_BIT
    );
    if (queryResult == VK_SUCCESS && timestamps[1] >= timestamps[0])
    {
        const u64 gpuTimeNs = static_cast<u64>(static_cast<double>(timestamps[1] - timestamps[0]) * static_cast<double>(timestampPeriodNs));
        compositorGpuWindow.Add(gpuTimeNs);
    }

    resource.timestampPending = false;
}

void VulkanOutput::logPerformanceIfNeeded()
{
    if (!areRendererDebugToolsEnabled())
        return;

    if (!composeCpuWindow.Ready())
        return;

    const PerfSampleWindow<120>::Summary composeSummary = composeCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary packedSummary = packedUploadCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary waitSummary = waitCpuWindow.SummarizeAndReset();
    const PerfSampleWindow<120>::Summary gpuSummary = compositorGpuWindow.SummarizeAndReset();

    melonDS::Platform::Log(
        melonDS::Platform::LogLevel::Warn,
        "VulkanPerf[Output]: compose cpu avg=%.3fms p95=%.3fms max=%.3fms packed avg=%.3fms p95=%.3fms max=%.3fms wait avg=%.3fms p95=%.3fms max=%.3fms gpu avg=%.3fms p95=%.3fms max=%.3fms",
        PerfNsToMs(composeSummary.MeanNs),
        PerfNsToMs(composeSummary.P95Ns),
        PerfNsToMs(composeSummary.MaxNs),
        PerfNsToMs(packedSummary.MeanNs),
        PerfNsToMs(packedSummary.P95Ns),
        PerfNsToMs(packedSummary.MaxNs),
        PerfNsToMs(waitSummary.MeanNs),
        PerfNsToMs(waitSummary.P95Ns),
        PerfNsToMs(waitSummary.MaxNs),
        PerfNsToMs(gpuSummary.MeanNs),
        PerfNsToMs(gpuSummary.P95Ns),
        PerfNsToMs(gpuSummary.MaxNs)
    );
}

bool VulkanOutput::readResourceImagePixels(
    FrameResource& resource,
    const Frame* frame,
    VkImage image,
    u32 width,
    u32 height,
    u32* destinationPixels,
    size_t destinationPixelCount,
    u64 waitTimeoutNs)
{
    if (!initialized || frame == nullptr || destinationPixels == nullptr || image == VK_NULL_HANDLE || width == 0 || height == 0)
        return false;

    const size_t requiredPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (destinationPixelCount < requiredPixels || resource.stagingSize < static_cast<VkDeviceSize>(requiredPixels * sizeof(u32)))
        return false;

    if (!waitForFrame(frame, waitTimeoutNs))
        return false;

    if (!beginFrameCommand(resource, waitTimeoutNs))
        return false;

    VkImageMemoryBarrier toCopyBarrier{};
    toCopyBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toCopyBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    toCopyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toCopyBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toCopyBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toCopyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toCopyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toCopyBarrier.image = image;
    toCopyBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toCopyBarrier.subresourceRange.baseMipLevel = 0;
    toCopyBarrier.subresourceRange.levelCount = 1;
    toCopyBarrier.subresourceRange.baseArrayLayer = 0;
    toCopyBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toCopyBarrier
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
    copyRegion.imageExtent.width = width;
    copyRegion.imageExtent.height = height;
    copyRegion.imageExtent.depth = 1;

    vkCmdCopyImageToBuffer(
        resource.commandBuffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        resource.stagingBuffer,
        1,
        &copyRegion
    );

    VkImageMemoryBarrier toGeneralBarrier{};
    toGeneralBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneralBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toGeneralBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    toGeneralBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toGeneralBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneralBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarrier.image = image;
    toGeneralBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneralBarrier.subresourceRange.baseMipLevel = 0;
    toGeneralBarrier.subresourceRange.levelCount = 1;
    toGeneralBarrier.subresourceRange.baseArrayLayer = 0;
    toGeneralBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toGeneralBarrier
    );

    VkBufferMemoryBarrier toHostBarrier{};
    toHostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    toHostBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    toHostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toHostBarrier.buffer = resource.stagingBuffer;
    toHostBarrier.offset = 0;
    toHostBarrier.size = resource.stagingSize;

    vkCmdPipelineBarrier(
        resource.commandBuffer,
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

    if (!submitFrameCommand(nullptr, resource, false))
        return false;

    if (vkWaitForFences(device, 1, &resource.submitFence, VK_TRUE, waitTimeoutNs) != VK_SUCCESS)
        return false;

    void* mappedMemory = nullptr;
    if (vkMapMemory(device, resource.stagingMemory, 0, resource.stagingSize, 0, &mappedMemory) != VK_SUCCESS)
        return false;

    std::memcpy(destinationPixels, mappedMemory, requiredPixels * sizeof(u32));
    vkUnmapMemory(device, resource.stagingMemory);
    return true;
}

bool VulkanOutput::readPreparedRenderer3dPixels(
    const Frame* frame,
    u32* destinationPixels,
    size_t destinationPixelCount,
    u32& outWidth,
    u32& outHeight,
    u64 waitTimeoutNs)
{
    outWidth = 0;
    outHeight = 0;

    if (!initialized || frame == nullptr)
        return false;

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return false;

    FrameResource& resource = iterator->second;
    if (!resource.hasPreparedInputs || !resource.hasRenderer3dSnapshot)
        return false;

    outWidth = resource.snapshotWidth;
    outHeight = resource.snapshotHeight;
    return readResourceImagePixels(
        resource,
        frame,
        resource.renderer3dSnapshot,
        resource.snapshotWidth,
        resource.snapshotHeight,
        destinationPixels,
        destinationPixelCount,
        waitTimeoutNs);
}

bool VulkanOutput::readFramePixels(const Frame* frame, u32* destinationPixels, size_t destinationPixelCount, u64 waitTimeoutNs)
{
    if (!initialized || frame == nullptr || destinationPixels == nullptr)
        return false;

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return false;

    FrameResource& resource = iterator->second;
    return readResourceImagePixels(
        resource,
        frame,
        resource.image,
        resource.width,
        resource.height,
        destinationPixels,
        destinationPixelCount,
        waitTimeoutNs);
}

VkImage VulkanOutput::getFrameImage(const Frame* frame) const
{
    if (frame == nullptr)
        return VK_NULL_HANDLE;

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return VK_NULL_HANDLE;

    return iterator->second.image;
}

VkImageView VulkanOutput::getFrameImageView(const Frame* frame) const
{
    if (frame == nullptr)
        return VK_NULL_HANDLE;

    auto iterator = resources.find(const_cast<Frame*>(frame));
    if (iterator == resources.end())
        return VK_NULL_HANDLE;

    return iterator->second.imageView;
}

}
