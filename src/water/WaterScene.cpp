#include "WaterScene.h"

#include <VulkanLaunchpad.h>

#include "../utils/PathUtils.h"
#include "../utils/VulkanSetup.h"

WaterScene setupWaterScene(VkDevice vk_device, const TerrainParams& terrain_params) {
    WaterScene scene{};

    scene.vertexShaderPath = gcgFindShaderFile("assets/shaders/water.vert");
    scene.fragmentShaderPath = gcgFindShaderFile("assets/shaders/water.frag");

    std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings = {
        VkDescriptorSetLayoutBinding{0u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
    };

    VklGraphicsPipelineConfig pipeline_config{
        scene.vertexShaderPath.c_str(),
        scene.fragmentShaderPath.c_str(),
        {VkVertexInputBindingDescription{0u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX}},
        {VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, 0u}},
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        descriptor_set_layout_bindings,
        /* enableAlphaBlending: */ true,
    };
    scene.pipeline = vklCreateGraphicsPipeline(pipeline_config);

    std::vector<VkDescriptorPoolSize> pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u}};
    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.maxSets = 1u;
    descriptor_pool_create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    descriptor_pool_create_info.pPoolSizes = pool_sizes.data();
    VkResult result = vkCreateDescriptorPool(vk_device, &descriptor_pool_create_info, nullptr, &scene.descriptor_pool);
    VKL_CHECK_VULKAN_RESULT(result);

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
    descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_set_layout_create_info.bindingCount = static_cast<uint32_t>(descriptor_set_layout_bindings.size());
    descriptor_set_layout_create_info.pBindings = descriptor_set_layout_bindings.data();
    result = vkCreateDescriptorSetLayout(vk_device, &descriptor_set_layout_create_info, nullptr, &scene.descriptor_set_layout);
    VKL_CHECK_VULKAN_RESULT(result);

    scene.ub_water_vert = vklCreateHostCoherentBufferWithBackingMemory(
        sizeof(UniformBufferWaterVert),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    scene.ds_water = allocDescriptorSet(vk_device, scene.descriptor_pool, scene.descriptor_set_layout);

    VkDescriptorBufferInfo vert_buffer_info = {};
    vert_buffer_info.buffer = scene.ub_water_vert;
    vert_buffer_info.offset = static_cast<VkDeviceSize>(0);
    vert_buffer_info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        scene.ds_water,
        /* dstBinding: */ 0u,
        0u,
        1u,
        /* descriptorType: */ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        nullptr,
        /* pBufferInfo: */ &vert_buffer_info,
        nullptr
    };
    vkUpdateDescriptorSets(vk_device, 1u, &write, 0u, nullptr);

    // chunkGeometry is populated lazily by updateWaterChunks() as chunks load.
    return scene;
}

void updateWaterChunks(VkDevice vk_device, WaterScene& scene, const ChunkManager& chunkManager) {
    // Create a water quad for any newly-loaded terrain chunk.
    for (auto& entry : chunkManager.loadedChunks) {
        const ChunkCoord& coord = entry.first;
        if (scene.chunkGeometry.count(coord)) continue;
        scene.chunkGeometry[coord] = buildWaterChunkGeometry(coord, chunkManager.baseParams);
    }

    // Destroy the water quad for any chunk that's no longer loaded. Wait for the GPU to finish first
    // — a previous frame's command buffer may still be referencing these buffers.
    bool destroyed_any = false;
    for (auto it = scene.chunkGeometry.begin(); it != scene.chunkGeometry.end();) {
        if (!chunkManager.loadedChunks.count(it->first)) {
            if (!destroyed_any) {
                vkDeviceWaitIdle(vk_device);
                destroyed_any = true;
            }
            destroyWaterChunkGeometry(it->second);
            it = scene.chunkGeometry.erase(it);
        } else {
            ++it;
        }
    }
}

void cleanupWaterScene(VkDevice vk_device, WaterScene& scene) {
    for (auto& entry : scene.chunkGeometry) {
        destroyWaterChunkGeometry(entry.second);
    }
    vkDestroyDescriptorSetLayout(vk_device, scene.descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, scene.descriptor_pool, nullptr);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_water_vert);
    vklDestroyGraphicsPipeline(scene.pipeline);
}
