#include "TerrainScene.h"

#include <VulkanLaunchpad.h>

VkPipeline buildTerrainPipeline(const TerrainScene& scene, size_t polygon_mode_index, size_t cull_mode_index) {
    VklGraphicsPipelineConfig pipeline_config{
        scene.vertexShaderPath.c_str(),
        scene.fragmentShaderPath.c_str(),
        {
            VkVertexInputBindingDescription{0u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
            VkVertexInputBindingDescription{1u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
            VkVertexInputBindingDescription{2u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
            VkVertexInputBindingDescription{3u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
        },
        {
            VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
            VkVertexInputAttributeDescription{1u, 1u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
            VkVertexInputAttributeDescription{2u, 2u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
            VkVertexInputAttributeDescription{3u, 3u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
        },
        TERRAIN_POLYGON_MODES[polygon_mode_index],
        TERRAIN_CULL_MODES[cull_mode_index],
        scene.descriptorSetLayoutBindings,
        /* enableAlphaBlending: */ false,
        {VkPushConstantRange{VK_SHADER_STAGE_VERTEX_BIT, 0u, sizeof(TerrainPushConstants)}},
    };
    return vklCreateGraphicsPipeline(pipeline_config);
}

void drawGeometryWithMaterial(
    VkPipeline pipeline,
    const LoadedChunk& chunk,
    VkBuffer indices_buffer,
    uint32_t number_of_indices,
    VkDescriptorSet material,
    uint32_t num_instances
) {
    VkCommandBuffer cb = vklGetCurrentCommandBuffer();

    VkPipelineLayout pipeline_layout = vklGetLayoutForPipeline(pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0u, 1u, &material, 0u, nullptr);

    vklCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    // Not blending: `from` is empty, so substitute `to` in its place.
    bool is_blending = chunk.from.vertexBuffer != VK_NULL_HANDLE;
    const Geometry& geometry_from = is_blending ? chunk.from : chunk.to;
    // Positions and normals share one combined buffer per Geometry — the same VkBuffer is bound
    // twice here, once per offset, which Vulkan allows.
    VkBuffer vertex_buffers[4] = {geometry_from.vertexBuffer, chunk.to.vertexBuffer, geometry_from.vertexBuffer, chunk.to.vertexBuffer};
    VkDeviceSize offsets[4] = {0, 0, geometry_from.normalsOffset, chunk.to.normalsOffset};
    vkCmdBindVertexBuffers(cb, 0u, 4u, vertex_buffers, offsets);

    vkCmdBindIndexBuffer(cb, indices_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, number_of_indices, num_instances, 0u, 0u, 0u);
}

void cleanupTerrainScene(VkDevice vk_device, TerrainScene& scene) {
    vkDestroyDescriptorSetLayout(vk_device, scene.descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, scene.descriptor_pool, nullptr);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_terrain_vert);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_terrain_frag);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_dirlight);

    for (size_t i = 0; i < POLYMODES; ++i) {
        for (size_t j = 0; j < CULLMODES; ++j) {
            vklDestroyGraphicsPipeline(scene.pipelines[i][j]);
        }
    }
    cleanupChunkManager(scene.chunkManager);
}
