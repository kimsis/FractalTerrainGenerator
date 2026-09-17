#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <unordered_map>

#include "../algorithms/DiamondSquareGenerator.h"
#include "../terrain/ChunkCoord.h"
#include "../terrain/ChunkManager.h"
#include "WaterGeometry.h"

/*!
 *	Matches water.vert's uniform block exactly. No fragment-stage uniforms are needed since
 *	water.frag uses a fixed color.
 */
struct UniformBufferWaterVert {
    glm::mat4 modelMatrix;
    glm::mat4 viewProjMatrix;
};

/*!
 *	Holds every GPU resource that makes up the water plane: one pipeline/descriptor set/uniform
 *	buffer shared by all chunks, and one small baked quad per currently-loaded terrain chunk.
 */
struct WaterScene {
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    VkPipeline pipeline;
    std::string vertexShaderPath;
    std::string fragmentShaderPath;

    VkBuffer ub_water_vert;
    VkDescriptorSet ds_water;

    std::unordered_map<ChunkCoord, WaterChunkGeometry> chunkGeometry;
};

/*!
 *	Builds the water plane's shared pipeline, uniform buffer, and descriptor set.
 *	@param	vk_device		Valid handle to the logical device to create the resources on.
 *	@param	terrain_params	The terrain's generation parameters (grid size/spacing) the water quads are baked to match.
 *	@return		A fully set-up WaterScene.
 */
WaterScene setupWaterScene(VkDevice vk_device, const TerrainParams& terrain_params);

/*!
 *	Creates a small world-space-baked water quad for any newly-loaded terrain chunk, and destroys
 *	it for any chunk that's since been unloaded. Call once per frame, right after
 *	updateLoadedChunks(), outside vklStartRecordingCommands()/vklEndRecordingCommands().
 *	@param	vk_device		Valid handle to the logical device to create/destroy resources on.
 *	@param	scene			The water scene whose per-chunk geometry shall be updated.
 *	@param	chunkManager	The terrain's chunk manager, used to determine which chunks are currently loaded.
 */
void updateWaterChunks(VkDevice vk_device, WaterScene& scene, const ChunkManager& chunkManager);

/*!
 *	Destroys all GPU resources owned by the given water scene.
 *	@param	vk_device	Valid handle to the logical device the resources were created on.
 *	@param	scene		The water scene whose GPU resources shall be destroyed.
 */
void cleanupWaterScene(VkDevice vk_device, WaterScene& scene);
