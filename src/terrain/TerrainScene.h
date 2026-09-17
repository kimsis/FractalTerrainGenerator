#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "ChunkManager.h"

constexpr size_t POLYMODES = 2;
constexpr size_t CULLMODES = 3;
constexpr VkPolygonMode kTerrainPolygonModes[POLYMODES] = {VK_POLYGON_MODE_FILL, VK_POLYGON_MODE_LINE};
constexpr VkCullModeFlags kTerrainCullModes[CULLMODES] = {VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT};

/*!
 *	It matches the definition and sizes of the corresponding GPU-side struct exactly, which is used in shaders.
 */
struct UniformBufferVert {
    /*! Storage for the model matrix, consisting of 16 float values (inherently aligned to 16 bytes) */
    glm::mat4 modelMatrix;

    /*! Storage for the model matrix suitable for normals transformation, consisting of 16 float values (inherently aligned to 16 bytes) */
    glm::mat4 modelMatrixForNormals;

    /*! Storage for the view-projection matrix, consisting of 16 float values (inherently aligned to 16 bytes) */
    glm::mat4 viewProjMatrix;
};

/*!
 *	Per-chunk blend state for terrain.vert, pushed via vkCmdPushConstants immediately before each
 *	chunk's draw call.
 */
struct TerrainPushConstants {
    /*! 0-1 blend progress between this chunk's `from` and `to` geometry. */
    float blendFactor;

    /*! Whether this chunk is currently blending (`from` is valid). */
    uint32_t isBlending;
};

struct UniformBufferFrag {
    /*! Storage for the camera's world space position (aligned to 16 bytes) */
    glm::vec4 cameraPosition;

    /*! Illumination properties ka, kd, ks, alpha (in that order)
     *	First three are material coefficients, the last one is specular alpha. */
    glm::vec4 materialProperties;

    /*! Debug visualization toggles: x = drawNormals (N), y = highlightChunkBorders (F3). */
    glm::uvec2 debugToggles;

    /*! Whether the camera is below the water plane; drives the underwater tint in terrain.frag. */
    uint32_t isUnderwater;

    /*! World-space width of one terrain chunk, for the chunk-border-highlight distance check. */
    float chunkWidth;

    /*! Surface roughness in [0, 1]: 0 is sharp/shiny, 1 is broad/matte. See buildGUI's slider. */
    float roughness;

    /*! World-space heights (not scaled by heightScale) where the height-based color gradient
     *	transitions dirt->grass and grass->rock. See g_dirt_to_grass_height_offset/g_grass_to_rock_height_offset. */
    float dirtToGrassHeight;
    float grassToRockHeight;

    /*! Half-width of each color transition above, same units as dirtToGrassHeight. */
    float heightColorTransitionBand;
};

/*!
 *	Holds every GPU resource (pipelines, geometries, uniform buffers, descriptor sets, textures)
 *	that make up the terrain scene (Terrain + dir light).
 */
struct TerrainScene {
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    /*! Pipelines are built lazily, on first use of each (polygon mode, cull mode) combination. */
    VkPipeline pipelines[POLYMODES][CULLMODES];
    std::string vertexShaderPath;
    std::string fragmentShaderPath;
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;

    VkBuffer ub_dirlight;

    VkBuffer ub_terrain_vert;
    VkBuffer ub_terrain_frag;
    VkDescriptorSet ds_terrain;

    ChunkManager chunkManager;

    float heightScale;
    float waterLevel;
    float roughness;
};

/*!
 *	Builds (compiles + creates) the terrain pipeline for one (polygon mode, cull mode) combination,
 *	identified by their indices into kTerrainPolygonModes/kTerrainCullModes.
 *	@param	scene				The terrain scene to build the pipeline for (shaders, descriptor set layout bindings).
 *	@param	polygon_mode_index	Index into kTerrainPolygonModes selecting the pipeline's polygon (wireframe) mode.
 *	@param	cull_mode_index		Index into kTerrainCullModes selecting the pipeline's culling mode.
 *	@return		A valid VkPipeline handle for the requested (polygon mode, cull mode) combination.
 */
VkPipeline buildTerrainPipeline(const TerrainScene& scene, size_t polygon_mode_index, size_t cull_mode_index);

/*!
 *	Bind the given descriptor set to use the material it represents for subsequent draw calls
 *	with the given pipeline, and render the given chunk (using its vertex and index buffers).
 *	Record everything into the current command buffer as provided by the framework.
 *	@param	pipeline		Valid handle to a given pipeline which shall be used for drawing.
 *	@param	chunk			The loaded chunk to draw. Which of its two vertex buffers plays the
 *							"from" role is derived internally from `chunk.from`'s validity.
 *	@param	indices_buffer		The shared index buffer — see ChunkManager::sharedIndicesBuffer.
 *	@param	number_of_indices	How many indices to draw from indices_buffer.
 *	@param	material		Valid handle to a descriptor set that refers to resources that contain material properties.
 *	@param	num_instances	How many instances to draw of the given geometry. Default = one single instance.
 */
void drawGeometryWithMaterial(
    VkPipeline pipeline,
    const LoadedChunk& chunk,
    VkBuffer indices_buffer,
    uint32_t number_of_indices,
    VkDescriptorSet material,
    uint32_t num_instances = 1u
);

/*!
 *	Destroys all GPU resources owned by the given terrain scene.
 *	@param	vk_device	Valid handle to the logical device the resources were created on.
 *	@param	scene		The terrain scene whose GPU resources shall be destroyed.
 */
void cleanupTerrainScene(VkDevice vk_device, TerrainScene& scene);
