#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <glm/glm.hpp>

#include "../terrain/ChunkManager.h"

/*!
 *	How many chunks chunkManager still has to generate/upload before it's fully caught up.
 *	@param	chunkManager	The chunk manager to query.
 *	@return		The number of chunks still pending generation, normal-computation, or upload.
 */
size_t chunksStillGenerating(const ChunkManager& chunkManager);

/*!
 *	Builds a minimal ImGui panel shown while the initial chunks are generating, before the real
 *	terrain scene's own controls exist yet.
 *	@param	pendingChunkCount	How many chunks are still pending, shown in the panel's text.
 */
void buildLoadingGUI(size_t pendingChunkCount);

/*!
 *	Blocks until every chunk in the initial (2 * viewRadius + 1)^2 window around cameraPos has been
 *	generated and uploaded, while keeping the window responsive (polling events and drawing a
 *	loading screen) for however long that takes. Use this once, at startup, before the main render
 *	loop begins — per-frame streaming (updateLoadedChunks) takes over from there.
 *	@param	vk_device		Valid handle to the logical device to record/submit loading-screen frames on.
 *	@param	chunkManager	The chunk manager whose initial chunks shall be generated.
 *	@param	cameraPos		The camera position the initial chunk window is centered on.
 */
void generateTerrainGeometryWithLoadingScreen(VkDevice vk_device, ChunkManager& chunkManager, const glm::vec3& cameraPos);
