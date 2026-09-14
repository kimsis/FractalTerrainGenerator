#pragma once

#include <future>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>

#include "ChunkCoord.h"
#include "DiamondSquareGenerator.h"
#include "TerrainGeometry.h"

/*!
 *	One loaded chunk's GPU geometry, with an optional in-progress blend from an earlier Hurst/seed
 *	value (`from`, valid only while blending) to its current one (`to`, always valid once loaded).
 */
struct LoadedChunk {
    Geometry from;
    Geometry to;
    double blendStartTime;
};

/*!
 *	Owns the set of terrain chunks currently loaded around the camera, generating new ones on
 *	demand and discarding distant ones. Generation is split into two background-threaded phases:
 *	phase 1 (pendingChunks) computes a chunk's heights/positions/indices independently per chunk,
 *	and phase 2 (pendingNormals) derives normals using boundary data from whichever neighbors have
 *	also finished phase 1.
 */
struct ChunkManager {
    TerrainParams baseParams;

    int viewRadius = 8;

    // Duration of a per-chunk Hurst/reseed blend, in the same units as updateLoadedChunks's currentTime.
    double blendDuration = 2.0;

    // Final, GPU-uploaded chunks, each with its own independent from/to blend state.
    std::unordered_map<ChunkCoord, LoadedChunk> loadedChunks;

    // CPU-side heights/positions/indices, retained for as long as a chunk exists.
    std::unordered_map<ChunkCoord, GeometryData> chunkData;

    // Phase 1 (heights/positions/indices) in flight.
    std::unordered_map<ChunkCoord, std::future<GeometryData>> pendingChunks;

    // Phase 1 done, phase 2 not yet dispatched.
    std::unordered_set<ChunkCoord> readyForNormals;

    // Phase 2 (normal derivation) in flight.
    std::unordered_map<ChunkCoord, std::future<std::vector<glm::vec3>>> pendingNormals;
};

/*!
 *	Call once per frame: advances chunk generation/upload/destroy around the camera, evicting
 *	anything more than (viewRadius + 1) chunks away. Waits for the GPU to go idle first if anything
 *	is actually destroyed this call.
 *
 *	`currentTime` should be the same clock used when drawing (glfwGetTime()).
 */
void updateLoadedChunks(VkDevice vk_device, ChunkManager& manager, const glm::vec3& cameraPos, double currentTime);

/*!
 *	Call after changing `manager.baseParams.hurst` or `.seed`. Clears `chunkData` for every loaded
 *	coordinate (without touching GPU resources), so the next updateLoadedChunks call regenerates
 *	them under the new params and blends into the result once ready.
 *
 *	Callers must check `!isRegenerating(manager)` first — `updateLoadedChunks` assumes a chunk's
 *	`from` is always empty at the moment a new one is about to be assigned.
 */
void invalidateAllLoadedChunks(ChunkManager& manager);

/*!
 *	True while any chunk is still mid-generation or mid-blend. Callers should refuse a new
 *	Hurst/reseed change while this is true.
 */
bool isRegenerating(const ChunkManager& manager);

/*!
 *	Destroys the GPU resources for every currently-loaded chunk. Call once at shutdown.
 */
void cleanupChunkManager(ChunkManager& manager);
