#pragma once

#include <cstdint>
#include <future>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../algorithms/DiamondSquareGenerator.h"
#include "ChunkCoord.h"
#include "TerrainGeometry.h"

// Bits into LoadedChunk::missingNeighborMask, one per cardinal direction.
enum NeighborBit : uint8_t {
    NEIGHBOR_LEFT = 1u << 0,
    NEIGHBOR_RIGHT = 1u << 1,
    NEIGHBOR_TOP = 1u << 2,
    NEIGHBOR_BOTTOM = 1u << 3,
};

/*!
 *	One loaded chunk's GPU geometry, with an optional in-progress blend from an earlier Hurst/seed
 *	value (`from`, valid only while blending) to its current one (`to`, always valid once loaded).
 */
struct LoadedChunk {
    Geometry from;
    Geometry to;

    // Ping-pong partner to `to`. VK_NULL_HANDLE until the chunk's first regeneration. While
    // blending, duplicates `from.vertexBuffer`.
    VkBuffer idleVertexBuffer = VK_NULL_HANDLE;

    double blendStartTime;

    // Which of this chunk's neighbors (see NeighborBit) were missing the last time `to`'s normals
    // were computed, so that edge was extrapolated instead of using real neighbor data. Cleared
    // bit-by-bit as updateLoadedChunks patches in a neighbor once it loads.
    uint8_t missingNeighborMask = 0;
};

/*!
 *	A chunk's GPU geometry evicted from `loadedChunks`, queued for destruction rather than freed
 *	immediately. `callsRemaining` is decremented once per updateLoadedChunks call and the geometry
 *	is freed once it reaches 0.
 */
struct PendingDestroy {
    Geometry geometry;
    int callsRemaining;
};

/*!
 *	Creates and uploads an index buffer. Called exactly once for the whole ChunkManager.
 *
 *	@param	indices	The triangle index data to upload. Must not be empty.
 *	@return	A handle to the newly created, uploaded index buffer.
 */
VkBuffer createAndUploadIndexBuffer(const std::vector<uint32_t>& indices);

/*!
 *	Kicks off terrain generation for the given params on a background thread.
 */
std::future<GeometryData> startTerrainGeneration(const TerrainParams& params);

/*!
 *	Owns the set of terrain chunks currently loaded around the camera, generating new ones on
 *	demand and discarding distant ones. Generation is split into two background-threaded phases:
 *	phase 1 (pendingChunks) computes a chunk's heights/positions/indices independently per chunk,
 *	and phase 2 (pendingNormals) derives normals using boundary data from whichever neighbors have
 *	also finished phase 1.
 */
struct ChunkManager {
    TerrainParams baseParams;

    // Index buffer shared by every currently-loaded chunk. Created lazily on the first chunk load;
    // VK_NULL_HANDLE until then. Freed once, at cleanupChunkManager.
    VkBuffer sharedIndicesBuffer = VK_NULL_HANDLE;
    uint32_t sharedNumberOfIndices = 0;

    int viewRadius = 16;

    // Duration of a per-chunk Hurst/reseed blend, in the same units as updateLoadedChunks's currentTime.
    double blendDuration = 2.0;

    // Caps how many pendingDestroys entries updateLoadedChunks frees per call.
    int maxDestroysPerFrame = 16;

    // Caps how many newly-loaded chunks updateLoadedChunks uploads per call. Regeneration uploads
    // are uncapped.
    int maxUploadsPerFrame = 8;

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

    // Parallel to pendingNormals: the missingNeighborMask each dispatch was computed with.
    std::unordered_map<ChunkCoord, uint8_t> pendingNormalsMask;

    // Re-derivation of an already-loaded chunk's normals once a missing neighbor becomes available.
    // Unlike pendingNormals, being in flight here doesn't count as isRegenerating().
    std::unordered_map<ChunkCoord, std::future<std::vector<glm::vec3>>> pendingRenormals;

    // Geometry evicted, not yet actually freed. See PendingDestroy.
    std::vector<PendingDestroy> pendingDestroys;
};

/*!
 *	Call once per frame: advances chunk generation/upload/destroy around the camera, evicting
 *	anything more than (viewRadius + 1) chunks away. Uploads at most `maxUploadsPerFrame` newly-loaded
 *	chunks per call. An evicted chunk is queued in `pendingDestroys` rather than freed immediately;
 *	up to `maxDestroysPerFrame` eligible entries are actually freed per call. Also revisits any
 *	loaded chunk with a nonzero `missingNeighborMask`, re-deriving its normals in the background once
 *	a previously-missing neighbor has real data.
 *
 *	`currentTime` should be the same clock used when drawing (glfwGetTime()).
 */
void updateLoadedChunks(ChunkManager& manager, const glm::vec3& cameraPos, double currentTime);

/*!
 *	Call after changing `manager.baseParams.hurst` or `.seed`. Clears `chunkData` for every loaded
 *	coordinate (without touching GPU resources), so the next updateLoadedChunks call regenerates
 *	them under the new params and blends into the result once ready.
 *
 *	Callers must check `!isRegenerating(manager)` first.
 */
void invalidateAllLoadedChunks(ChunkManager& manager);

/*!
 *	True while any chunk is still mid-generation or mid-blend. Callers should refuse a new
 *	Hurst/reseed change while this is true. Ignores `pendingRenormals`.
 */
bool isRegenerating(const ChunkManager& manager);

/*!
 *	Destroys the GPU resources for every currently-loaded chunk, plus anything still sitting in
 *	`pendingDestroys` regardless of its remaining deferral. Call once at shutdown, after a
 *	`vkDeviceWaitIdle`.
 */
void cleanupChunkManager(ChunkManager& manager);
