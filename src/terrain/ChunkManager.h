#pragma once

#include <cstdint>
#include <future>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ChunkCoord.h"
#include "DiamondSquareGenerator.h"
#include "TerrainGeometry.h"

// Bits into LoadedChunk::missingNeighborMask, one per cardinal direction.
enum NeighborBit : uint8_t {
    kNeighborLeft = 1u << 0,
    kNeighborRight = 1u << 1,
    kNeighborTop = 1u << 2,
    kNeighborBottom = 1u << 3,
};

/*!
 *	One loaded chunk's GPU geometry, with an optional in-progress blend from an earlier Hurst/seed
 *	value (`from`, valid only while blending) to its current one (`to`, always valid once loaded).
 */
struct LoadedChunk {
    Geometry from;
    Geometry to;

    // The chunk's other persistent vertex buffer — its ping-pong partner to `to`. A chunk's vertex
    // count is fixed after first load, so regeneration alternates writing into these two buffers
    // instead of allocating a new one each time. VK_NULL_HANDLE until the chunk's first
    // regeneration (allocated lazily then). While blending, duplicates `from.vertexBuffer`.
    VkBuffer idleVertexBuffer = VK_NULL_HANDLE;

    double blendStartTime;

    // Which of this chunk's neighbors (see NeighborBit) were missing — outside the loaded window,
    // not merely still generating — the last time `to`'s normals were computed, and so had that edge
    // extrapolated instead of using real neighbor data. Cleared bit-by-bit as updateLoadedChunks
    // notices a previously-missing neighbor has since loaded and patches that edge in.
    uint8_t missingNeighborMask = 0;
};

/*!
 *	A chunk's GPU geometry that's no longer referenced by `loadedChunks` (evicted — a finished
 *	blend's `from` destroys nothing; see LoadedChunk::idleVertexBuffer), queued for actual
 *	destruction rather than freed immediately. Destroying it while still referenced by an
 *	already-submitted command buffer would be a GPU-side use-after-free; a `vkDeviceWaitIdle` before
 *	every destroy would guard against that but drains the entire GPU pipeline (measured live: ~30ms
 *	per call). Instead, `callsRemaining` (decremented once per updateLoadedChunks call, freed at 0)
 *	relies on the per-frame fence wait (`vklWaitForNextSwapchainImage()`) that already runs between
 *	every pair of updateLoadedChunks calls: with `CONCURRENT_FRAMES == 1` in VulkanLaunchpad.cpp,
 *	that fence wait alone guarantees the previous command buffer has finished — the same pattern
 *	VulkanLaunchpad.cpp itself uses for retiring hot-reloaded pipelines (`mPipelineGraveyard`).
 */
struct PendingDestroy {
    Geometry geometry;
    int callsRemaining;
};

/*!
 *	Creates and uploads an index buffer. Called exactly once, ever, for the whole ChunkManager (see
 *	ChunkManager::sharedIndicesBuffer) — a chunk's triangle topology is a pure function of grid size
 *	alone (`gridSizeExponent`, fixed for the app's lifetime), so every chunk produces identical
 *	index data and one shared buffer serves all of them.
 *
 *	@param	indices	The triangle index data to upload. Must not be empty.
 *	@return	A handle to the newly created, uploaded index buffer.
 */
VkBuffer createAndUploadIndexBuffer(const std::vector<uint32_t>& indices);

/*!
 *	Owns the set of terrain chunks currently loaded around the camera, generating new ones on
 *	demand and discarding distant ones. Generation is split into two background-threaded phases:
 *	phase 1 (pendingChunks) computes a chunk's heights/positions/indices independently per chunk,
 *	and phase 2 (pendingNormals) derives normals using boundary data from whichever neighbors have
 *	also finished phase 1.
 */
struct ChunkManager {
    TerrainParams baseParams;

    // One index buffer, shared by every currently-loaded chunk (see createAndUploadIndexBuffer).
    // Created lazily on the first chunk load; VK_NULL_HANDLE until then. Freed once, at
    // cleanupChunkManager.
    VkBuffer sharedIndicesBuffer = VK_NULL_HANDLE;
    uint32_t sharedNumberOfIndices = 0;

    int viewRadius = 8;

    // Duration of a per-chunk Hurst/reseed blend, in the same units as updateLoadedChunks's currentTime.
    double blendDuration = 2.0;

    // Caps how many pendingDestroys entries updateLoadedChunks actually frees (once their deferral
    // has elapsed) in one call — a mass regeneration or chunk-border crossing can make a whole batch
    // eligible for destruction at once, and each free is real GPU work.
    int maxDestroysPerFrame = 16;

    // Caps how many brand-new (never-before-loaded) chunks updateLoadedChunks uploads in one call —
    // travelling fast (or a large view radius) can bring a whole ring of new chunks into range at
    // once. A first-time load never blends (see LoadedChunk), so a chunk left past the cap has
    // nothing to desync. Regeneration uploads are left uncapped, since capping those would need
    // batch-sync bookkeeping to avoid desyncing blends.
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

    // Parallel to pendingNormals: the missingNeighborMask each dispatch in there was computed with,
    // carried through so it can be stored on the LoadedChunk once that upload completes.
    std::unordered_map<ChunkCoord, uint8_t> pendingNormalsMask;

    // Re-derivation of an already-loaded chunk's normals, for a chunk whose missingNeighborMask
    // indicated a real neighbor has since become available. Separate from pendingNormals since this
    // patches an existing chunk in place (see updateGeometryNormals) rather than producing a new
    // upload, and — unlike pendingNormals — being in flight here doesn't count as isRegenerating():
    // it's a background refinement of stable geometry, not a Hurst/reseed change in progress.
    std::unordered_map<ChunkCoord, std::future<std::vector<glm::vec3>>> pendingRenormals;

    // Geometry evicted, not yet actually freed. See PendingDestroy.
    std::vector<PendingDestroy> pendingDestroys;
};

/*!
 *	Call once per frame: advances chunk generation/upload/destroy around the camera, evicting
 *	anything more than (viewRadius + 1) chunks away. Uploads at most `maxUploadsPerFrame` newly-loaded
 *	chunks per call (regeneration uploads are uncapped — see LoadedChunk::idleVertexBuffer for why
 *	those stay cheap in steady state). An evicted chunk is queued in `pendingDestroys` rather than
 *	freed immediately (see that struct's doc); up to `maxDestroysPerFrame` eligible entries are
 *	actually freed per call. Also revisits any loaded chunk with a nonzero `missingNeighborMask`,
 *	re-deriving its normals in the background once a previously-missing neighbor has real data (see
 *	LoadedChunk::missingNeighborMask and pendingRenormals).
 *
 *	`currentTime` should be the same clock used when drawing (glfwGetTime()).
 */
void updateLoadedChunks(ChunkManager& manager, const glm::vec3& cameraPos, double currentTime);

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
 *	Hurst/reseed change while this is true. Deliberately ignores `pendingRenormals` — a background
 *	edge-normal touch-up on already-stable geometry, not a Hurst/reseed change in progress.
 */
bool isRegenerating(const ChunkManager& manager);

/*!
 *	Destroys the GPU resources for every currently-loaded chunk, plus anything still sitting in
 *	`pendingDestroys` regardless of its remaining deferral. Call once at shutdown, after a
 *	`vkDeviceWaitIdle` — unlike updateLoadedChunks's own deferred destruction, this does not wait for
 *	anything itself.
 */
void cleanupChunkManager(ChunkManager& manager);
