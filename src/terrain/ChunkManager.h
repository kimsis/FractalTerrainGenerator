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
 *	A chunk's GPU geometry that's no longer referenced by `loadedChunks` (evicted or a finished
 *	blend's `from`), queued for actual destruction rather than freed immediately. Destroying it while
 *	still referenced by an already-submitted command buffer would be a GPU-side use-after-free; the
 *	naive guard is a synchronous `vkDeviceWaitIdle` before every destroy, but that drains the entire
 *	GPU pipeline and is expensive enough on its own to cause a visible stutter (measured live: ~30ms
 *	for one such call). Instead, `callsRemaining` (decremented once per updateLoadedChunks call, freed
 *	once it reaches 0) leans on this app's own per-frame fence wait (`vklWaitForNextSwapchainImage()`,
 *	called once between every pair of updateLoadedChunks calls): with `CONCURRENT_FRAMES == 1` in
 *	VulkanLaunchpad.cpp, that fence wait alone already guarantees the GPU has finished the previous
 *	command buffer by the next updateLoadedChunks call, so no extra wait is needed at all — the same
 *	pattern VulkanLaunchpad.cpp itself uses for retiring hot-reloaded pipelines (`mPipelineGraveyard`).
 */
struct PendingDestroy {
    Geometry geometry;
    int callsRemaining;
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

    // Caps how many pendingDestroys entries updateLoadedChunks actually frees (once their deferral
    // has elapsed) in one call. A mass Hurst/reseed regeneration or a single chunk-border crossing
    // can each make a whole ring/batch of chunks eligible for destruction at once, and each free is
    // real GPU work; anything past the cap just waits one more call.
    int maxDestroysPerFrame = 16;

    // Caps how many brand-new (never-before-loaded) chunks updateLoadedChunks uploads in one call —
    // travelling fast enough (or a large view radius) can bring a whole ring of new chunks into range
    // at once, and each upload is real, synchronous GPU work. A chunk left past the cap just stays
    // queued for a later call; it has no blend to desync since a first-time load never blends (see
    // LoadedChunk). Regeneration uploads (an already-loaded chunk starting a new blend, e.g. from a
    // Hurst/reseed change) are deliberately left uncapped — that stutter is accepted as a tradeoff for
    // not needing the batch-sync bookkeeping a regeneration cap would require to avoid desyncing blends.
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

    // Geometry evicted or retired from a finished blend, not yet actually freed. See PendingDestroy.
    std::vector<PendingDestroy> pendingDestroys;
};

/*!
 *	Call once per frame: advances chunk generation/upload/destroy around the camera, evicting
 *	anything more than (viewRadius + 1) chunks away. Uploads at most `maxUploadsPerFrame` newly-loaded
 *	chunks per call (regeneration uploads are uncapped). An evicted chunk or a finished blend's `from`
 *	is queued in `pendingDestroys` rather than freed immediately (see that struct's doc for why); up
 *	to `maxDestroysPerFrame` eligible entries are actually freed per call.
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
 *	Hurst/reseed change while this is true.
 */
bool isRegenerating(const ChunkManager& manager);

/*!
 *	Destroys the GPU resources for every currently-loaded chunk, plus anything still sitting in
 *	`pendingDestroys` regardless of its remaining deferral. Call once at shutdown, after a
 *	`vkDeviceWaitIdle` — unlike updateLoadedChunks's own deferred destruction, this does not wait for
 *	anything itself.
 */
void cleanupChunkManager(ChunkManager& manager);
