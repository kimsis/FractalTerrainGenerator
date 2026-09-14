#pragma once

#include <future>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>

#include "ChunkCoord.h"
#include "DiamondSquareGenerator.h"
#include "TerrainGeometry.h"

/*!
 *	One loaded chunk's GPU geometry, with an optional in-progress blend: `to` is always valid once
 *	loaded; `from` is only valid (non-VK_NULL_HANDLE buffers) while this chunk is blending from an
 *	earlier Hurst/seed value to its current one, and is destroyed once that finishes.
 */
struct LoadedChunk {
    Geometry from;
    Geometry to;
    double blendStartTime;
};

/*!
 *	Owns the set of terrain chunks currently loaded around the camera, generating new ones on
 *	demand and discarding distant ones, so memory/generation cost stays bounded by viewRadius
 *	regardless of how far the camera has traveled.
 *
 *	Generation is split into two phases, each of which runs on background threads: phase 1
 *	(pendingChunks) computes a chunk's heights/positions/indices — fully independent between
 *	chunks, since it's a pure function of (seed, global coords, level) — and phase 2
 *	(pendingNormals) derives that chunk's normals using real boundary data borrowed from whichever
 *	neighbors have *also* finished phase 1, once they have. This avoids both a visible seam (no
 *	real neighbor data at all) and having to regenerate a full throwaway copy of each neighbor just
 *	to read one boundary row/column from it.
 */
struct ChunkManager {
    TerrainParams baseParams;

    int viewRadius = 8;

    // How long a per-chunk Hurst/reseed blend takes, in the same time units as the currentTime
    // passed to updateLoadedChunks/drawing (glfwGetTime()'s seconds).
    double blendDuration = 2.0;

    // Final, GPU-uploaded chunks (heights/positions/indices + normals), each with its own
    // independent from/to blend state.
    std::unordered_map<ChunkCoord, LoadedChunk> loadedChunks;

    // CPU-side heights/positions/indices, retained for as long as a chunk exists (from the moment
    // phase 1 finishes until it's destroyed) — this is what lets a chunk still mid-phase-2, or
    // already loaded, cheaply hand a boundary row/column to a neighbor without regenerating anything.
    std::unordered_map<ChunkCoord, GeometryData> chunkData;

    // Phase 1 (heights/positions/indices) in flight.
    std::unordered_map<ChunkCoord, std::future<GeometryData>> pendingChunks;

    // Phase 1 done, phase 2 not yet dispatched — waiting on a still-generating neighbor that will
    // eventually exist (one outside the view radius never blocks dispatch; see updateLoadedChunks).
    std::unordered_set<ChunkCoord> readyForNormals;

    // Phase 2 (normal derivation) in flight.
    std::unordered_map<ChunkCoord, std::future<std::vector<glm::vec3>>> pendingNormals;
};

/*!
 *	Call once per frame (or throttled): computes the camera's current chunk, kicks off background
 *	phase-1 generation for any newly-needed chunk in the (2 * viewRadius + 1)^2 window around it,
 *	dispatches phase-2 normal derivation for any chunk whose own and all still-relevant neighbors'
 *	phase 1 has finished, uploads any chunk whose phase 2 has finished (starting a from/to blend if
 *	that coordinate was already loaded — see LoadedChunk), destroys the now-unused `from` of any
 *	chunk whose blend has finished, and destroys/frees resources for chunks that have fallen more
 *	than (viewRadius + 1) chunks away (the +1 gives a little hysteresis so a camera sitting near a
 *	boundary doesn't thrash chunks in and out every frame). If anything is actually destroyed on the
 *	GPU this call, waits for the GPU to go idle first — a chunk's buffers might still be referenced
 *	by a previous frame's in-flight command buffer, and destroying them out from under that would be
 *	a GPU-side use-after-free (manifests as VK_ERROR_DEVICE_LOST).
 *
 *	`currentTime` should be the same clock used when drawing (glfwGetTime()), so a chunk's blend
 *	progress is computed consistently between where it's started and where it's read for rendering.
 */
void updateLoadedChunks(VkDevice vk_device, ChunkManager& manager, const glm::vec3& cameraPos, double currentTime);

/*!
 *	Call after changing `manager.baseParams.hurst` or `.seed` — both affect every chunk's heights,
 *	not just position, so every currently-*loaded* chunk needs to be regenerated, not just chunks
 *	requested from this point on. Clears `chunkData` for every coordinate in `loadedChunks`, without
 *	touching their GPU resources — this makes the very next `updateLoadedChunks` call see those
 *	coordinates as missing and kick off a fresh phase-1/phase-2 generation for them, using the
 *	now-current `baseParams`, while the existing geometry keeps rendering normally in the meantime.
 *	Once that generation's upload completes, `updateLoadedChunks` recognizes the coordinate was
 *	already loaded and starts a blend from the old geometry to the new one instead of replacing it
 *	outright (see LoadedChunk).
 *
 *	Callers must check `!isRegenerating(manager)` first and not call this while a previous
 *	regeneration is still in flight or blending — `updateLoadedChunks` assumes a chunk's `from` is
 *	always empty at the moment a new one is about to be assigned, and does not guard against or
 *	clean up a still-in-progress one.
 */
void invalidateAllLoadedChunks(ChunkManager& manager);

/*!
 *	True while any chunk is still mid-generation (phase 1 or phase 2, from either initial streaming
 *	or a Hurst/reseed regeneration) or mid-blend (a non-empty `from`). Callers should refuse a new
 *	Hurst/reseed change while this is true, both to keep the control disabled/inert in the UI and
 *	because `invalidateAllLoadedChunks`/`updateLoadedChunks` require it (see their docs).
 */
bool isRegenerating(const ChunkManager& manager);

/*!
 *	Destroys the GPU resources for every currently-loaded chunk. Call once at shutdown.
 */
void cleanupChunkManager(ChunkManager& manager);
