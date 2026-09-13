#pragma once

#include <future>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>

#include "ChunkCoord.h"
#include "DiamondSquareGenerator.h"
#include "Geometry.h"

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

    // Final, GPU-uploaded chunks (heights/positions/indices + normals).
    std::unordered_map<ChunkCoord, Geometry> loadedChunks;

    // CPU-side heights/positions/indices, retained for as long as a chunk exists (from the moment
    // phase 1 finishes until eviction) — this is what lets a chunk still mid-phase-2, or already
    // loaded, cheaply hand a boundary row/column to a neighbor without regenerating anything.
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
 *	phase 1 has finished, uploads any chunk whose phase 2 has finished, and evicts/frees resources
 *	for chunks that have fallen more than (viewRadius + 1) chunks away (the +1 gives a little
 *	hysteresis so a camera sitting near a boundary doesn't thrash chunks in and out every frame). If
 *	anything is actually evicted from the GPU this call, waits for the GPU to go idle first — a
 *	chunk's buffers might still be referenced by a previous frame's in-flight command buffer, and
 *	destroying them out from under that would be a GPU-side use-after-free (manifests as
 *	VK_ERROR_DEVICE_LOST).
 */
void updateLoadedChunks(VkDevice vk_device, ChunkManager& manager, const glm::vec3& cameraPos);

/*!
 *	Destroys the GPU resources for every currently-loaded chunk. Call once at shutdown.
 */
void cleanupChunkManager(ChunkManager& manager);
