#pragma once

#include <future>
#include <glm/glm.hpp>
#include <unordered_map>

#include "ChunkCoord.h"
#include "DiamondSquareGenerator.h"
#include "Geometry.h"

/*!
 *	Owns the set of terrain chunks currently loaded around the camera, generating new ones on
 *	demand and discarding distant ones, so memory/generation cost stays bounded by viewRadius
 *	regardless of how far the camera has traveled.
 */
struct ChunkManager {
    TerrainParams baseParams;

    int viewRadius = 8;

    std::unordered_map<ChunkCoord, Geometry> loadedChunks;
    std::unordered_map<ChunkCoord, std::future<GeometryData>> pendingChunks;
};

/*!
 *	Call once per frame (or throttled): computes the camera's current chunk, kicks off background
 *	generation for any newly-needed chunk in the (2 * viewRadius + 1)^2 window around it, uploads
 *	any chunk whose generation has finished, and evicts/frees GPU resources for chunks that have
 *	fallen more than (viewRadius + 1) chunks away (the +1 gives a little hysteresis so a camera
 *	sitting near a boundary doesn't thrash chunks in and out every frame).
 */
void updateLoadedChunks(ChunkManager& manager, const glm::vec3& cameraPos);

/*!
 *	Destroys the GPU resources for every currently-loaded chunk. Call once at shutdown.
 */
void cleanupChunkManager(ChunkManager& manager);
