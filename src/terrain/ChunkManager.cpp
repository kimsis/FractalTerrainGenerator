#include "ChunkManager.h"

#include <chrono>
#include <cstdlib>

std::future<GeometryData> startTerrainGeneration(const TerrainParams& params);

void updateLoadedChunks(ChunkManager& manager, const glm::vec3& cameraPos) {
    ChunkCoord center = cameraToChunkCoord(cameraPos, manager.baseParams);
    int evictRadius = manager.viewRadius + 1;

    // Kick off generation for anything in the desired window that isn't loaded or already
    // pending. No cap on concurrent std::async calls here: each one is cheap now that
    // generateTerrainGeometry() actually generates at the requested (small) chunk size instead of
    // wastefully building a full default-sized grid first (see Geometry.cpp) — that was the real
    // cause of an earlier OOM crash, not thread count.
    for (int dx = -manager.viewRadius; dx <= manager.viewRadius; dx++) {
        for (int dy = -manager.viewRadius; dy <= manager.viewRadius; dy++) {
            ChunkCoord coord{center.cx + dx, center.cy + dy};
            if (manager.loadedChunks.count(coord) || manager.pendingChunks.count(coord)) continue;

            TerrainParams chunkParams = manager.baseParams;
            chunkParams.chunkX = coord.cx;
            chunkParams.chunkY = coord.cy;
            manager.pendingChunks[coord] = startTerrainGeneration(chunkParams);
        }
    }

    // Resolve any chunk whose background generation has completed: upload it only if it's still
    // within range of the *current* center, otherwise discard it (the camera moved on before it
    // finished generating). Either way it's removed from pendingChunks here — this is what
    // actually bounds pendingChunks; without it, a fast-moving camera queues chunks around several
    // different centers before any of them finish, and nothing ever prunes the ones that are no
    // longer wanted.
    for (auto it = manager.pendingChunks.begin(); it != manager.pendingChunks.end();) {
        if (it->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            GeometryData data = it->second.get();
            int distX = std::abs(it->first.cx - center.cx);
            int distY = std::abs(it->first.cy - center.cy);
            if (distX <= evictRadius && distY <= evictRadius) {
                manager.loadedChunks[it->first] = createAndUploadIntoGpuMemory(data);
            }
            it = manager.pendingChunks.erase(it);
        } else {
            ++it;
        }
    }

    // Evict anything that's fallen more than (viewRadius + 1) chunks away (Chebyshev distance,
    // matching the square load window), freeing its GPU resources.
    for (auto it = manager.loadedChunks.begin(); it != manager.loadedChunks.end();) {
        int distX = std::abs(it->first.cx - center.cx);
        int distY = std::abs(it->first.cy - center.cy);
        if (distX > evictRadius || distY > evictRadius) {
            destroyGeometryGpuMemory(it->second);
            it = manager.loadedChunks.erase(it);
        } else {
            ++it;
        }
    }
}

void cleanupChunkManager(ChunkManager& manager) {
    for (auto& entry : manager.loadedChunks) {
        destroyGeometryGpuMemory(entry.second);
    }
    manager.loadedChunks.clear();
}
