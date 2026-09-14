#include "ChunkManager.h"

#include <chrono>
#include <cstdlib>
#include <optional>

std::future<GeometryData> startTerrainGeneration(const TerrainParams& params);

// True if `coord` falls inside the square window that's actually kept loaded around `center` — a
// neighbor outside this window will never be generated at the current view radius, so an edge
// bordering one falls back to extrapolation instead of blocking phase 2 forever.
static bool isWithinViewRadius(const ChunkCoord& coord, const ChunkCoord& center, int viewRadius) {
    return std::abs(coord.cx - center.cx) <= viewRadius && std::abs(coord.cy - center.cy) <= viewRadius;
}

static std::vector<float> sampleColumn(const GeometryData& data, int size, int localX) {
    std::vector<float> column(size);
    for (int y = 0; y < size; y++) column[y] = data.positions[localX * size + y].z;
    return column;
}

static std::vector<float> sampleRow(const GeometryData& data, int size, int localY) {
    std::vector<float> row(size);
    for (int x = 0; x < size; x++) row[x] = data.positions[x * size + localY].z;
    return row;
}

// True if `geometry`'s buffers are all real (as opposed to the all-VK_NULL_HANDLE state a
// LoadedChunk's `from` sits in whenever that chunk isn't currently blending).
static bool isValidGeometry(const Geometry& geometry) {
    return geometry.positionsBuffer != VK_NULL_HANDLE;
}

void updateLoadedChunks(VkDevice vk_device, ChunkManager& manager, const glm::vec3& cameraPos, double currentTime) {
    ChunkCoord center = cameraToChunkCoord(cameraPos, manager.baseParams);
    int destroyRadius = manager.viewRadius + 1;
    int size = (1 << manager.baseParams.gridSizeExponent) + 1;

    // 1. Kick off phase-1 generation (heights/positions/indices, no normals) for anything in the
    // desired window that isn't already loaded, in flight, or past phase 1.
    for (int dx = -manager.viewRadius; dx <= manager.viewRadius; dx++) {
        for (int dy = -manager.viewRadius; dy <= manager.viewRadius; dy++) {
            ChunkCoord coord{center.cx + dx, center.cy + dy};
            if (manager.chunkData.count(coord) || manager.pendingChunks.count(coord)) continue;

            TerrainParams chunkParams = manager.baseParams;
            chunkParams.chunkX = coord.cx;
            chunkParams.chunkY = coord.cy;
            manager.pendingChunks[coord] = startTerrainGeneration(chunkParams);
        }
    }

    // 2. Drain phase-1 results: keep the CPU data (for skirt lookups and eventual upload) only if
    // still within range of the *current* center, otherwise discard it (the camera moved on before
    // it finished generating).
    for (auto it = manager.pendingChunks.begin(); it != manager.pendingChunks.end();) {
        if (it->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            GeometryData data = it->second.get();
            if (isWithinViewRadius(it->first, center, destroyRadius)) {
                manager.chunkData[it->first] = std::move(data);
                manager.readyForNormals.insert(it->first);
            }
            it = manager.pendingChunks.erase(it);
        } else {
            ++it;
        }
    }

    // 3. Dispatch phase-2 (normal derivation) for any chunk whose own phase 1 is done and whose
    // every still-relevant neighbor (i.e. one that will actually be generated at the current view
    // radius) has *also* finished phase 1. Each needed boundary row/column is copied off the main
    // thread here (cheap — O(size), not a full regeneration) and handed to the background task, so
    // the task doesn't touch manager state that might be destroyed while it's running. A neighbor
    // that's outside the view radius can never resolve that edge with real data, so it's left as
    // nullptr and deriveTerrainNormals() falls back to extrapolating it instead of waiting forever.
    static const ChunkCoord kOffsets[4] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};  // left, right, top, bottom
    for (auto it = manager.readyForNormals.begin(); it != manager.readyForNormals.end();) {
        ChunkCoord coord = *it;
        ChunkCoord leftCoord{coord.cx - 1, coord.cy};
        ChunkCoord rightCoord{coord.cx + 1, coord.cy};
        ChunkCoord topCoord{coord.cx, coord.cy + 1};
        ChunkCoord bottomCoord{coord.cx, coord.cy - 1};

        bool blockedOnNeighbor = false;
        for (const ChunkCoord& offset : kOffsets) {
            ChunkCoord neighbor{coord.cx + offset.cx, coord.cy + offset.cy};
            bool neighborWillExist = isWithinViewRadius(neighbor, center, manager.viewRadius);
            if (neighborWillExist && !manager.chunkData.count(neighbor)) {
                blockedOnNeighbor = true;
                break;
            }
        }
        if (blockedOnNeighbor) {
            ++it;
            continue;
        }

        std::optional<std::vector<float>> leftSkirt, rightSkirt, topSkirt, bottomSkirt;
        if (manager.chunkData.count(leftCoord)) leftSkirt = sampleColumn(manager.chunkData[leftCoord], size, size - 2);
        if (manager.chunkData.count(rightCoord)) rightSkirt = sampleColumn(manager.chunkData[rightCoord], size, 1);
        if (manager.chunkData.count(topCoord)) topSkirt = sampleRow(manager.chunkData[topCoord], size, 1);
        if (manager.chunkData.count(bottomCoord)) bottomSkirt = sampleRow(manager.chunkData[bottomCoord], size, size - 2);

        std::vector<glm::vec3> positions = manager.chunkData[coord].positions;
        int spacing = manager.baseParams.spacing;
        manager.pendingNormals[coord] = std::async(
            std::launch::async,
            [positions = std::move(positions), size, spacing, leftSkirt, rightSkirt, topSkirt, bottomSkirt]() {
                return deriveTerrainNormals(
                    positions,
                    size,
                    spacing,
                    leftSkirt ? &*leftSkirt : nullptr,
                    rightSkirt ? &*rightSkirt : nullptr,
                    topSkirt ? &*topSkirt : nullptr,
                    bottomSkirt ? &*bottomSkirt : nullptr
                );
            }
        );
        it = manager.readyForNormals.erase(it);
    }

    // Shared across steps 4-6 below: batches every GPU destroy this call into at most one
    // vkDeviceWaitIdle, no matter how many separate reasons (blend completion, eviction) trigger one.
    bool destroyed_any = false;

    // 4. Drain phase-2 results: combine the derived normals with the retained positions/indices and
    // upload to the GPU, if still in range. If this coordinate was already loaded, this is a
    // regeneration (Hurst/seed changed — see invalidateAllLoadedChunks): snap its current `to` to be
    // the new `from` and start a fresh blend toward the just-uploaded geometry, rather than
    // replacing it outright. Callers are required to keep Hurst/reseed changes disabled while
    // isRegenerating() is true, so `from` is guaranteed empty here — nothing mid-blend to supersede.
    for (auto it = manager.pendingNormals.begin(); it != manager.pendingNormals.end();) {
        if (it->second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            std::vector<glm::vec3> normals = it->second.get();
            if (isWithinViewRadius(it->first, center, destroyRadius) && manager.chunkData.count(it->first)) {
                GeometryData data = manager.chunkData[it->first];
                data.normals = std::move(normals);
                Geometry newGeometry = createAndUploadIntoGpuMemory(data);

                auto existing = manager.loadedChunks.find(it->first);
                if (existing == manager.loadedChunks.end()) {
                    manager.loadedChunks[it->first] = LoadedChunk{Geometry{}, newGeometry, 0.0};
                } else {
                    existing->second.from = existing->second.to;
                    existing->second.to = newGeometry;
                    existing->second.blendStartTime = currentTime;
                }
            }
            it = manager.pendingNormals.erase(it);
        } else {
            ++it;
        }
    }

    // 5. Destroy the now-unused `from` of any chunk whose blend has finished.
    for (auto& entry : manager.loadedChunks) {
        LoadedChunk& chunk = entry.second;
        if (isValidGeometry(chunk.from) && currentTime - chunk.blendStartTime >= manager.blendDuration) {
            if (!destroyed_any) {
                vkDeviceWaitIdle(vk_device);
                destroyed_any = true;
            }
            destroyGeometryGpuMemory(chunk.from);
            chunk.from = Geometry{};
        }
    }

    // 6. Destroy anything that's fallen more than (viewRadius + 1) chunks away (Chebyshev distance,
    // matching the square load window). GPU resources are freed with a synchronization guard (see
    // header); the retained CPU data has no such requirement.
    for (auto it = manager.loadedChunks.begin(); it != manager.loadedChunks.end();) {
        if (!isWithinViewRadius(it->first, center, destroyRadius)) {
            if (!destroyed_any) {
                vkDeviceWaitIdle(vk_device);
                destroyed_any = true;
            }
            if (isValidGeometry(it->second.from)) destroyGeometryGpuMemory(it->second.from);
            destroyGeometryGpuMemory(it->second.to);
            it = manager.loadedChunks.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = manager.chunkData.begin(); it != manager.chunkData.end();) {
        if (!isWithinViewRadius(it->first, center, destroyRadius)) {
            manager.readyForNormals.erase(it->first);
            it = manager.chunkData.erase(it);
        } else {
            ++it;
        }
    }
}

void invalidateAllLoadedChunks(ChunkManager& manager) {
    for (auto& entry : manager.loadedChunks) {
        manager.chunkData.erase(entry.first);
    }
}

bool isRegenerating(const ChunkManager& manager) {
    if (!manager.pendingChunks.empty() || !manager.readyForNormals.empty() || !manager.pendingNormals.empty()) return true;
    for (auto& entry : manager.loadedChunks) {
        if (isValidGeometry(entry.second.from)) return true;
    }
    return false;
}

void cleanupChunkManager(ChunkManager& manager) {
    for (auto& entry : manager.loadedChunks) {
        if (isValidGeometry(entry.second.from)) destroyGeometryGpuMemory(entry.second.from);
        destroyGeometryGpuMemory(entry.second.to);
    }
    manager.loadedChunks.clear();
}
