#include "ChunkManager.h"

#include <chrono>
#include <cstdlib>
#include <optional>

std::future<GeometryData> startTerrainGeneration(const TerrainParams& params);

// True if `coord` falls inside the square window kept loaded around `center`.
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

// True if `geometry`'s buffers are real, not the all-VK_NULL_HANDLE state a LoadedChunk's `from`
// sits in whenever that chunk isn't currently blending.
static bool isValidGeometry(const Geometry& geometry) {
    return geometry.positionsBuffer != VK_NULL_HANDLE;
}

// How many updateLoadedChunks calls a PendingDestroy waits before it's actually freed. One call
// would already be provably safe (see PendingDestroy's doc), but two gives a small margin in case
// VulkanLaunchpad.cpp's CONCURRENT_FRAMES ever changes from its current value of 1.
static constexpr int kDestroyDeferralCalls = 2;

void updateLoadedChunks(ChunkManager& manager, const glm::vec3& cameraPos, double currentTime) {
    ChunkCoord center = cameraToChunkCoord(cameraPos, manager.baseParams);
    int destroyRadius = manager.viewRadius + 1;
    int size = (1 << manager.baseParams.gridSizeExponent) + 1;

    // 1. Kick off phase-1 generation for anything in the desired window not already loaded or in flight.
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

    // 2. Drain phase-1 results, keeping the CPU data only if still within range of the current center.
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

    // 3. Dispatch phase-2 (normal derivation) for any chunk whose own phase 1 is done and whose every
    // still-relevant neighbor has also finished phase 1. A neighbor outside the view radius will never
    // exist, so it's left as nullptr and deriveTerrainNormals() extrapolates that edge instead.
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

    // 4. Drain phase-2 results and upload to the GPU, if still in range. If this coordinate was
    // already loaded, it's a regeneration: snap its current `to` to be the new `from` and start a
    // fresh blend from `currentTime` — regeneration uploads are uncapped (see maxUploadsPerFrame).
    // A brand-new chunk (never loaded before) is capped at maxUploadsPerFrame per call instead, since
    // travelling can bring a whole ring of new chunks into range at once; a ready one past the cap is
    // left queued in `pendingNormals` (peeked via wait_for, not consumed) for a later call.
    int uploaded_this_frame = 0;
    for (auto it = manager.pendingNormals.begin(); it != manager.pendingNormals.end();) {
        if (it->second.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            ++it;
            continue;
        }
        bool inRange = isWithinViewRadius(it->first, center, destroyRadius) && manager.chunkData.count(it->first);
        bool isNewChunk = inRange && !manager.loadedChunks.count(it->first);
        if (isNewChunk && uploaded_this_frame >= manager.maxUploadsPerFrame) {
            ++it;
            continue;
        }
        std::vector<glm::vec3> normals = it->second.get();
        if (inRange) {
            GeometryData data = manager.chunkData[it->first];
            data.normals = std::move(normals);
            Geometry newGeometry = createAndUploadIntoGpuMemory(data);

            auto existing = manager.loadedChunks.find(it->first);
            if (existing == manager.loadedChunks.end()) {
                manager.loadedChunks[it->first] = LoadedChunk{Geometry{}, newGeometry, 0.0};
                uploaded_this_frame++;
            } else {
                existing->second.from = existing->second.to;
                existing->second.to = newGeometry;
                existing->second.blendStartTime = currentTime;
            }
        }
        it = manager.pendingNormals.erase(it);
    }

    // 5-6. One pass over loadedChunks: evict anything that's fallen more than (viewRadius + 1)
    // chunks away, else clear the now-unused `from` of any chunk whose blend has finished. Both cases
    // just queue the freed Geometry in pendingDestroys (see PendingDestroy) rather than destroy it
    // outright, so this pass is cheap bookkeeping regardless of how many chunks a single chunk-border
    // crossing or mass regeneration affects at once — the real GPU cost is paid gradually below.
    for (auto it = manager.loadedChunks.begin(); it != manager.loadedChunks.end();) {
        LoadedChunk& chunk = it->second;
        if (!isWithinViewRadius(it->first, center, destroyRadius)) {
            if (isValidGeometry(chunk.from)) manager.pendingDestroys.push_back({chunk.from, kDestroyDeferralCalls});
            manager.pendingDestroys.push_back({chunk.to, kDestroyDeferralCalls});
            it = manager.loadedChunks.erase(it);
            continue;
        }
        if (isValidGeometry(chunk.from) && currentTime - chunk.blendStartTime >= manager.blendDuration) {
            manager.pendingDestroys.push_back({chunk.from, kDestroyDeferralCalls});
            chunk.from = Geometry{};
        }
        ++it;
    }
    for (auto it = manager.chunkData.begin(); it != manager.chunkData.end();) {
        if (!isWithinViewRadius(it->first, center, destroyRadius)) {
            manager.readyForNormals.erase(it->first);
            it = manager.chunkData.erase(it);
        } else {
            ++it;
        }
    }

    // Actually free up to maxDestroysPerFrame pendingDestroys entries whose deferral has elapsed —
    // the only place real GPU destroy calls happen, so this is what caps how much of that cost lands
    // in any one call, regardless of how many chunks became eligible for destruction above.
    int destroyed_this_frame = 0;
    for (auto it = manager.pendingDestroys.begin(); it != manager.pendingDestroys.end();) {
        it->callsRemaining--;
        if (it->callsRemaining > 0 || destroyed_this_frame >= manager.maxDestroysPerFrame) {
            ++it;
            continue;
        }
        destroyGeometryGpuMemory(it->geometry);
        destroyed_this_frame++;
        it = manager.pendingDestroys.erase(it);
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
    for (auto& pending : manager.pendingDestroys) {
        destroyGeometryGpuMemory(pending.geometry);
    }
    manager.pendingDestroys.clear();
}
