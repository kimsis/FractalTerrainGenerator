#include "ChunkManager.h"

#include <VulkanLaunchpad.h>

#include <chrono>
#include <cstdlib>
#include <optional>
#include <utility>

std::future<GeometryData> startTerrainGeneration(const TerrainParams& params) {
    return std::async(std::launch::async, generateTerrainGeometry, params);
}

VkBuffer createAndUploadIndexBuffer(const std::vector<uint32_t>& indices) {
    if (indices.empty()) {
        VKL_EXIT_WITH_ERROR("An empty indices vector has been passed to createAndUploadIndexBuffer(...)");
    }

    size_t indices_buffer_byte_size = indices.size() * sizeof(indices[0]);
    return vklCreateHostCoherentBufferAndUploadData(
        indices.data(),
        static_cast<VkDeviceSize>(indices_buffer_byte_size),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );
}

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

// True if `geometry`'s buffers are real (not the all-VK_NULL_HANDLE state `from` sits in while not
// blending).
static bool isValidGeometry(const Geometry& geometry) { return geometry.vertexBuffer != VK_NULL_HANDLE; }

// The chunk's persistent vertex buffer that isn't currently `to` — its ping-pong partner. While
// blending, that's `from.vertexBuffer`; otherwise it's `idleVertexBuffer`. VK_NULL_HANDLE if the
// chunk has never regenerated.
static VkBuffer otherVertexBuffer(const LoadedChunk& chunk) {
    return isValidGeometry(chunk.from) ? chunk.from.vertexBuffer : chunk.idleVertexBuffer;
}

// Which of coord's 4 neighbors currently have no chunkData (see LoadedChunk::missingNeighborMask).
static uint8_t missingNeighborMaskFor(const ChunkManager& manager, const ChunkCoord& coord) {
    uint8_t mask = 0;
    if (!manager.chunkData.count(ChunkCoord{coord.cx - 1, coord.cy})) mask |= NEIGHBOR_LEFT;
    if (!manager.chunkData.count(ChunkCoord{coord.cx + 1, coord.cy})) mask |= NEIGHBOR_RIGHT;
    if (!manager.chunkData.count(ChunkCoord{coord.cx, coord.cy + 1})) mask |= NEIGHBOR_TOP;
    if (!manager.chunkData.count(ChunkCoord{coord.cx, coord.cy - 1})) mask |= NEIGHBOR_BOTTOM;
    return mask;
}

// Dispatches background normal derivation for `coord` using whatever of its 4 neighbors currently
// have chunkData available (the rest are extrapolated — see deriveTerrainNormals). Shared by the
// initial phase-2 dispatch and by the later re-derivation that patches a previously-extrapolated
// edge once real neighbor data arrives.
static std::future<std::vector<glm::vec3>> dispatchNormalDerivation(const ChunkManager& manager, const ChunkCoord& coord, int size) {
    ChunkCoord leftCoord{coord.cx - 1, coord.cy};
    ChunkCoord rightCoord{coord.cx + 1, coord.cy};
    ChunkCoord topCoord{coord.cx, coord.cy + 1};
    ChunkCoord bottomCoord{coord.cx, coord.cy - 1};

    std::optional<std::vector<float>> leftSkirt, rightSkirt, topSkirt, bottomSkirt;
    if (manager.chunkData.count(leftCoord)) leftSkirt = sampleColumn(manager.chunkData.at(leftCoord), size, size - 2);
    if (manager.chunkData.count(rightCoord)) rightSkirt = sampleColumn(manager.chunkData.at(rightCoord), size, 1);
    if (manager.chunkData.count(topCoord)) topSkirt = sampleRow(manager.chunkData.at(topCoord), size, 1);
    if (manager.chunkData.count(bottomCoord)) bottomSkirt = sampleRow(manager.chunkData.at(bottomCoord), size, size - 2);

    std::vector<glm::vec3> positions = manager.chunkData.at(coord).positions;
    int spacing = manager.baseParams.spacing;
    return std::async(std::launch::async, [positions = std::move(positions), size, spacing, leftSkirt, rightSkirt, topSkirt, bottomSkirt]() {
        return deriveTerrainNormals(
            positions,
            size,
            spacing,
            leftSkirt ? &*leftSkirt : nullptr,
            rightSkirt ? &*rightSkirt : nullptr,
            topSkirt ? &*topSkirt : nullptr,
            bottomSkirt ? &*bottomSkirt : nullptr
        );
    });
}

void updateLoadedChunks(ChunkManager& manager, const glm::vec3& cameraPos, double currentTime) {
    // How many updateLoadedChunks calls a PendingDestroy waits before it's actually freed.
    static constexpr int DESTROY_DEFERRAL_CALLS = 2;

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
    static const ChunkCoord OFFSETS[4] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}}; // left, right, top, bottom
    for (auto it = manager.readyForNormals.begin(); it != manager.readyForNormals.end();) {
        ChunkCoord coord = *it;

        bool blockedOnNeighbor = false;
        for (const ChunkCoord& offset : OFFSETS) {
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

        manager.pendingNormalsMask[coord] = missingNeighborMaskFor(manager, coord);
        manager.pendingNormals[coord] = dispatchNormalDerivation(manager, coord, size);
        it = manager.readyForNormals.erase(it);
    }

    // 4. Drain phase-2 results and upload to the GPU, if still in range. An already-loaded coordinate
    // is a regeneration: snap `to` to `from` and start a blend from `currentTime` (uncapped). A
    // brand-new chunk is capped at maxUploadsPerFrame instead; one left over stays queued in
    // `pendingNormals` (peeked via wait_for, not consumed) for a later call.
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
        uint8_t missingMask = manager.pendingNormalsMask.at(it->first);
        if (inRange) {
            GeometryData data = manager.chunkData[it->first];
            data.normals = std::move(normals);

            auto existing = manager.loadedChunks.find(it->first);
            if (existing == manager.loadedChunks.end()) {
                Geometry newGeometry = createAndUploadIntoGpuMemory(data);
                // Created once, ever, from whichever chunk happens to load first — every chunk's
                // indices are identical (see createAndUploadIndexBuffer).
                if (manager.sharedIndicesBuffer == VK_NULL_HANDLE) {
                    manager.sharedIndicesBuffer = createAndUploadIndexBuffer(data.indices);
                    manager.sharedNumberOfIndices = static_cast<uint32_t>(data.indices.size());
                }
                manager.loadedChunks[it->first] = LoadedChunk{Geometry{}, newGeometry, VK_NULL_HANDLE, 0.0, missingMask};
                uploaded_this_frame++;
            } else {
                // Regeneration: the shared index buffer isn't touched (topology never changes).
                // Positions/normals reuse whichever of the chunk's two persistent buffers isn't
                // currently `to` — only the first-ever regeneration allocates, to create that
                // second buffer.
                Geometry newGeometry;
                if (existing->second.idleVertexBuffer == VK_NULL_HANDLE) {
                    newGeometry = createAndUploadIntoGpuMemory(data);
                } else {
                    newGeometry.vertexBuffer = existing->second.idleVertexBuffer;
                    newGeometry.normalsOffset = uploadVertexDataInPlace(newGeometry.vertexBuffer, data);
                }
                existing->second.idleVertexBuffer = existing->second.to.vertexBuffer;
                existing->second.from = existing->second.to;
                existing->second.to = newGeometry;
                existing->second.blendStartTime = currentTime;
                existing->second.missingNeighborMask = missingMask;
            }
        }
        manager.pendingNormalsMask.erase(it->first);
        it = manager.pendingNormals.erase(it);
    }

    // 5. Re-derive normals for loaded chunks with a nonzero missingNeighborMask once a previously-
    // missing neighbor has chunkData. Uncapped and excluded from isRegenerating() — a background
    // refinement of stable geometry, not a shape change.
    for (auto& entry : manager.loadedChunks) {
        const ChunkCoord& coord = entry.first;
        LoadedChunk& chunk = entry.second;
        if (chunk.missingNeighborMask == 0) continue;
        if (manager.pendingRenormals.count(coord)) continue;

        uint8_t stillMissing = missingNeighborMaskFor(manager, coord);
        if ((chunk.missingNeighborMask & ~stillMissing) == 0) continue; // nothing newly available yet

        manager.pendingRenormals[coord] = dispatchNormalDerivation(manager, coord, size);
    }

    // 6. Drain re-derived normals from step 5 and patch `to`'s normals buffer in place. Discarded
    // silently if the chunk was invalidated/evicted while in flight.
    for (auto it = manager.pendingRenormals.begin(); it != manager.pendingRenormals.end();) {
        if (it->second.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            ++it;
            continue;
        }
        std::vector<glm::vec3> normals = it->second.get();
        auto loaded = manager.loadedChunks.find(it->first);
        if (loaded != manager.loadedChunks.end() && manager.chunkData.count(it->first)) {
            updateGeometryNormals(loaded->second.to, normals);
            loaded->second.missingNeighborMask = missingNeighborMaskFor(manager, it->first);
        }
        it = manager.pendingRenormals.erase(it);
    }

    // 7-8. Evict chunks past (viewRadius + 1), queuing their Geometry in pendingDestroys rather than
    // destroying outright. A finished blend destroys nothing: `from`'s buffer is the chunk's
    // ping-pong spare, reused by the next regeneration.
    for (auto it = manager.loadedChunks.begin(); it != manager.loadedChunks.end();) {
        LoadedChunk& chunk = it->second;
        if (!isWithinViewRadius(it->first, center, destroyRadius)) {
            // `to`'s vertex buffer and the chunk's other persistent buffer (if any) are freed here;
            // the index buffer is shared across all chunks, freed only once, at cleanupChunkManager.
            VkBuffer other = otherVertexBuffer(chunk);
            if (other != VK_NULL_HANDLE) manager.pendingDestroys.push_back({Geometry{other, 0}, DESTROY_DEFERRAL_CALLS});
            manager.pendingDestroys.push_back({chunk.to, DESTROY_DEFERRAL_CALLS});
            it = manager.loadedChunks.erase(it);
            continue;
        }
        if (isValidGeometry(chunk.from) && currentTime - chunk.blendStartTime >= manager.blendDuration) {
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

    // Free up to maxDestroysPerFrame pendingDestroys entries whose deferral has elapsed — the only
    // place real GPU destroy calls happen.
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
        VkBuffer other = otherVertexBuffer(entry.second);
        if (other != VK_NULL_HANDLE) destroyGeometryGpuMemory(Geometry{other, 0});
        destroyGeometryGpuMemory(entry.second.to);
    }
    manager.loadedChunks.clear();
    // The shared index buffer is freed once here, at shutdown.
    if (manager.sharedIndicesBuffer != VK_NULL_HANDLE) {
        destroyGeometryGpuMemory(Geometry{manager.sharedIndicesBuffer, 0});
        manager.sharedIndicesBuffer = VK_NULL_HANDLE;
    }
    for (auto& pending : manager.pendingDestroys) {
        destroyGeometryGpuMemory(pending.geometry);
    }
    manager.pendingDestroys.clear();
}
