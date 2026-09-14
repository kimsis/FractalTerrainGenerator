/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */

#include "TerrainGeometry.h"

#include <glm/gtc/constants.hpp>
#include <VulkanLaunchpad.h>

#include "DiamondSquareGenerator.h"

// Phase 1 of terrain generation: heights + positions + indices only, no normals. This is the part
// that's fully independent between chunks — a pure function of (seed, global coords, level) — so
// it can run on a background thread with zero cross-chunk communication. Normals are derived in a
// separate, dependency-aware phase 2 once neighboring chunks' phase 1 data is available (see
// deriveTerrainNormals() and ChunkManager's updateLoadedChunks()), instead of regenerating full
// throwaway neighbor grids just to read a boundary row/column from them.
GeometryData generateTerrainGeometry(const TerrainParams& params) {
    GeometryData data;
    DiamondSquareGenerator generator(params);
    data.positions = generator.getPositions();
    data.indices = generator.getIndices();
    return data;
}

Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data, bool upload_indices) {
    if (geometry_data.positions.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::positions vector has been passed to createAndUploadIntoGpuMemory(...)");
    }
    if (upload_indices && geometry_data.indices.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::indices vector has been passed to createAndUploadIntoGpuMemory(...)");
    }

    Geometry result;

    // Create vertex positions from buffer and copy data into it:
    size_t positions_buffer_byte_size = geometry_data.positions.size() * sizeof(geometry_data.positions[0]);
    result.positionsBuffer = vklCreateHostCoherentBufferAndUploadData(
        static_cast<const void*>(geometry_data.positions.data()),
        positions_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    // Create vertex normals from buffer and copy data into it:
    size_t normals_buffer_byte_size = geometry_data.normals.size() * sizeof(geometry_data.normals[0]);
    result.normalsBuffer = vklCreateHostCoherentBufferAndUploadData(
        geometry_data.normals.data(),
        normals_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    if (upload_indices) {
        // Create indices buffer and copy data into it:
        size_t indices_buffer_byte_size = geometry_data.indices.size() * sizeof(geometry_data.indices[0]);
        result.indicesBuffer = vklCreateHostCoherentBufferAndUploadData(
            geometry_data.indices.data(),
            static_cast<VkDeviceSize>(indices_buffer_byte_size),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
        );
        // Also store the number of indices:
        result.numberOfIndices = static_cast<uint32_t>(geometry_data.indices.size());
    } else {
        // Caller already owns a still-valid index buffer for this chunk (topology never changes
        // across a Hurst/reseed regeneration) and will fill these fields in themselves.
        result.indicesBuffer = VK_NULL_HANDLE;
        result.numberOfIndices = 0;
    }

    return result;
}

void destroyGeometryGpuMemory(const Geometry& geometry) {
    // A VK_NULL_HANDLE field means this Geometry doesn't own that buffer (see createAndUploadIntoGpuMemory's
    // upload_indices) — vklDestroyHostCoherentBufferAndItsBackingMemory itself errors out on a null handle,
    // so each field must be skipped explicitly rather than passed through unconditionally.
    if (geometry.indicesBuffer != VK_NULL_HANDLE) {
        vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.indicesBuffer);
    }
    if (geometry.positionsBuffer != VK_NULL_HANDLE) {
        vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.positionsBuffer);
    }
    if (geometry.normalsBuffer != VK_NULL_HANDLE) {
        vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.normalsBuffer);
    }
}

void updateGeometryNormals(const Geometry& geometry, const std::vector<glm::vec3>& normals) {
    vklCopyDataIntoHostCoherentBuffer(geometry.normalsBuffer, normals.data(), normals.size() * sizeof(normals[0]));
}
