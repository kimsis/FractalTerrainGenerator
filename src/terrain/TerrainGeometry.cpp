/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */

#include "TerrainGeometry.h"

#include <VulkanLaunchpad.h>

#include "DiamondSquareGenerator.h"

// Positions and normals share one buffer (see Geometry::vertexBuffer); normals start at this
// byte-aligned offset past the positions region. 16 bytes comfortably covers any reasonable
// vertex-attribute format without relying on a driver-specific minimum vertex-buffer-offset
// alignment (the core Vulkan spec places no explicit alignment requirement on vkCmdBindVertexBuffers
// offsets, unlike e.g. uniform/storage buffers).
static constexpr VkDeviceSize kVertexSubBufferAlignment = 16;

static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) { return (value + alignment - 1) / alignment * alignment; }

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

VkDeviceSize uploadVertexDataInPlace(VkBuffer vertexBuffer, const GeometryData& geometry_data) {
    size_t positions_buffer_byte_size = geometry_data.positions.size() * sizeof(geometry_data.positions[0]);
    size_t normals_buffer_byte_size = geometry_data.normals.size() * sizeof(geometry_data.normals[0]);
    VkDeviceSize normals_offset = alignUp(static_cast<VkDeviceSize>(positions_buffer_byte_size), kVertexSubBufferAlignment);

    vklCopyDataIntoHostCoherentBuffer(vertexBuffer, 0, geometry_data.positions.data(), positions_buffer_byte_size);
    vklCopyDataIntoHostCoherentBuffer(vertexBuffer, normals_offset, geometry_data.normals.data(), normals_buffer_byte_size);

    return normals_offset;
}

Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data) {
    if (geometry_data.positions.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::positions vector has been passed to createAndUploadIntoGpuMemory(...)");
    }

    Geometry result;

    // Positions and normals combined into one buffer/allocation — positions at offset 0, normals
    // right after (aligned).
    size_t positions_buffer_byte_size = geometry_data.positions.size() * sizeof(geometry_data.positions[0]);
    size_t normals_buffer_byte_size = geometry_data.normals.size() * sizeof(geometry_data.normals[0]);
    VkDeviceSize normals_offset = alignUp(static_cast<VkDeviceSize>(positions_buffer_byte_size), kVertexSubBufferAlignment);
    VkDeviceSize vertex_buffer_byte_size = normals_offset + static_cast<VkDeviceSize>(normals_buffer_byte_size);

    result.vertexBuffer =
        vklCreateHostCoherentBufferWithBackingMemory(vertex_buffer_byte_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    result.normalsOffset = uploadVertexDataInPlace(result.vertexBuffer, geometry_data);

    return result;
}

void destroyGeometryGpuMemory(const Geometry& geometry) {
    // vklDestroyHostCoherentBufferAndItsBackingMemory errors out on a null handle, so a legitimately
    // null vertexBuffer (see ChunkManager::LoadedChunk::idleVertexBuffer) must be skipped explicitly.
    if (geometry.vertexBuffer != VK_NULL_HANDLE) {
        vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.vertexBuffer);
    }
}

void updateGeometryNormals(const Geometry& geometry, const std::vector<glm::vec3>& normals) {
    vklCopyDataIntoHostCoherentBuffer(geometry.vertexBuffer, geometry.normalsOffset, normals.data(), normals.size() * sizeof(normals[0]));
}
