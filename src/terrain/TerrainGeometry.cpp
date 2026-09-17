/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */

#include "TerrainGeometry.h"

#include <VulkanLaunchpad.h>

#include <cmath>

#include "../algorithms/DiamondSquareGenerator.h"

// Positions and normals share one buffer (see Geometry::vertexBuffer); normals start at this
// byte-aligned offset past the positions region.
static constexpr VkDeviceSize kVertexSubBufferAlignment = 16;

static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) { return (value + alignment - 1) / alignment * alignment; }

// Phase 1 of terrain generation: heights + positions + indices only, no normals. Normals are
// derived in a separate phase 2 (see deriveTerrainNormals) once neighboring chunks' phase-1 data
// is available.
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
    // null vertexBuffer must be skipped explicitly.
    if (geometry.vertexBuffer != VK_NULL_HANDLE) {
        vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.vertexBuffer);
    }
}

void updateGeometryNormals(const Geometry& geometry, const std::vector<glm::vec3>& normals) {
    vklCopyDataIntoHostCoherentBuffer(geometry.vertexBuffer, geometry.normalsOffset, normals.data(), normals.size() * sizeof(normals[0]));
}

std::optional<Hit> raycastTerrain(const glm::vec3& origin, const glm::vec3& direction) {
    constexpr float kGroundPlaneZ = 0.0f;
    if (std::abs(direction.z) < 1e-6f) return std::nullopt;
    float t = (kGroundPlaneZ - origin.z) / direction.z;
    if (t < 0.0f) return std::nullopt;
    glm::vec3 point = origin + t * direction;
    return Hit{point, t};
}
