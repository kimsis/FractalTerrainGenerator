/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */

#include "Geometry.h"

#include <glm/gtc/constants.hpp>

#include "../utils/Utils.h"
#include "DiamondSquareGenerator.h"

GeometryData generateTerrainGeometry(const TerrainParams& params) {
    GeometryData data;
    DiamondSquareGenerator generator = DiamondSquareGenerator();
    generator.SetParams(params);

    data.positions = generator.getPositions();

    data.normals = generator.getNormals();

    data.indices = generator.getIndices();

    return data;
}

Geometry createAndUploadIntoGpuMemory(const GeometryData& old_geometry_data, const GeometryData& new_geometry_data) {
    if (old_geometry_data.positions.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::positions vector has been passed to createAndUploadIntoGpuMemory(...)");
    }
    if (old_geometry_data.indices.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::indices vector has been passed to createAndUploadIntoGpuMemory(...)");
    }

    Geometry result;

    // Create vertex positions from buffer and copy data into it:
    size_t positions_buffer_byte_size = old_geometry_data.positions.size() * sizeof(old_geometry_data.positions[0]);
    result.positionsBuffer = vklCreateHostCoherentBufferAndUploadData(
        static_cast<const void*>(old_geometry_data.positions.data()),
        positions_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    // Create vertex normals from buffer and copy data into it:
    size_t normals_buffer_byte_size = old_geometry_data.normals.size() * sizeof(old_geometry_data.normals[0]);
    result.normalsBuffer = vklCreateHostCoherentBufferAndUploadData(
        old_geometry_data.normals.data(),
        normals_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    // Create indices buffer and copy data into it:
    size_t indices_buffer_byte_size = new_geometry_data.indices.size() * sizeof(new_geometry_data.indices[0]);
    result.indicesBuffer = vklCreateHostCoherentBufferAndUploadData(
        new_geometry_data.indices.data(),
        static_cast<VkDeviceSize>(indices_buffer_byte_size),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );
    // Also store the number of indices:
    result.numberOfIndices = static_cast<uint32_t>(old_geometry_data.indices.size());

    return result;
}

void destroyGeometryGpuMemory(const Geometry& geometry) {
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.indicesBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.positionsBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.normalsBuffer);
}
