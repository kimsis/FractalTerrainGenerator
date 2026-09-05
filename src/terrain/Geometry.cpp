/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */

#include "Geometry.h"

#include <glm/gtc/constants.hpp>

#include "Utils.h"

#undef min
#undef max

constexpr float CORNELL_LEFT_R = 0.49f;
constexpr float CORNELL_LEFT_G = 0.06f;
constexpr float CORNELL_LEFT_B = 0.22f;
constexpr float CORNELL_RIGHT_R = 0.0f;
constexpr float CORNELL_RIGHT_G = 0.13f;
constexpr float CORNELL_RIGHT_B = 0.31f;

GeometryData createTerrainGeometry(float width, float height, float depth) {
    GeometryData data;

    data.positions = {
        // back
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        // right
        glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f),
        // left
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        // top
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        // bottom
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f)
    };

    data.normals = {
        // back
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        // right
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        // left
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        // top
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        // bottom
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0)
    };

    glm::vec3 colors[5] = {
        glm::vec3(CORNELL_LEFT_R, CORNELL_LEFT_G, CORNELL_LEFT_B),    // left
        glm::vec3(CORNELL_RIGHT_R, CORNELL_RIGHT_G, CORNELL_RIGHT_B), // right
        glm::vec3(0.96, 0.93, 0.85),                                  // top
        glm::vec3(0.64, 0.64, 0.64),                                  // bottom
        glm::vec3(0.76, 0.74, 0.68)                                   // back
    };

    data.colors = {colors[4], colors[4], colors[4], colors[4],

                   colors[1], colors[1], colors[1], colors[1],

                   colors[0], colors[0], colors[0], colors[0],

                   colors[2], colors[2], colors[2], colors[2],

                   colors[3], colors[3], colors[3], colors[3]};

    data.textureCoordinates = {
        // back
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        // right
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        // left
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        // top
        glm::vec2(0, 1),
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        // bottom
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1)
    };

    // clang-format off
    data.indices = {
        // back
		2, 1, 0,
		0, 3, 2,
        // right
		6, 5, 4,
		4, 7, 6,
        // left
		10, 9, 8,
		8, 11, 10,
        // top
		14, 13, 12,
		12, 15, 14,
        // bottom
		18, 17, 16,
		16, 19, 18
    };
    // clang-format on

    return data;
}

Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data) {
    if (geometry_data.positions.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::positions vector has been passed to createAndUploadIntoGpuMemory(...)");
    }
    if (geometry_data.indices.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::indices vector has been passed to createAndUploadIntoGpuMemory(...)");
    }

    Geometry result;

    // Create vertex positions buffer and copy data into it:
    size_t positions_buffer_byte_size = geometry_data.positions.size() * sizeof(geometry_data.positions[0]);
    result.positionsBuffer = vklCreateHostCoherentBufferAndUploadData(
        static_cast<const void*>(geometry_data.positions.data()),
        positions_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    // Create vertex color buffer and copy data into it:
    result.colorsBuffer = VK_NULL_HANDLE;
    if (geometry_data.colors.size() > 0) {
        size_t colors_buffer_byte_size = geometry_data.colors.size() * sizeof(geometry_data.colors[0]);
        result.colorsBuffer = vklCreateHostCoherentBufferAndUploadData(
            geometry_data.colors.data(),
            colors_buffer_byte_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
        );
    }

    // Create vertex normals buffer and copy data into it:
    size_t normals_buffer_byte_size = geometry_data.normals.size() * sizeof(geometry_data.normals[0]);
    result.normalsBuffer = vklCreateHostCoherentBufferAndUploadData(
        geometry_data.normals.data(),
        normals_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    // Create vertex texture coordinates buffer and copy data into it:
    size_t texture_coordinates_buffer_byte_size = geometry_data.textureCoordinates.size() * sizeof(geometry_data.textureCoordinates[0]);
    result.textureCoordinatesBuffer = vklCreateHostCoherentBufferAndUploadData(
        geometry_data.textureCoordinates.data(),
        static_cast<VkDeviceSize>(texture_coordinates_buffer_byte_size),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    // Create indices buffer and copy data into it:
    size_t indices_buffer_byte_size = geometry_data.indices.size() * sizeof(geometry_data.indices[0]);
    result.indicesBuffer = vklCreateHostCoherentBufferAndUploadData(
        geometry_data.indices.data(),
        static_cast<VkDeviceSize>(indices_buffer_byte_size),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );
    // Also store the number of indices:
    result.numberOfIndices = static_cast<uint32_t>(geometry_data.indices.size());

    return result;
}

void destroyGeometryGpuMemory(const Geometry& geometry) {
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.indicesBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.textureCoordinatesBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.normalsBuffer);
    if (geometry.colorsBuffer != VK_NULL_HANDLE) {
        vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.colorsBuffer);
    }
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.positionsBuffer);
}
