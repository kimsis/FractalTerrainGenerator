#include "WaterGeometry.h"

#include <VulkanLaunchpad.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

WaterChunkGeometry buildWaterChunkGeometry(const ChunkCoord& coord, const TerrainParams& baseParams) {
    int size = (1 << baseParams.gridSizeExponent) + 1;
    float half_extent = ((size - 1) / 2.0f) * baseParams.spacing; // (size-1) cells rendered, not size vertices
    float centerX = coord.cx * (size - 1) * static_cast<float>(baseParams.spacing);
    float centerY = coord.cy * (size - 1) * static_cast<float>(baseParams.spacing);

    std::vector<glm::vec3> positions = {
        {centerX - half_extent, centerY - half_extent, 0.0f},
        {centerX + half_extent, centerY - half_extent, 0.0f},
        {centerX + half_extent, centerY + half_extent, 0.0f},
        {centerX - half_extent, centerY + half_extent, 0.0f},
    };
    std::vector<uint32_t> indices = {0u, 1u, 2u, 0u, 2u, 3u};

    WaterChunkGeometry geometry{};
    geometry.positionsBuffer = vklCreateHostCoherentBufferAndUploadData(
        positions.data(),
        positions.size() * sizeof(glm::vec3),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    geometry.indicesBuffer = vklCreateHostCoherentBufferAndUploadData(
        indices.data(),
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );
    geometry.numberOfIndices = static_cast<uint32_t>(indices.size());
    return geometry;
}

void destroyWaterChunkGeometry(const WaterChunkGeometry& geometry) {
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.positionsBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.indicesBuffer);
}
