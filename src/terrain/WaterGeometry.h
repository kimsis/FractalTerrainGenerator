#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

#include "ChunkCoord.h"
#include "DiamondSquareGenerator.h"

/*!
 *	One loaded chunk's water quad: positions are baked directly in world space (that chunk's XY
 *	offset already added in), like terrain's own per-chunk Geometry.
 */
struct WaterChunkGeometry {
    VkBuffer positionsBuffer;
    VkBuffer indicesBuffer;
    uint32_t numberOfIndices;
};

/*!
 *	Builds one chunk's water quad with world-space XY already baked in (local z=0 — a shared
 *	per-frame model matrix handles the water-level Z).
 */
WaterChunkGeometry buildWaterChunkGeometry(const ChunkCoord& coord, const TerrainParams& baseParams);

/*!
 *	Destroys the GPU buffers for one chunk's water quad, created via buildWaterChunkGeometry.
 */
void destroyWaterChunkGeometry(const WaterChunkGeometry& geometry);
