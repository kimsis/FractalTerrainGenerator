#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

#include "ChunkCoord.h"

struct TerrainParams {
    int gridSizeExponent = 4; // gridSize = 2^gridSizeExponent + 1
    float hurst = 0.8f;
    uint32_t seed = 1337u;
    float initialVariance = 25.0f;
    // only needed for non-square maps, kept just in case
    int spacing = 10;
    int chunkX = 0;
    int chunkY = 0;
};

/*!
 *	Inverse of getWorldGridX/getWorldGridY: given a world-space position, returns which chunk
 *	(under the given grid size/spacing) that position falls in. Ignores pos.z (chunks are laid
 *	out in the XY plane).
 */
ChunkCoord cameraToChunkCoord(const glm::vec3& pos, const TerrainParams& params);

class DiamondSquareGenerator {
   private:
    uint16_t size;
    TerrainParams params;
    std::vector<float> heights;
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;

   public:
    DiamondSquareGenerator(const TerrainParams& newParams = TerrainParams());
    void SetParams(const TerrainParams& newParams);
    const TerrainParams& GetParams() const;
    const std::vector<uint32_t>& getIndices() const;
    const std::vector<glm::vec3>& getPositions() const;
    const int getWorldGridX(int x) const;
    const int getWorldGridY(int y) const;
    void GenerateIndices();
    void GenerateHeightMap();
    void GeneratePositions();

    // Computes heights + positions + indices only — no normals. This is the part that's fully
    // independent between chunks (see deriveTerrainNormals for why normals aren't).
    void ComputeTerrain();

    ~DiamondSquareGenerator();
};

/*!
 *	Derives per-vertex normals for a `size`x`size` grid of `positions` (indexed as
 *	positions[x*size+y], z = height) via central differences. At the grid's own boundary, a
 *	one-sided difference can't be computed from `positions` alone, so each side optionally takes a
 *	real boundary row/column sampled from that neighbor chunk (each exactly `size` values, indexed
 *	the same way as this chunk's own y/x respectively) — pass nullptr for a side whose neighbor
 *	doesn't exist or hasn't finished its own heights/positions yet, and that edge falls back to
 *	linearly extrapolating through the boundary cell from its interior instead.
 */
std::vector<glm::vec3> deriveTerrainNormals(
    const std::vector<glm::vec3>& positions,
    int size,
    int spacing,
    const std::vector<float>* leftSkirt,
    const std::vector<float>* rightSkirt,
    const std::vector<float>* topSkirt,
    const std::vector<float>* bottomSkirt
);
