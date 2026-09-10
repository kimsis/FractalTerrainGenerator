#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

struct TerrainParams {
    int gridSizeExponent = 9; // gridSize = 2^gridSizeExponent + 1
    float hurst = 0.6f;
    uint32_t seed = 1337u;
    float initialVariance = 1.0f;
    // only needed for non-square maps, kept just in case
    int spacing = 1;
};

class DiamondSquareGenerator {
   private:
    uint16_t size;
    TerrainParams params;
    std::vector<float> heights;
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> positions;

   public:
    DiamondSquareGenerator(const TerrainParams& newParams = TerrainParams());
    void SetParams(const TerrainParams& newParams);
    const TerrainParams& GetParams() const { return params; };
    const std::vector<uint32_t>& getIndices() const { return indices; };
    const std::vector<glm::vec3>& getNormals() const { return normals; };
    const std::vector<glm::vec3>& getPositions() const { return positions; };
    void GenerateIndices();
    void GenerateHeightMap();
    void GeneratePositions();
    void DerriveNormals();
    void ComputeTerrain();
    ~DiamondSquareGenerator();
};
