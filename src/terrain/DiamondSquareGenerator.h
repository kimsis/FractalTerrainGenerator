#pragma once
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

struct TerrainParams {
    int gridSizeExponent = 11; // gridSize = 2^gridSizeExponent + 1
    float hurst = 0.8f;
    uint32_t seed = 1337u;
    float initialVariance = 25.0f;
    // only needed for non-square maps, kept just in case
    int spacing = 1;
    int chunkX = 0;
    int chunkY = 0;
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
    const TerrainParams& GetParams() const;
    const std::vector<uint32_t>& getIndices() const;
    const std::vector<glm::vec3>& getNormals() const;
    const std::vector<glm::vec3>& getPositions() const;
    const int getWorldGridX(int x) const;
    const int getWorldGridY(int y) const;
    void GenerateIndices();
    void GenerateHeightMap();
    void GeneratePositions();
    void DerriveNormals();
    void ComputeTerrain();
    ~DiamondSquareGenerator();
};
