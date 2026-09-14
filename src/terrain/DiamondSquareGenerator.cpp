#include "DiamondSquareGenerator.h"

#include <cmath>

#include "../utils/MathUtils.h"

DiamondSquareGenerator::DiamondSquareGenerator(const TerrainParams& newParams) {
    // Guard: guarantees SetParams sees this as a size change, so it generates indices itself.
    params.gridSizeExponent = -1;
    SetParams(newParams);
}

void DiamondSquareGenerator::SetParams(const TerrainParams& newParams) {
    bool sizeChanged = newParams.gridSizeExponent != params.gridSizeExponent;
    params = newParams;
    ComputeTerrain();
    if (sizeChanged) GenerateIndices();
}
const TerrainParams& DiamondSquareGenerator::GetParams() const { return params; };
const std::vector<uint32_t>& DiamondSquareGenerator::getIndices() const { return indices; };
const std::vector<glm::vec3>& DiamondSquareGenerator::getPositions() const { return positions; };
const int DiamondSquareGenerator::getWorldGridX(int x) const { return params.chunkX * (size - 1) + x; };
const int DiamondSquareGenerator::getWorldGridY(int y) const { return params.chunkY * (size - 1) + y; };

ChunkCoord cameraToChunkCoord(const glm::vec3& pos, const TerrainParams& params) {
    int size = (1 << params.gridSizeExponent) + 1;
    float globalGridX = pos.x / params.spacing + size / 2.0f;
    float globalGridY = pos.y / params.spacing + size / 2.0f;
    int cx = static_cast<int>(std::floor(globalGridX / (size - 1)));
    int cy = static_cast<int>(std::floor(globalGridY / (size - 1)));
    return ChunkCoord{cx, cy};
}

void DiamondSquareGenerator::GenerateIndices() {
    indices.resize((size - 1) * (size - 1) * 6);
    int counter = 0;
    for (int x = 0; x < size - 1; x++) {
        for (int y = 0; y < size - 1; y++) {
            uint32_t topLeft = x * size + y;
            uint32_t topRight = (x + 1) * size + y;
            uint32_t botLeft = x * size + y + 1;
            uint32_t botRight = (x + 1) * size + y + 1;
            // triangle 1
            indices[counter++] = topLeft;
            indices[counter++] = topRight;
            indices[counter++] = botLeft;
            // triangle 2
            indices[counter++] = topRight;
            indices[counter++] = botRight;
            indices[counter++] = botLeft;
        }
    }
}

void DiamondSquareGenerator::GenerateHeightMap() {
    heights.resize(size * size);
    int step = size - 1;
    // Init corner values, a.k.a. square step 0
    float variance = varianceAt(0, params.hurst, params.initialVariance);
    at(heights, size, 0, 0) = batesOffset(params.seed, getWorldGridX(0), getWorldGridY(0), 0, variance);
    at(heights, size, 0, step) = batesOffset(params.seed, getWorldGridX(0), getWorldGridY(step), 0, variance);
    at(heights, size, step, 0) = batesOffset(params.seed, getWorldGridX(step), getWorldGridY(0), 0, variance);
    at(heights, size, step, step) = batesOffset(params.seed, getWorldGridX(step), getWorldGridY(step), 0, variance);
    for (int i = 0; i < params.gridSizeExponent; i++) {
        variance = varianceAt(i + 1, params.hurst, params.initialVariance);
        int halfStep = step / 2;
        // Diamond step
        for (int x = halfStep; x < size - 1; x += step) {
            for (int y = halfStep; y < size - 1; y += step) {
                float leftTop = at(heights, size, x - halfStep, y - halfStep);
                float rightTop = at(heights, size, x + halfStep, y - halfStep);
                float leftBot = at(heights, size, x - halfStep, y + halfStep);
                float rightBot = at(heights, size, x + halfStep, y + halfStep);
                float offset = batesOffset(params.seed, getWorldGridX(x), getWorldGridY(y), i + 1, variance);
                float value = (leftTop + rightTop + leftBot + rightBot) / 4.0f + offset;
                at(heights, size, x, y) = value;
            }
        }

        // Square step
        for (int x = 0; x < size; x += halfStep) {
            int kStart = ((x / halfStep) % 2 == 0) ? halfStep : 0;
            for (int y = kStart; y < size; y += step) {
                float sum = 0.0f;
                int count = 0;
                if (x - halfStep >= 0 && x != 0 && x != size - 1) {
                    sum += at(heights, size, x - halfStep, y);
                    ++count;
                }
                if (x + halfStep < size && x != 0 && x != size - 1) {
                    sum += at(heights, size, x + halfStep, y);
                    ++count;
                }
                if (y - halfStep >= 0 && y != 0 && y != size - 1) {
                    sum += at(heights, size, x, y - halfStep);
                    ++count;
                }
                if (y + halfStep < size && y != 0 && y != size - 1) {
                    sum += at(heights, size, x, y + halfStep);
                    ++count;
                }
                float offset = batesOffset(params.seed, getWorldGridX(x), getWorldGridY(y), i + 1, variance);
                float value = sum / count + offset;
                at(heights, size, x, y) = value;
            }
        }
        step /= 2;
    }
}

void DiamondSquareGenerator::GeneratePositions() {
    positions.resize(size * size);
    if (heights.empty()) GenerateHeightMap();
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            float worldX = (getWorldGridX(x) - (size - 1) / 2.0f) * params.spacing;
            float worldY = (getWorldGridY(y) - (size - 1) / 2.0f) * params.spacing;
            float worldZ = at(heights, size, x, y);
            at(positions, size, x, y) = glm::vec3(worldX, worldY, worldZ);
        }
    }
}

void DiamondSquareGenerator::ComputeTerrain() {
    size = (1 << params.gridSizeExponent) + 1;
    GenerateHeightMap();
    GeneratePositions();
}

std::vector<glm::vec3> deriveTerrainNormals(
    const std::vector<glm::vec3>& positions,
    int size,
    int spacing,
    const std::vector<float>* leftSkirt,
    const std::vector<float>* rightSkirt,
    const std::vector<float>* topSkirt,
    const std::vector<float>* bottomSkirt
) {
    std::vector<glm::vec3> normals(size * size);
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            float here = at(positions, size, x, y).z;

            // At a chunk edge, prefer the real neighbor value (borrowed from that neighbor's
            // already-computed positions) if we have one; otherwise fall back to linearly
            // extrapolating through this cell from the real neighbor on the other side — used only
            // when that side's neighbor genuinely doesn't exist (outside the loaded view radius).
            float left = (x > 0) ? at(positions, size, x - 1, y).z : leftSkirt ? (*leftSkirt)[y] : 2.0f * here - at(positions, size, x + 1, y).z;
            float right = (x < size - 1) ? at(positions, size, x + 1, y).z
                          : rightSkirt   ? (*rightSkirt)[y]
                                         : 2.0f * here - at(positions, size, x - 1, y).z;
            float dzdx = (right - left) / (2.0f * spacing);

            float down = (y > 0) ? at(positions, size, x, y - 1).z : bottomSkirt ? (*bottomSkirt)[x] : 2.0f * here - at(positions, size, x, y + 1).z;
            float up = (y < size - 1) ? at(positions, size, x, y + 1).z : topSkirt ? (*topSkirt)[x] : 2.0f * here - at(positions, size, x, y - 1).z;
            float dzdy = (up - down) / (2.0f * spacing);

            // Cross product of both derivatives
            glm::vec3 normal(-dzdx, -dzdy, 1);
            normals[x * size + y] = glm::normalize(normal);
        }
    }
    return normals;
}

DiamondSquareGenerator::~DiamondSquareGenerator() {}
