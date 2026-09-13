#include "DiamondSquareGenerator.h"

#include <iostream>

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
const std::vector<glm::vec3>& DiamondSquareGenerator::getNormals() const { return normals; };
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
    at(heights, size, 0, 0) = batesOffset(params.seed, 0, 0, 0, variance);
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
            float worldX = (getWorldGridX(x) - size / 2.0f) * params.spacing;
            float worldY = (getWorldGridY(y) - size / 2.0f) * params.spacing;
            float worldZ = at(heights, size, x, y);
            at(positions, size, x, y) = glm::vec3(worldX, worldY, worldZ);
        }
    }
}

void DiamondSquareGenerator::DerriveNormals() {
    normals.resize(size * size);
    if (heights.empty()) GenerateHeightMap();
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            int neighboursCount = 0;
            float dzdx = 0.0f;
            if (x > 0) {
                dzdx -= at(heights, size, x - 1, y);
                neighboursCount++;
            }
            if (x < size - 1) {
                dzdx += at(heights, size, x + 1, y);
                neighboursCount++;
            }
            dzdx /= (neighboursCount * params.spacing);
            neighboursCount = 0;
            float dzdy = 0.0f;
            if (y > 0) {
                dzdy -= at(heights, size, x, y - 1);
                neighboursCount++;
            }
            if (y < size - 1) {
                dzdy += at(heights, size, x, y + 1);
                neighboursCount++;
            }
            dzdy /= (neighboursCount * params.spacing);
            // Cross product of both derivatives
            glm::vec3 normal(-dzdx, -dzdy, 1);
            at(normals, size, x, y) = glm::normalize(normal);
        }
    }
}

void DiamondSquareGenerator::ComputeTerrain() {
    size = (1 << params.gridSizeExponent) + 1;
    GenerateHeightMap();
    GeneratePositions();
    DerriveNormals();
}

DiamondSquareGenerator::~DiamondSquareGenerator() {}
