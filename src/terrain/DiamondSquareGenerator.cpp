#include "DiamondSquareGenerator.h"

#include <iostream>

#include "../utils/MathUtils.h"

DiamondSquareGenerator::DiamondSquareGenerator(const TerrainParams& newParams) {
    // Guard: guarantees SetParams sees this as a size change, so it generates indices itself.
    params.gridSizeExponent = -1;
    SetParams(newParams);
}

void DiamondSquareGenerator::ComputeTerrain() {
    size = (1 << params.gridSizeExponent) + 1;
    heights.resize(size * size);
    normals.resize(size * size);
    positions.resize(size * size);
    GenerateHeightMap();
    GeneratePositions();
    DerriveNormals();
}

void DiamondSquareGenerator::SetParams(const TerrainParams& newParams) {
    bool sizeChanged = newParams.gridSizeExponent != params.gridSizeExponent;
    params = newParams;
    ComputeTerrain();
    if (sizeChanged) GenerateIndices();
}

void DiamondSquareGenerator::GenerateIndices() {
    indices.assign((size - 1) * (size - 1) * 6, 0u);
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
    int step = size - 1;
    // Init corner values, a.k.a. square step 0
    float variance = varianceAt(0, params.hurst, params.initialVariance);
    at(heights, size, 0, 0) = batesOffset(params.seed, 0, 0, 0, variance);
    at(heights, size, 0, step) = batesOffset(params.seed, 0, step, 0, variance);
    at(heights, size, step, 0) = batesOffset(params.seed, step, 0, 0, variance);
    at(heights, size, step, step) = batesOffset(params.seed, step, step, 0, variance);
    for (int i = 0; i < params.gridSizeExponent; i++) {
        variance = varianceAt(i + 1, params.hurst, params.initialVariance);
        int halfStep = step / 2;
        // Diamond step
        for (int j = halfStep; j < size - 1; j += step) {
            for (int k = halfStep; k < size - 1; k += step) {
                float leftTop = at(heights, size, j - halfStep, k - halfStep);
                float rightTop = at(heights, size, j + halfStep, k - halfStep);
                float leftBot = at(heights, size, j - halfStep, k + halfStep);
                float rightBot = at(heights, size, j + halfStep, k + halfStep);
                float offset = batesOffset(params.seed, j, k, i + 1, variance);
                float value = (leftTop + rightTop + leftBot + rightBot) / 4.0f + offset;
                at(heights, size, j, k) = value;
            }
        }

        // Square step
        for (int j = 0; j < size; j += halfStep) {
            int kStart = ((j / halfStep) % 2 == 0) ? halfStep : 0;
            for (int k = kStart; k < size; k += step) {
                float sum = 0.0f;
                int count = 0;
                if (j - halfStep >= 0) {
                    sum += at(heights, size, j - halfStep, k);
                    ++count;
                }
                if (j + halfStep < size) {
                    sum += at(heights, size, j + halfStep, k);
                    ++count;
                }
                if (k - halfStep >= 0) {
                    sum += at(heights, size, j, k - halfStep);
                    ++count;
                }
                if (k + halfStep < size) {
                    sum += at(heights, size, j, k + halfStep);
                    ++count;
                }
                float offset = batesOffset(params.seed, j, k, i + 1, variance);
                float value = sum / count + offset;
                at(heights, size, j, k) = value;
            }
        }
        step /= 2;
    }
}

void DiamondSquareGenerator::GeneratePositions() {
    if (heights.empty()) GenerateHeightMap();
    for (int x = 0; x < size; x++) {
        for (int y = 0; y < size; y++) {
            float worldX = (x - size / 2.0f) * params.spacing;
            float worldY = (y - size / 2.0f) * params.spacing;
            float worldZ = at(heights, size, x, y);
            at(positions, size, x, y) = glm::vec3(worldX, worldY, worldZ);
        }
    }
}

void DiamondSquareGenerator::DerriveNormals() {
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

DiamondSquareGenerator::~DiamondSquareGenerator() {}
