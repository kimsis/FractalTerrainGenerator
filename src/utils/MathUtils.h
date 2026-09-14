#pragma once

#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

inline uint32_t packUint16Pair(int high, int low) { return (static_cast<uint32_t>(high) << 16) | (static_cast<uint32_t>(low) & 0xFFFFu); }

inline glm::u32vec4 hash4d(int x, int y, int z, int w) {
    glm::u32vec4 v(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z), static_cast<uint32_t>(w));
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> 16u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
};

inline float batesOffset(uint32_t seed, int x, int y, int level, float targetVariance, int sampleCount = 10) {
    float sum = 0.0f;
    for (int k = 0; k < sampleCount; k++) {
        uint32_t packed = packUint16Pair(level, k);
        sum += hash4d(seed, x, y, packed).x / static_cast<float>(UINT32_MAX);
    }
    float average = sum / sampleCount;
    float centered = average - 0.5f;
    // offset = contered * s, Var(offset) = s² × Var(centered) =
    // = s² × (1/12*sample) => s² = 12*sample*targetVariance => s = sqrt(12*sample*targetVariance)
    return centered * std::sqrt(12 * sampleCount * targetVariance);
}

inline float varianceAt(int level, float hurst, float initialVariance) {
    return initialVariance * std::pow(2.0f, -static_cast<float>(level) * hurst);
}

template <typename T>
inline T& at(std::vector<T>& heights, int size, int x, int y) {
    return heights[x * size + y];
}

template <typename T>
inline const T& at(const std::vector<T>& heights, int size, int x, int y) {
    return heights[x * size + y];
}