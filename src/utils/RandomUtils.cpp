#include "RandomUtils.h"

#include <random>

uint32_t generateRandomSeed() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<uint32_t> dist;
    return dist(rng);
}

float generateRandomHurst() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> dist(0.4f, 0.95f);
    return dist(rng);
}

float generateRandomHeightScale() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> dist(1.0f, 5.0f);
    return dist(rng);
}

float generateRandomWaterLevel() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-15.0f, 15.0f);
    return dist(rng);
}
