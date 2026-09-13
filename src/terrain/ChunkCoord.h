#pragma once

#include <cstdint>
#include <functional>

#include "../utils/MathUtils.h"

struct ChunkCoord {
    int cx, cy;
    bool operator==(const ChunkCoord& other) const { return cx == other.cx && cy == other.cy; }
};

namespace std {
template <>
struct hash<ChunkCoord> {
    std::size_t operator()(const ChunkCoord& c) const { return std::hash<int>()(c.cx) ^ (std::hash<int>()(c.cy) << 1); }
};
} // namespace std
