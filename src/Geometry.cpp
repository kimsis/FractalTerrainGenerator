/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */

#include "Geometry.h"

#include <glm/gtc/constants.hpp>

#include "Utils.h"

#undef min
#undef max

constexpr float CORNELL_LEFT_R = 0.49f;
constexpr float CORNELL_LEFT_G = 0.06f;
constexpr float CORNELL_LEFT_B = 0.22f;
constexpr float CORNELL_RIGHT_R = 0.0f;
constexpr float CORNELL_RIGHT_G = 0.13f;
constexpr float CORNELL_RIGHT_B = 0.31f;

GeometryData createBoxGeometry(float width, float height, float depth) {
    GeometryData data;

    data.positions = {
        // front
        glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f),
        // back
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        // right
        glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f),
        // left
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        // top
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        // bottom
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f)
    };

    data.normals = {
        // front
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        // back
        glm::vec3(0, 0, -1),
        glm::vec3(0, 0, -1),
        glm::vec3(0, 0, -1),
        glm::vec3(0, 0, -1),
        // right
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        // left
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        // top
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),
        // bottom
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0)
    };

    data.textureCoordinates = {
        // front
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        // back
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        // right
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        // left
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        // top
        glm::vec2(0, 1),
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        // bottom
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1)
    };

    // clang-format off
    data.indices = {
        // front
		0, 1, 2,
		2, 3, 0,
        // back
		4, 5, 6,
		6, 7, 4,
        // right
		8, 9, 10,
		10, 11, 8,
        // left
		12, 13, 14,
		14, 15, 12,
        // top
		16, 17, 18,
		18, 19, 16,
        // bottom
		20, 21, 22,
		22, 23, 20
    };
    // clang-format on
    return data;
}

GeometryData createCornellBoxGeometry(float width, float height, float depth) {
    GeometryData data;

    data.positions = {
        // back
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        // right
        glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f),
        // left
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        // top
        glm::vec3(-width / 2.0f, height / 2.0f, -depth / 2.0f),
        glm::vec3(-width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, depth / 2.0f),
        glm::vec3(width / 2.0f, height / 2.0f, -depth / 2.0f),
        // bottom
        glm::vec3(-width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, -depth / 2.0f),
        glm::vec3(width / 2.0f, -height / 2.0f, depth / 2.0f),
        glm::vec3(-width / 2.0f, -height / 2.0f, depth / 2.0f)
    };

    data.normals = {
        // back
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),
        // right
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0),
        // left
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),
        // top
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0),
        // bottom
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0)
    };

    glm::vec3 colors[5] = {
        glm::vec3(CORNELL_LEFT_R, CORNELL_LEFT_G, CORNELL_LEFT_B),    // left
        glm::vec3(CORNELL_RIGHT_R, CORNELL_RIGHT_G, CORNELL_RIGHT_B), // right
        glm::vec3(0.96, 0.93, 0.85),                                  // top
        glm::vec3(0.64, 0.64, 0.64),                                  // bottom
        glm::vec3(0.76, 0.74, 0.68)                                   // back
    };

    data.colors = {colors[4], colors[4], colors[4], colors[4],

                   colors[1], colors[1], colors[1], colors[1],

                   colors[0], colors[0], colors[0], colors[0],

                   colors[2], colors[2], colors[2], colors[2],

                   colors[3], colors[3], colors[3], colors[3]};

    data.textureCoordinates = {
        // back
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        // right
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        // left
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1),
        // top
        glm::vec2(0, 1),
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        // bottom
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(1, 1),
        glm::vec2(0, 1)
    };

    // clang-format off
    data.indices = {
        // back
		2, 1, 0,
		0, 3, 2,
        // right
		6, 5, 4,
		4, 7, 6,
        // left
		10, 9, 8,
		8, 11, 10,
        // top
		14, 13, 12,
		12, 15, 14,
        // bottom
		18, 17, 16,
		16, 19, 18
    };
    // clang-format on

    return data;
}
// clang-format on
GeometryData createCylinderGeometry(uint32_t segments, float height, float radius) {
    GeometryData data;

    // center vertices
    data.positions.push_back(glm::vec3(0, -height / 2.0f, 0));
    data.normals.push_back(glm::vec3(0, -1, 0));
    data.textureCoordinates.push_back(glm::vec2(0.5f, 0.5f));
    data.positions.push_back(glm::vec3(0, height / 2.0f, 0));
    data.normals.push_back(glm::vec3(0, 1, 0));
    data.textureCoordinates.push_back(glm::vec2(0.5f, 0.5f));

    // Seam vertices for the lateral (side) surface: duplicates of segment 0's side vertices
    // with u=1 instead of u=0, so the last quad's texture doesn't wrap backwards to u=0.
    uint32_t bottomSeamIndex = 2 + segments * 4;
    uint32_t topSeamIndex = 3 + segments * 4;

    // circle segments
    float angle_step = 2.0f * glm::pi<float>() / float(segments);
    for (uint32_t i = 0; i < segments; i++) {
        glm::vec3 circlePos = glm::vec3(glm::cos(i * angle_step) * radius, -height / 2.0f, glm::sin(i * angle_step) * radius);

        glm::vec2 squareToCircleUV = glm::vec2((circlePos.x / radius) * 0.5f + 0.5f, (circlePos.z / radius) * 0.5f + 0.5f);

        // bottom ring vertex
        data.positions.push_back(circlePos);
        data.positions.push_back(circlePos);
        data.normals.push_back(glm::vec3(0, -1, 0));
        data.normals.push_back(glm::normalize(circlePos - glm::vec3(0, -height / 2.0f, 0)));
        data.textureCoordinates.push_back(glm::vec2(squareToCircleUV.x, 1.0f - squareToCircleUV.y));
        data.textureCoordinates.push_back(glm::vec2(i * angle_step / (2.0f * glm::pi<float>()), 0));

        // top ring vertex
        circlePos.y = height / 2.0f;
        data.positions.push_back(circlePos);
        data.positions.push_back(circlePos);
        data.normals.push_back(glm::vec3(0, 1, 0));
        data.normals.push_back(glm::normalize(circlePos - glm::vec3(0, height / 2.0f, 0)));
        data.textureCoordinates.push_back(squareToCircleUV);
        data.textureCoordinates.push_back(glm::vec2(i * angle_step / (2.0f * glm::pi<float>()), 1));

        // bottom face
        data.indices.push_back(0);
        data.indices.push_back(2 + i * 4);
        data.indices.push_back(i == segments - 1 ? 2 : 2 + (i + 1) * 4);

        // top face
        data.indices.push_back(1);
        data.indices.push_back(i == segments - 1 ? 4 : (i + 2) * 4);
        data.indices.push_back((i + 1) * 4);

        // side faces
        data.indices.push_back(3 + i * 4);
        data.indices.push_back(i == segments - 1 ? topSeamIndex : 5 + (i + 1) * 4);
        data.indices.push_back(i == segments - 1 ? bottomSeamIndex : 3 + (i + 1) * 4);

        data.indices.push_back(i == segments - 1 ? topSeamIndex : 5 + (i + 1) * 4);
        data.indices.push_back(3 + i * 4);
        data.indices.push_back(5 + i * 4);
    }

    // Append the seam vertices (same position/normal as segment 0's side vertices, u=1)
    data.positions.push_back(data.positions[3]);
    data.normals.push_back(data.normals[3]);
    data.textureCoordinates.push_back(glm::vec2(1.0f, 0.0f));

    data.positions.push_back(data.positions[5]);
    data.normals.push_back(data.normals[5]);
    data.textureCoordinates.push_back(glm::vec2(1.0f, 1.0f));

    return data;
}

// Function to calculate binomial coefficient (n choose k)
int binomialCoefficient(int n, int k) {
    int result = 1;
    for (int i = 1; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}

// Function to calculate a point on the Bezier curve
glm::vec3 calculateBezierPoint(const std::vector<glm::vec3>& controlPoints, float t) {
    int n = controlPoints.size() - 1;
    glm::vec3 point(0.0f, 0.0f, 0.0f);
    for (int i = 0; i <= n; ++i) {
        float blend = binomialCoefficient(n, i) * pow(t, i) * pow(1 - t, n - i);
        point += controlPoints[i] * blend;
    }
    return point;
}

// Function to calculate derivative (tangent) at a point on the Bezier curve
glm::vec3 calculateBezierTangent(const std::vector<glm::vec3>& controlPoints, float t) {
    int n = controlPoints.size() - 1;
    glm::vec3 tangent(0.0f);

    for (int i = 0; i < n; ++i) {
        glm::vec3 diff = controlPoints[i + 1] - controlPoints[i];
        float blend = binomialCoefficient(n - 1, i) * pow(t, i) * pow(1 - t, (n - 1) - i);
        tangent += diff * (float)n * blend;
    }

    return glm::normalize(tangent);
}

// Function to generate a Bezier curve and subdivide it into N segments
void generateBezierCurve(
    const std::vector<glm::vec3>& controlPoints,
    int numSegments,
    std::vector<glm::vec3>& positions,
    std::vector<glm::vec3>& tangents
) {
    float deltaT = 1.0f / (numSegments);
    for (int i = 0; i <= numSegments; ++i) {
        float t = i * deltaT;
        glm::vec3 position = calculateBezierPoint(controlPoints, t);
        glm::vec3 tangent = calculateBezierTangent(controlPoints, t);
        positions.push_back(position);
        tangents.push_back(tangent);
    }
}

GeometryData
createBezierCylinderGeometry(uint32_t n_circular_segments, std::vector<glm::vec3> controlPoints, uint32_t s_bezier_segments, float radius) {
    GeometryData data;
    std::vector<glm::vec3> bezierPoints;
    std::vector<glm::vec3> bezierTangents;
    generateBezierCurve(controlPoints, s_bezier_segments, bezierPoints, bezierTangents);
    float v = 0;
    float angleStep = 2.0f * glm::pi<float>() / float(n_circular_segments);
    for (int point = 0; point < bezierPoints.size(); point++) {
        glm::vec3 forwardAxis = bezierTangents[point];
        glm::vec3 rightAxis = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), forwardAxis));
        glm::vec3 upAxis = glm::normalize(glm::cross(forwardAxis, rightAxis));

        // Circle segments
        uint32_t startIndex = data.positions.size();
        for (uint32_t i = 0; i < n_circular_segments; i++) {
            float cosTheta = glm::cos(i * angleStep);
            float sinTheta = glm::sin(i * angleStep);
            glm::vec3 circlePos = bezierPoints[point] + cosTheta * radius * rightAxis + sinTheta * radius * upAxis;
            data.positions.push_back(circlePos);

            data.normals.push_back(circlePos - bezierPoints[point]);
            float u = static_cast<float>(i) / static_cast<float>(n_circular_segments);
            data.textureCoordinates.push_back(glm::vec2(u, v));
            // Side faces (ring stride is n_circular_segments + 1 to account for the seam vertex below)
            if (point < bezierPoints.size() - 1) {
                uint32_t ringStride = n_circular_segments + 1;
                data.indices.push_back(startIndex + i);
                data.indices.push_back(startIndex + i + 1);
                data.indices.push_back(startIndex + ringStride + i + 1);

                data.indices.push_back(startIndex + ringStride + i + 1);
                data.indices.push_back(startIndex + ringStride + i);
                data.indices.push_back(startIndex + i);
            }
        }
        // Seam vertex: duplicate of this ring's i=0 vertex with u=1, so the lateral texture
        // doesn't wrap backwards from u=(n-1)/n to u=0 across the last quad.
        glm::vec3 seamCirclePos = bezierPoints[point] + radius * rightAxis;
        data.positions.push_back(seamCirclePos);
        data.normals.push_back(seamCirclePos - bezierPoints[point]);
        data.textureCoordinates.push_back(glm::vec2(1.0f, v));

        if (point < bezierPoints.size() - 1) {
            v += glm::min(glm::length(bezierPoints[point + 1] - bezierPoints[point]), 1.0f);
        }
    }
    // top face
    data.positions.push_back(bezierPoints[bezierPoints.size() - 1]);
    data.normals.push_back(bezierTangents[bezierTangents.size() - 1]);
    data.textureCoordinates.push_back(glm::vec2(0.5f, 0.5f));
    glm::vec3 forwardAxis = bezierTangents[bezierTangents.size() - 1];
    glm::vec3 rightAxis = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), forwardAxis));
    glm::vec3 upAxis = glm::normalize(glm::cross(forwardAxis, rightAxis));
    int numberpositions = data.positions.size() - 1;
    for (unsigned int i = 0; i <= n_circular_segments; i++) {
        data.normals.push_back(bezierTangents[bezierTangents.size() - 1]);
        glm::vec3 circlePosFlat = glm::vec3(glm::cos(i * angleStep) * radius, 0, glm::sin(i * angleStep) * radius);
        glm::vec2 squareToCircleUV = glm::vec2((circlePosFlat.x / radius) * 0.5f + 0.5f, (circlePosFlat.z / radius) * 0.5f + 0.5f);
        data.textureCoordinates.push_back(squareToCircleUV);
        float cosTheta = glm::cos(i * angleStep);
        float sinTheta = glm::sin(i * angleStep);

        glm::vec3 circlePos = bezierPoints[bezierPoints.size() - 1] + cosTheta * radius * rightAxis + sinTheta * radius * upAxis;
        data.positions.push_back(circlePos);
        data.indices.push_back(numberpositions + (i + 1));
        data.indices.push_back(numberpositions);
        data.indices.push_back(numberpositions + i);
    }

    // Bottom face
    data.positions.push_back(bezierPoints[0]);
    numberpositions = data.positions.size() - 1;
    forwardAxis = bezierTangents[0];
    rightAxis = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), forwardAxis));
    upAxis = glm::normalize(glm::cross(forwardAxis, rightAxis));
    data.normals.push_back(-bezierTangents[0]);
    data.textureCoordinates.push_back(glm::vec2(0.5f, 0.5f));
    for (unsigned int i = 0; i <= n_circular_segments; i++) {
        data.normals.push_back(-bezierTangents[0]);
        glm::vec3 circlePosFlat = glm::vec3(glm::cos(i * angleStep) * radius, 0, glm::sin(i * angleStep) * radius);
        glm::vec2 squareToCircleUV = glm::vec2((circlePosFlat.x / radius) * 0.5f + 0.5f, (circlePosFlat.z / radius) * 0.5f + 0.5f);
        data.textureCoordinates.push_back(squareToCircleUV);
        float cosTheta = glm::cos(i * angleStep);
        float sinTheta = glm::sin(i * angleStep);

        glm::vec3 circlePos = bezierPoints[0] + cosTheta * radius * rightAxis + sinTheta * radius * upAxis;
        data.positions.push_back(circlePos);
        data.indices.push_back(numberpositions);
        data.indices.push_back(numberpositions + (i + 1));
        data.indices.push_back(numberpositions + i);
    }
    return data;
}

GeometryData createSphereGeometry(uint32_t longitude_segments, uint32_t latitude_segments, float radius) {
    GeometryData data;

    // Each ring gets one extra seam vertex (u=1, duplicate of j=0) so the lateral texture
    // doesn't wrap backwards across the last quad.
    uint32_t verticesPerRing = longitude_segments + 1;

    // North pole: one vertex per longitude segment (instead of a single shared vertex) so
    // each fan triangle interpolates its own correctly varying U instead of pinching to one point.
    uint32_t northPoleStart = static_cast<uint32_t>(data.positions.size());
    for (uint32_t j = 0; j < longitude_segments; j++) {
        data.positions.push_back(glm::vec3(0.0f, radius, 0.0f));
        data.normals.push_back(glm::vec3(0.0f, radius, 0.0f));
        float u = (float(j) + 0.5f) / float(longitude_segments);
        data.textureCoordinates.push_back(glm::vec2(u, 0.0f));
    }

    // South pole: same reasoning as the north pole.
    uint32_t southPoleStart = static_cast<uint32_t>(data.positions.size());
    for (uint32_t j = 0; j < longitude_segments; j++) {
        data.positions.push_back(glm::vec3(0.0f, -radius, 0.0f));
        data.normals.push_back(glm::vec3(0.0f, -radius, 0.0f));
        float u = (float(j) + 0.5f) / float(longitude_segments);
        data.textureCoordinates.push_back(glm::vec2(u, 1.0f));
    }

    uint32_t firstRingStart = static_cast<uint32_t>(data.positions.size());

    // vertices and rings
    for (uint32_t i = 1; i < latitude_segments; i++) {
        float verticalAngle = float(i) * glm::pi<float>() / float(latitude_segments);
        for (uint32_t j = 0; j <= longitude_segments; j++) {
            float horizontalAngle = float(j) * 2.0f * glm::pi<float>() / float(longitude_segments);
            glm::vec3 position = glm::vec3(
                radius * glm::sin(verticalAngle) * glm::cos(horizontalAngle),
                radius * glm::cos(verticalAngle),
                radius * glm::sin(verticalAngle) * glm::sin(horizontalAngle)
            );
            data.positions.push_back(position);
            data.normals.push_back(glm::normalize(position));

            float u = float(j) / float(longitude_segments);
            data.textureCoordinates.push_back(glm::vec2(u, verticalAngle / glm::pi<float>()));

            if (i == 1 || j == longitude_segments) continue;

            uint32_t ringStart = firstRingStart + (i - 1) * verticesPerRing;
            uint32_t prevRingStart = firstRingStart + (i - 2) * verticesPerRing;

            data.indices.push_back(ringStart + j);
            data.indices.push_back(prevRingStart + j + 1);
            data.indices.push_back(ringStart + j + 1);

            data.indices.push_back(prevRingStart + j + 1);
            data.indices.push_back(ringStart + j);
            data.indices.push_back(prevRingStart + j);
        }
    }

    // North pole to first ring
    for (uint32_t j = 0; j < longitude_segments; j++) {
        data.indices.push_back(northPoleStart + j);
        data.indices.push_back(firstRingStart + j + 1);
        data.indices.push_back(firstRingStart + j);
    }

    // Last ring to south pole
    uint32_t lastRingStart = firstRingStart + (latitude_segments - 2) * verticesPerRing;
    for (uint32_t j = 0; j < longitude_segments; j++) {
        data.indices.push_back(southPoleStart + j);
        data.indices.push_back(lastRingStart + j);
        data.indices.push_back(lastRingStart + j + 1);
    }

    return data;
}

Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data) {
    if (geometry_data.positions.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::positions vector has been passed to createAndUploadIntoGpuMemory(...)");
    }
    if (geometry_data.indices.empty()) {
        VKL_EXIT_WITH_ERROR("An empty GeometryData::indices vector has been passed to createAndUploadIntoGpuMemory(...)");
    }

    Geometry result;

    // Create vertex positions buffer and copy data into it:
    size_t positions_buffer_byte_size = geometry_data.positions.size() * sizeof(geometry_data.positions[0]);
    result.positionsBuffer = vklCreateHostCoherentBufferAndUploadData(
        static_cast<const void*>(geometry_data.positions.data()),
        positions_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );

    // Create vertex color buffer and copy data into it:
    result.colorsBuffer = VK_NULL_HANDLE;
    if (geometry_data.colors.size() > 0) {
        size_t colors_buffer_byte_size = geometry_data.colors.size() * sizeof(geometry_data.colors[0]);
        result.colorsBuffer = vklCreateHostCoherentBufferAndUploadData(
            geometry_data.colors.data(),
            colors_buffer_byte_size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
        );
    }

    // Create vertex normals buffer and copy data into it:
    size_t normals_buffer_byte_size = geometry_data.normals.size() * sizeof(geometry_data.normals[0]);
    result.normalsBuffer = vklCreateHostCoherentBufferAndUploadData(
        geometry_data.normals.data(),
        normals_buffer_byte_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    // Create vertex texture coordinates buffer and copy data into it:
    size_t texture_coordinates_buffer_byte_size = geometry_data.textureCoordinates.size() * sizeof(geometry_data.textureCoordinates[0]);
    result.textureCoordinatesBuffer = vklCreateHostCoherentBufferAndUploadData(
        geometry_data.textureCoordinates.data(),
        static_cast<VkDeviceSize>(texture_coordinates_buffer_byte_size),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    // Create indices buffer and copy data into it:
    size_t indices_buffer_byte_size = geometry_data.indices.size() * sizeof(geometry_data.indices[0]);
    result.indicesBuffer = vklCreateHostCoherentBufferAndUploadData(
        geometry_data.indices.data(),
        static_cast<VkDeviceSize>(indices_buffer_byte_size),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );
    // Also store the number of indices:
    result.numberOfIndices = static_cast<uint32_t>(geometry_data.indices.size());

    return result;
}

void destroyGeometryGpuMemory(const Geometry& geometry) {
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.indicesBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.textureCoordinatesBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.normalsBuffer);
    if (geometry.colorsBuffer != VK_NULL_HANDLE) {
        vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.colorsBuffer);
    }
    vklDestroyHostCoherentBufferAndItsBackingMemory(geometry.positionsBuffer);
}
