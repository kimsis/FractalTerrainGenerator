/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <optional>
#include <vector>

#include "../algorithms/DiamondSquareGenerator.h"

/*!
 *	A struct that contains all data for a geometry object on the CPU-side
 */
struct GeometryData {
    // A vector of vertex positions.
    std::vector<glm::vec3> positions;

    // A vector of vertex indices.
    // Each triple of indices defines one triangle.
    std::vector<uint32_t> indices;

    // A vector of vertex normals.
    std::vector<glm::vec3> normals;
};

/*!
 *	A struct that contains the GPU-side vertex data buffer for one from/to blend snapshot. Does not
 *	include an index buffer — that's owned once at the chunk level (see ChunkManager::LoadedChunk).
 */
struct Geometry {
    // A handle to a single GPU buffer holding both vertex positions (at byte offset 0) and vertex
    // normals (at byte offset normalsOffset). Bound twice in the draw call (once per offset) as
    // separate vertex-input bindings.
    VkBuffer vertexBuffer;

    // Byte offset into vertexBuffer where normal data begins.
    VkDeviceSize normalsOffset;
};

/*!
 *	Creates terrain geometry (positions, normals, indices) for the given params.
 *	@param params parameters for terrain generation
 *	@return all terrain data
 */
GeometryData generateTerrainGeometry(const TerrainParams& params);

/*!
 * Based on the (already populated!) GeometryData, creates a single combined GPU buffer holding
 * positions+normals in host coherent GPU memory and uploads the data into it, returning a new
 * Geometry struct with a handle to that buffer. Ensure to free the memory by using
 * destroyGeometryGpuMemory(...)!
 *
 * @param	geometry_data	The CPU-side geometry that shall be transferred into GPU-side buffers.
 *							Its positions and normals are combined into a single `vertexBuffer`
 *							allocation (see Geometry). `indices` is ignored — see
 *							ChunkManager::createAndUploadIndexBuffer.
 * @return	A new Geometry instance containing a handle to the newly created GPU buffer.
 */
Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data);

/*!
 *	Overwrites an already-allocated vertexBuffer's positions+normals regions in place (same layout
 *	as createAndUploadIntoGpuMemory) — no GPU allocation happens here. `vertexBuffer` must already
 *	be big enough to hold `geometry_data`'s positions+normals.
 *
 *	@param	vertexBuffer	An existing buffer, at least as large as this call will write into it.
 *	@param	geometry_data	The CPU-side geometry to overwrite it with.
 *	@return	The byte offset within vertexBuffer where normals were written.
 */
VkDeviceSize uploadVertexDataInPlace(VkBuffer vertexBuffer, const GeometryData& geometry_data);

/*!
 *	Frees `geometry`'s vertex buffer, if it's non-null. A VK_NULL_HANDLE `vertexBuffer` is
 *	explicitly safe (a no-op). Only destroys the vertex buffer — a chunk's index buffer is freed
 *	separately, also via this function, wrapped in a throwaway Geometry.
 *	@param	geometry	The Geometry whose vertexBuffer shall be freed.
 */
void destroyGeometryGpuMemory(const Geometry& geometry);

/*!
 *	Overwrites an already-uploaded Geometry's normals region in place, leaving positions untouched.
 *	@param	geometry	The already-uploaded Geometry whose normals region shall be overwritten.
 *	@param	normals		The new normals to write; must match the vertex count geometry was uploaded with.
 */
void updateGeometryNormals(const Geometry& geometry, const std::vector<glm::vec3>& normals);

/*!
 * A raycast hit result.
 */
struct Hit {
    glm::vec3 point;
    float distance;
};

/*!
 *	Intersects a ray with the terrain's ground plane (z = 0).
 *	@param	origin		The ray's world-space origin.
 *	@param	direction	The ray's world-space direction (need not be normalized).
 *	@return		The hit point and distance along the ray, or std::nullopt if the ray is parallel
 *				to the ground plane or points away from it.
 */
std::optional<Hit> raycastTerrain(const glm::vec3& origin, const glm::vec3& direction);
