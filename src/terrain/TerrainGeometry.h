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
#include <vector>

#include "DiamondSquareGenerator.h"

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
 *	A struct that contains the GPU-side vertex data buffer for one from/to blend snapshot.
 *	Deliberately does NOT include an index buffer: a chunk's topology is invariant across every
 *	Hurst/reseed regeneration, so it's owned once at the chunk level instead (see
 *	ChunkManager::LoadedChunk) rather than duplicated into every from/to Geometry.
 */
struct Geometry {
    // A handle to a single GPU buffer holding BOTH vertex positions (at byte offset 0) and vertex
    // normals (at byte offset normalsOffset) — combined into one allocation so a Hurst/reseed
    // regeneration costs one vkCreateBuffer/vkAllocateMemory call instead of two (see
    // createAndUploadIntoGpuMemory). Bound twice in the draw call (once per offset) as separate
    // vertex-input bindings.
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
 *							allocation (see Geometry). `indices` is ignored — its buffer is created
 *							separately, once per chunk, by ChunkManager (see LoadedChunk::indicesBuffer).
 * @return	A new Geometry instance containing a handle to the newly created GPU buffer.
 */
Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data);

/*!
 *	Overwrites an ALREADY-ALLOCATED vertexBuffer's positions+normals regions in place (same layout
 *	as createAndUploadIntoGpuMemory: positions at offset 0, normals at the returned offset) —
 *	no GPU allocation happens here at all. `vertexBuffer` must already be big enough to hold
 *	`geometry_data`'s positions+normals, which holds whenever it was originally sized for the same
 *	grid dimensions (a chunk's vertex count never changes after its first load — see ChunkManager's
 *	ping-pong `LoadedChunk::idleVertexBuffer`, the reason this function exists: reusing one of a
 *	chunk's two persistent vertex buffers on every regeneration instead of allocating a new one).
 *
 *	@param	vertexBuffer	An existing buffer, at least as large as this call will write into it.
 *	@param	geometry_data	The CPU-side geometry to overwrite it with.
 *	@return	The byte offset within vertexBuffer where normals were written.
 */
VkDeviceSize uploadVertexDataInPlace(VkBuffer vertexBuffer, const GeometryData& geometry_data);

/*!
 *	Frees `geometry`'s vertex buffer, if it's non-null. Passing a VK_NULL_HANDLE `vertexBuffer` is
 *	explicitly safe (a no-op) — used when wrapping a chunk-level buffer that legitimately might not
 *	have been allocated yet (see ChunkManager::LoadedChunk::idleVertexBuffer). Note this only
 *	destroys the vertex buffer — a chunk's index buffer (see ChunkManager::createAndUploadIndexBuffer)
 *	must be freed separately (also via this function, wrapped in a throwaway Geometry — see ChunkManager).
 */
void destroyGeometryGpuMemory(const Geometry& geometry);

/*!
 *	Overwrites an already-uploaded Geometry's normals region (within its combined vertexBuffer,
 *	at normalsOffset) in place, leaving positions untouched. Used to patch previously-extrapolated
 *	edge normals once a neighboring chunk's real data becomes available (see ChunkManager) — a
 *	chunk's positions never change after generation, so only its normals ever need updating
 *	post-upload.
 */
void updateGeometryNormals(const Geometry& geometry, const std::vector<glm::vec3>& normals);
