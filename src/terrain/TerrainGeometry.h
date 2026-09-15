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
 *	A struct that contains all data for a geometry object on the GPU-side:
 *	Contains all the buffer handles for vertex and index buffers which
 *	can be used for an indexed-geometry draw call.
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

    // A handle to a GPU buffer that contains face indices data. Kept as its own separate
    // allocation (not folded into vertexBuffer) because, unlike positions/normals, it's invariant
    // across a chunk's regenerations and is reused rather than recreated — see
    // createAndUploadIntoGpuMemory's `upload_indices` parameter.
    VkBuffer indicesBuffer;

    // The total number of indices contained within the indicesBuffer.
    uint32_t numberOfIndices;
};

/*!
 *	Creates terrain geometry (positions, normals, indices) for the given params.
 *	@param params parameters for terrain generation
 *	@return all terrain data
 */
GeometryData generateTerrainGeometry(const TerrainParams& params);

/*!
 * Based on the (already populated!) GeometryData, creates gpu buffers for each of its elements
 * in host coherent GPU memory, uploads the data into these buffers, and returns a new Geometry
 * struct which contains handles to these buffers. Ensure to free the memory by using
 * freeGpuMemory(...)!
 *
 * @param	geometry_data	The CPU-side geometry that shall be transferred into GPU-side buffers.
 *							Its positions and normals are combined into a single `vertexBuffer`
 *							allocation (see Geometry).
 * @param	upload_indices	If false, skips creating/uploading the index buffer entirely — the
 *							returned Geometry's `indicesBuffer` is VK_NULL_HANDLE and
 *							`numberOfIndices` is 0, for the caller to fill in from an existing
 *							index buffer instead (see ChunkManager: a chunk's topology never
 *							changes across a Hurst/reseed regeneration, so its one-time index
 *							buffer is reused rather than recreated on every regeneration).
 * @return	A new Geometry instance containing handles to the newly created GPU buffers.
 */
Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data, bool upload_indices = true);

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
 *	Frees whichever of `geometry`'s buffers are non-null. Passing a VK_NULL_HANDLE field is
 *	explicitly safe (a no-op for that buffer) — used when a Geometry doesn't own its index buffer
 *	(see createAndUploadIntoGpuMemory's `upload_indices`) and that field is left null.
 */
void destroyGeometryGpuMemory(const Geometry& geometry);

/*!
 *	Overwrites an already-uploaded Geometry's normals region (within its combined vertexBuffer,
 *	at normalsOffset) in place, leaving positions/indices untouched. Used to patch previously-
 *	extrapolated edge normals once a neighboring chunk's real data becomes available (see
 *	ChunkManager) — a chunk's positions/indices never change after generation, so only its
 *	normals ever need updating post-upload.
 */
void updateGeometryNormals(const Geometry& geometry, const std::vector<glm::vec3>& normals);
