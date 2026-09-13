/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
#pragma once

#include <vulkan/vulkan.h>

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
    // A handle to a GPU buffer that contains vertex position data.
    VkBuffer positionsBuffer;

    // A handle to a GPU buffer that contains vertex normal data.
    VkBuffer normalsBuffer;

    // A handle to a GPU buffer that contains face indices data.
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
 * @return	A new Geometry instance containing handles to the newly created GPU buffers.
 */
Geometry createAndUploadIntoGpuMemory(const GeometryData& geometry_data);

/*!
 *	Frees the GPU buffers that have been created via createAndUploadIntoGpuMemory.
 */
void destroyGeometryGpuMemory(const Geometry& geometry);
