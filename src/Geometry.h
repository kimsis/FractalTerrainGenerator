/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
#pragma once


#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

/*!
 *	A struct that contains all data for a geometry object on the CPU-side
 */
struct GeometryData {
    // A vector of vertex positions.
    std::vector<glm::vec3> positions;

    // A vector of vertex indices.
    // Each triple of indices defines one triangle.
    std::vector<uint32_t> indices;

    /*!
     * Vertex colors
     */
    std::vector<glm::vec3> colors;


    // A vector of vertex normals.
    std::vector<glm::vec3> normals;

    // A vector of vertex texture coordinates.
    std::vector<glm::vec2> textureCoordinates;
};

/*!
 *	A struct that contains all data for a geometry object on the GPU-side:
 *	Contains all the buffer handles for vertex and index buffers which
 *	can be used for an indexed-geometry draw call.
 */
struct Geometry {
    // A handle to a GPU buffer that contains vertex position data.
    VkBuffer positionsBuffer;

    // A handle to a GPU buffer that contains face indices data.
    VkBuffer indicesBuffer;

    // The total number of indices contained within the indicesBuffer.
    uint32_t numberOfIndices;

    // A handle to a GPU buffer that contains vertex normal data.
    VkBuffer colorsBuffer;


    // A handle to a GPU buffer that contains vertex normal data.
    VkBuffer normalsBuffer;

    // A handle to a GPU buffer that contains vertex texture coordinate data.
    VkBuffer textureCoordinatesBuffer;
};

/*!
 *	Creates a box geometry
 *	@param width		width of the box
 *	@param height		height of the box
 *	@param depth		depth of the box
 *	@return all box data
 */
GeometryData createBoxGeometry(float width, float height, float depth);

/*!
 *	Creates a cornell box geometry with vertex color attribute
 *	@param width		width of the box
 *	@param height		height of the box
 *	@param depth		depth of the box
 *	@return all box data
 */
GeometryData createCornellBoxGeometry(float width, float height, float depth);
/*!
 *	Creates a cylinder geometry
 *	@param segments		number of segments of the cylinder
 *	@param height		height of the cylinder
 *	@param radius		radius of the cylinder
 *	@return all cylinder data
 */
GeometryData createCylinderGeometry(uint32_t segments, float height, float radius);

/*!
 *	Creates a cylinder geometry along a bezier curve
 *	@param segments		number of segments of the cylinder
 *	@param controlPoints    control points of the bezier curve
 *  @param bezierSegments   number of segments of the bezier curve
 *	@param radius		radius of the cylinder
 *	@return all cylinder data
 */
GeometryData createBezierCylinderGeometry(unsigned int segments, std::vector<glm::vec3> controlPoints, unsigned int bezierSegments, float radius);
/*!
 *	Creates a sphere geometry
 *	@param longitude_segments	number of longitude segments of the sphere
 *	@param latitude_segments	number of latitude segments of the sphere
 *	@param radius				radius of the sphere
 *	@return all sphere data
 */
GeometryData createSphereGeometry(uint32_t longitude_segments, uint32_t latitude_segments, float radius);


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

