/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
#pragma once

#include "INIReader.h"
// #include "../../external/VulkanLaunchpad/VulkanLaunchpad.h"
// #include "VulkanLaunchpad/VulkanLaunchpad.h"
// #include <VulkanLaunchpad/VulkanLaunchpad.h>
// #include <VulkanLaunchpad.h>
#include <iostream>
#include <memory>
#include <vector>

/*!
 * Initializes the framework
 * Do not overwrite this function!
 */
bool gcgInitFramework(
    VkInstance vk_instance,
    VkSurfaceKHR vk_surface,
    VkPhysicalDevice vk_physical_device,
    VkDevice vk_device,
    VkQueue vk_queue,
    const VklSwapchainConfig& swapchain_config
);

/*!
 *  Destroys the framework
 * Do not overwrite this function!
 */
void gcgDestroyFramework();

/*!
 * Loads the path of the specified shader from the file system.
 */
std::string gcgLoadShaderFilePath(const std::string& filePath);

/*!
 * Draws a teapot in red with a default pipeline
 */
void gcgDrawTeapot();

/*!
 *	Draws a teapot using the passed pipeline.
 *
 *	@param	pipeline	A valid VkPipeline handle to a graphics pipeline.
 */
void gcgDrawTeapot(VkPipeline pipeline);

/*!
 *	Draws a teapot using the passed pipeline. Before the draw call is recorded,
 *	the given descriptor set is bound.
 *
 *	@param	pipeline			A valid VkPipeline handle to a graphics pipeline
 *								which will be used to draw the teapot model.
 *	@param	descriptor_set		A valid VkDescriptorSet handle which will be bound
 *								BEFORE drawing the teapot model with the given pipeline.
 */
void gcgDrawTeapot(VkPipeline pipeline, VkDescriptorSet descriptor_set);

/*!
 *	Creates a perspective projection matrix which transforms a part of the scene into a unit cube based on the given parameters.
 *	The scene is assumed to be given in a right-handed coordinate system with its y axis pointing upwards.
 *	The part of the scene that will end up within the unit cube is located towards the negative z axis.
 *	@param	field_of_view			The perspective projection's full field of view in radians.
 *	@param	aspect_ratio			The ratio of the screen's width to its height.
 *	@param	near_plane_distance		The distance from the camera's origin to the near plane.
 *	@param	far_plane_distance		The distance from the camera's origin to the far plane.
 *	@return	A perspective projection matrix based on the given parameters.
 */
glm::mat4 gcgCreatePerspectiveProjectionMatrix(float field_of_view, float aspect_ratio, float near_plane_distance, float far_plane_distance);

/*!
 *	Gets the vertex buffer (has VK_BUFFER_USAGE_VERTEX_BUFFER_BIT as usage flags)
 *	which contains the vertex positions of the teapot geometry.
 *	It contains contiguously stored vertex positions, each one stored as 3x float (x, y, z).
 */
VkBuffer gcgGetTeapotPositionsBuffer();

/*!
 *	Gets the index buffer (has VK_BUFFER_USAGE_INDEX_BUFFER_BIT as usage flags)
 *	which contains the face indices of the teapot geometry, which refer to
 *	the vertex positions within the gcgGetTeapotPositionsBuffer().
 *	It contains contiguously stored indices, each one stored as uint32_t.
 */
VkBuffer gcgGetTeapotIndicesBuffer();

/*!
 *	Gets the number of indices stored within the buffer returned by
 *	gcgGetTeapotIndicesBuffer(). Together with gcgGetTeapotPositionsBuffer()
 *	and gcgGetTeapotIndicesBuffer(), this number can be used for an indexed
 *	draw call via vkCmdDrawIndexed(...).
 */
uint32_t gcgGetNumTeapotIndices();

/*!
 *   Transfers a swapchain image to host coherent memory and saves it in the ppm format.
 *   @param  filename          The filename without .ppm format ending
 *   @param  swapchain_image   The current swapchain image retrieved via swapchain_image_handles[vklGetCurrentSwapChainImageIndex()]
 *   @param  width             The width of the swapchain image = window_width
 *   @param  height            The height of the swapchain image = window_height
 *   @param  format            The imageFormat used when creating the swapchain = surface_format.format
 *   @param  device            The device handle to be used for the host coherent image and memory creation.
 *   @param  physical_device   The physical device where to create the host coherent image and memory.
 *   @param  queue             The command queue to be used in submitting a new command buffer.
 *   @param  command_pool      The command pool to be used in creating a new command buffer.
 */
void gcgSaveScreenshot(
    std::string filename,
    VkImage swapchain_image,
    int width,
    int height,
    VkFormat format,
    VkDevice device,
    VkPhysicalDevice physical_device,
    VkQueue queue,
    uint32_t queue_family_index
);

struct CMDLineArgs {
    bool run_headless = false;
    bool init_camera = false;
    bool init_renderer = false;
    bool set_filename = false;

    std::string filename = "";
    std::string init_camera_filepath = "";
    std::string init_renderer_filepath = "";
};

void gcgParseArgs(CMDLineArgs& args, int argc, char** argv);
