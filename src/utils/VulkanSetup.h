#pragma once

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>
#include <VulkanLaunchpad.h>
#include <optional>
#include <string>
#include <vector>

#include "../Camera/Camera.h"

/*!
 *	Everything recreateSwapchainAndDependents needs that stays fixed for the whole render loop —
 *	built once before the loop starts.
 */
struct SwapchainRecreateContext {
    GLFWwindow* window;
    VkInstance vk_instance;
    VkPhysicalDevice vk_physical_device;
    VkDevice vk_device;
    VkQueue vk_queue;
    uint32_t selected_queue_family_index;
    VkSurfaceKHR vk_surface;
    VkSurfaceFormatKHR surface_format;
    bool depthtest;
    VkClearValue color_clear_value;
    VkClearValue depth_clear_value;
};

/*!
 *	Everything createSwapchain produces that main() still needs afterward, beyond the swapchain
 *	handle itself.
 */
struct SwapchainSetup {
    VkSwapchainKHR swapchain;
    VkSurfaceFormatKHR surface_format;
    VkSurfaceCapabilitiesKHR surface_capabilities;
    VkImageUsageFlags image_usage;
    std::vector<VkImage> image_handles;
};

/*!
 *	Add extension_name to the target vector ref_vector if the extension is supported on this system.
 *	@param	extension_name		The instance extension that shall be added to ref_vector
 *	@param	ref_vector			Reference to the vector that the instance extension shall be added to, if supported
 */
void addInstanceExtensionToVectorIfSupported(const char* extension_name, std::vector<const char*>& ref_vector);

/*!
 *	Determine the Vulkan instance extensions that are required by GLFW and Vulkan Launchpad.
 *	Required extensions from both sources are combined into one single vector (i.e., in
 *	contiguous memory, so that they can easily be passed to:
 *  VkInstanceCreateInfo::enabledExtensionCount and to VkInstanceCreateInfo::ppEnabledExtensionNames.
 *	@return     A std::vector of const char* elements, containing all required instance extensions.
 *	@example    std::vector<const char*> extensions = getRequiredInstanceExtensions();
 *	            VkInstanceCreateInfo create_info    = {};
 *	            create_info.enabledExtensionCount   = extensions.size();
 *	            create_info.ppEnabledExtensionNames = extensions.data();
 */
std::vector<const char*> getRequiredInstanceExtensions();

/*!
 *	Add validation_layer_name to the target vector ref_vector if the validation layer is supported on this system:
 *	@param	validation_layer_name	The validation layer name that shall be added to ref_vector
 *	@param	ref_vector				Reference to the vector that the validation layer name shall be added to, if supported
 */
void addValidationLayerNameToVectorIfSupported(const char* validation_layer_name, std::vector<const char*>& ref_vector);

/*!
 *	Creates the VkInstance: application info, required extensions (getRequiredInstanceExtensions),
 *	and the validation layer if supported.
 *	@return		A valid VkInstance handle.
 */
VkInstance createVulkanInstance();

/*!
 *	Creates the VkSurfaceKHR for the given window.
 *	@param	vk_instance		A valid VkInstance handle to create the surface from.
 *	@param	window			The GLFW window to create the surface for.
 *	@return		A valid VkSurfaceKHR handle.
 */
VkSurfaceKHR createWindowSurface(VkInstance vk_instance, GLFWwindow* window);

/*!
 *	From the given list of physical devices, select the first one that satisfies all requirements.
 *	@param		physical_devices		A pointer which points to contiguous memory of #physical_device_count sequentially
                                        stored VkPhysicalDevice handles is expected. The handles can (or should) be those
 *										that are returned from vkEnumeratePhysicalDevices.
 *	@param		physical_device_count	The number of consecutive physical device handles there are at the memory location
 *										that is pointed to by the physical_devices parameter.
 *	@param		surface					A valid VkSurfaceKHR handle, which is used to determine if a certain
 *										physical device supports presenting images to the given surface.
 *	@return		The index of the physical device that satisfies all requirements is returned.
 */
uint32_t selectPhysicalDeviceIndex(const VkPhysicalDevice* physical_devices, uint32_t physical_device_count, VkSurfaceKHR surface);

/*!
 *	From the given list of physical devices, select the first one that satisfies all requirements.
 *	@param		physical_devices	A vector containing all available VkPhysicalDevice handles, like those
 *									that are returned from vkEnumeratePhysicalDevices.
 *	@param		surface				A valid VkSurfaceKHR handle, which is used to determine if a certain
 *									physical device supports presenting images to the given surface.
 *	@return		The index of the physical device that satisfies all requirements is returned.
 */
uint32_t selectPhysicalDeviceIndex(const std::vector<VkPhysicalDevice>& physical_devices, VkSurfaceKHR surface);

/*!
 *	Enumerates the system's physical devices and selects one via selectPhysicalDeviceIndex.
 *	@param	vk_instance		A valid VkInstance handle to enumerate physical devices from.
 *	@param	vk_surface		A valid VkSurfaceKHR handle, used to check presentation support.
 *	@return		A valid VkPhysicalDevice handle.
 */
VkPhysicalDevice pickPhysicalDevice(VkInstance vk_instance, VkSurfaceKHR vk_surface);

/*!
 *	Based on the given physical device and the surface, select a queue family which supports both,
 *	graphics and presentation to the given surface. Return the INDEX of an appropriate queue family!
 *	@param	physical_device		A valid VkPhysicalDevice handle to search the queue families of.
 *	@param	surface				A valid VkSurfaceKHR handle, used to check presentation support.
 *	@return		The index of a queue family which supports the required features shall be returned.
 */
uint32_t selectQueueFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Builds the VkDeviceQueueCreateInfo for the given (already-selected, see selectQueueFamilyIndex)
 *	queue family, after sanity-checking that index against vkGetPhysicalDeviceQueueFamilyProperties.
 *	@param	vk_physical_device				A valid VkPhysicalDevice handle to sanity-check the queue family index against.
 *	@param	selected_queue_family_index	The index of the queue family to build the create info for.
 *	@return		A VkDeviceQueueCreateInfo requesting one queue from the given queue family.
 */
VkDeviceQueueCreateInfo createQueueCreateInfo(VkPhysicalDevice vk_physical_device, uint32_t selected_queue_family_index);

/*!
 *	Add extension_name to the target vector ref_vector if the extension is supported by the given physical device.
 *	@param	extension_name		The device extension that shall be added to ref_vector
 *	@param	physical_device		The physical device handle which must support the given extension
 *	@param	ref_vector			Reference to the vector that the device extension shall be added to, if supported
 */
void addDeviceExtensionToVectorIfSupported(const char* extension_name, VkPhysicalDevice physical_device, std::vector<const char*>& ref_vector);

/*!
 *	Creates the logical device (with the extensions/features this app needs) and retrieves its queue.
 *	@param	vk_physical_device				A valid VkPhysicalDevice handle to create the logical device from.
 *	@param	device_queue_create_info		The queue create info (see createQueueCreateInfo) to request the queue with.
 *	@param	selected_queue_family_index	The index of the queue family the requested queue belongs to.
 *	@param	out_vk_queue					Receives the handle of the retrieved queue.
 *	@return		A valid VkDevice handle.
 */
VkDevice createLogicalDeviceAndQueue(
    VkPhysicalDevice vk_physical_device,
    const VkDeviceQueueCreateInfo& device_queue_create_info,
    uint32_t selected_queue_family_index,
    VkQueue& out_vk_queue
);

/*!
 *	Based on the given physical device and the surface, a supported surface image format
 *	which can be used for the framebuffer's attachment formats is searched and returned.
 *	@param	physical_device		A valid VkPhysicalDevice handle to query supported surface formats of.
 *	@param	surface				A valid VkSurfaceKHR handle to query supported surface formats for.
 *	@return		A supported format is returned.
 */
VkSurfaceFormatKHR getSurfaceImageFormat(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Based on the given physical device and the surface, the physical device's surface capabilities are read and returned.
 *	@param	physical_device		A valid VkPhysicalDevice handle to query surface capabilities of.
 *	@param	surface				A valid VkSurfaceKHR handle to query surface capabilities for.
 *	@return		VkSurfaceCapabilitiesKHR data
 */
VkSurfaceCapabilitiesKHR getPhysicalDeviceSurfaceCapabilities(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Creates the swapchain sized to the surface's actual capabilities (clamping/overwriting
 *	window_width/window_height to match), and retrieves its images.
 *	@param	vk_physical_device				A valid VkPhysicalDevice handle to query surface support from.
 *	@param	vk_surface						A valid VkSurfaceKHR handle to create the swapchain for.
 *	@param	vk_device						A valid VkDevice handle to create the swapchain on.
 *	@param	selected_queue_family_index	The queue family index the swapchain images will be used with.
 *	@param	window_width					Requested width in, actual (clamped) swapchain width out.
 *	@param	window_height					Requested height in, actual (clamped) swapchain height out.
 *	@return		A SwapchainSetup with the new swapchain and its images.
 */
SwapchainSetup createSwapchain(
    VkPhysicalDevice vk_physical_device,
    VkSurfaceKHR vk_surface,
    VkDevice vk_device,
    uint32_t selected_queue_family_index,
    int& window_width,
    int& window_height
);

/*!
 *	Creates the depth buffer image used by every pipeline with depth test/write enabled.
 *	@param	vk_physical_device		A valid VkPhysicalDevice handle to allocate the depth buffer's memory from.
 *	@param	vk_device				A valid VkDevice handle to create the depth buffer image on.
 *	@param	width					The depth buffer's width, in pixels.
 *	@param	height					The depth buffer's height, in pixels.
 *	@return		A valid VkImage handle for the newly created depth buffer.
 */
VkImage createDepthBuffer(VkPhysicalDevice vk_physical_device, VkDevice vk_device, int width, int height);

/*!
 *	Builds the per-swapchain-image framebuffer composition the framework needs — one entry per
 *	image, each with a color attachment and (if depthtest) a shared depth attachment. Used both for
 *	the framework's initial setup and, identically, inside recreateSwapchainAndDependents.
 *	@param	vk_swapchain				The swapchain the framebuffers belong to.
 *	@param	swapchain_image_handles		The swapchain's color images, one framebuffer entry per image.
 *	@param	image_extent				The swapchain images' extent.
 *	@param	image_format				The swapchain images' format.
 *	@param	image_usage					The swapchain images' usage flags.
 *	@param	color_clear_value			The clear value to use for the color attachment.
 *	@param	depthtest					Whether a depth attachment shall be included.
 *	@param	depth_buffer				The depth buffer image to attach, if depthtest is true.
 *	@param	depth_clear_value			The clear value to use for the depth attachment, if depthtest is true.
 *	@return		A VklSwapchainConfig ready to pass to vklInitFramework/vklInitFrameworkVma.
 */
VklSwapchainConfig buildSwapchainConfig(
    VkSwapchainKHR vk_swapchain,
    const std::vector<VkImage>& swapchain_image_handles,
    VkExtent2D image_extent,
    VkFormat image_format,
    VkImageUsageFlags image_usage,
    VkClearValue color_clear_value,
    bool depthtest,
    VkImage depth_buffer,
    VkClearValue depth_clear_value
);

/*!
 *	Initializes Dear ImGui's GLFW and Vulkan backends against the already-initialized framework
 *	(needs vklGetRenderpass(), so must run after vklInitFramework()).
 *	@param	window							The GLFW window ImGui shall receive input from.
 *	@param	vk_instance						A valid VkInstance handle.
 *	@param	vk_physical_device				A valid VkPhysicalDevice handle.
 *	@param	vk_device						A valid VkDevice handle.
 *	@param	selected_queue_family_index	The queue family index ImGui's queue belongs to.
 *	@param	vk_queue						The queue ImGui shall submit its setup commands to.
 *	@param	min_image_count					The swapchain's minimum image count.
 *	@param	image_count						The swapchain's actual image count.
 */
void initImGui(
    GLFWwindow* window,
    VkInstance vk_instance,
    VkPhysicalDevice vk_physical_device,
    VkDevice vk_device,
    uint32_t selected_queue_family_index,
    VkQueue vk_queue,
    uint32_t min_image_count,
    uint32_t image_count
);

/*!
 *	Recreates the swapchain, depth buffer, and every pipeline against the window's current
 *	framebuffer size. Blocks (via glfwWaitEvents()) while the window is minimized. Takes mutable
 *	references to the Vulkan objects and cameras it needs to replace/update in place, since they
 *	live as locals in main().
 *	@param	ctx						Everything that stays fixed for the whole render loop (see SwapchainRecreateContext).
 *	@param	vk_swapchain			The swapchain handle to replace in place.
 *	@param	depth_buffer			The depth buffer image to replace in place.
 *	@param	swapchain_image_handles	The swapchain's color images, replaced in place.
 *	@param	window_width			Updated to the new swapchain width.
 *	@param	window_height			Updated to the new swapchain height.
 *	@param	trackballCamera			Receives the new aspect ratio.
 *	@param	flyCamera				Receives the new aspect ratio.
 */
void recreateSwapchainAndDependents(
    const SwapchainRecreateContext& ctx,
    VkSwapchainKHR& vk_swapchain,
    VkImage& depth_buffer,
    std::vector<VkImage>& swapchain_image_handles,
    int& window_width,
    int& window_height,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera
);

/*!
 *	Allocates a new descriptor set of the given layout from the given descriptor pool.
 *	It is not required to cleanup the returned descriptor set explicitly, it will be cleaned up when the descriptor pool is destroyed.
 *	@param	device					Valid handle to the logical device
 *	@param	descriptor_pool			Valid handle to a descriptor pool
 *	@param	descriptor_set_layout	Valid handle to a descriptor set layout
 *	@return		A valid VkDescriptorSet handle, allocated from descriptor_pool.
 */
VkDescriptorSet allocDescriptorSet(VkDevice device, VkDescriptorPool descriptor_pool, VkDescriptorSetLayout descriptor_set_layout);

/*!
 *	Writes the descriptor information to a given descriptor set which describes one uniform buffer at
 *	binding = 0 and another uniform buffer at binding = 1 — the two bindings every terrain material
 *	descriptor set always needs (its vertex-stage and fragment-stage uniform buffers).
 *	The given descriptor set must have been created from a descriptor set layout according to this structure.
 *	@param	device				Valid handle to the logical device
 *	@param	descriptor_set		Valid handle to a descriptor set, which concrete descriptor information will be written to.
 *	@param	vert_buffer		A descriptor for this uniform buffer will be written to binding = 0
 *	@param	frag_buffer		A descriptor for this uniform buffer will be written to binding = 1
 */
void writeDescriptorSet(VkDevice device, VkDescriptorSet descriptor_set, VkBuffer vert_buffer, VkBuffer frag_buffer);

/*!
 *	As the two-buffer overload above, plus a third, optional uniform buffer written to binding = 2 —
 *	for the terrain material, this is the directional light data.
 *	@param	device				Valid handle to the logical device
 *	@param	descriptor_set		Valid handle to a descriptor set, which concrete descriptor information will be written to.
 *	@param	vert_buffer		A descriptor for this uniform buffer will be written to binding = 0
 *	@param	frag_buffer		A descriptor for this uniform buffer will be written to binding = 1
 *	@param	directional_light_data		A descriptor for this uniform buffer will be written to binding = 2
 */
void writeDescriptorSet(VkDevice device, VkDescriptorSet descriptor_set, VkBuffer vert_buffer, VkBuffer frag_buffer, VkBuffer directional_light_data);
