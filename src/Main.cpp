/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
// vulkan/vulkan.h must come before GLFW/glfw3.h and VulkanLaunchpad.h below — neither of those
// headers includes it themselves, they just assume the includer already did.
#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>
#include <VulkanLaunchpad.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp> // for vk::OutOfDateKHRError, thrown by the framework's internal vulkan-hpp calls on resize

#include "Camera/Camera.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "terrain/ChunkCoord.h"
#include "terrain/ChunkManager.h"
#include "terrain/DiamondSquareGenerator.h"
#include "terrain/TerrainGeometry.h"
#include "terrain/WaterGeometry.h"
#include "utils/INIReader.h"
#include "utils/PathUtils.h"

#undef min
#undef max

/* ------------------------------------------------ */
// Constants
/* ------------------------------------------------ */
constexpr char WELCOME_MSG[] = ":::::: WELCOME TO MY TERRAIN GENERATOR ::::::";
constexpr char WINDOW_TITLE[] = "Fractal Terrain Generator";

constexpr size_t POLYMODES = 2;
constexpr size_t CULLMODES = 3;
constexpr VkPolygonMode kTerrainPolygonModes[POLYMODES] = {VK_POLYGON_MODE_FILL, VK_POLYGON_MODE_LINE};
constexpr VkCullModeFlags kTerrainCullModes[CULLMODES] = {VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT};

/*! Fixed width every GUI slider is drawn at, used by labelThenRightAlignedWidget. */
constexpr float kSliderWidth = 200.0f;

/* ------------------------------------------------ */
// Data Structures
/* ------------------------------------------------ */
/*!
 *	It matches the definition and sizes of the corresponding GPU-side struct exactly, which is used in shaders.
 */
struct UniformBufferVert {
    /*! Storage for the model matrix, consisting of 16 float values (inherently aligned to 16 bytes) */
    glm::mat4 modelMatrix;

    /*! Storage for the model matrix suitable for normals transformation, consisting of 16 float values (inherently aligned to 16 bytes) */
    glm::mat4 modelMatrixForNormals;

    /*! Storage for the view-projection matrix, consisting of 16 float values (inherently aligned to 16 bytes) */
    glm::mat4 viewProjMatrix;
};

/*!
 *	Per-chunk blend state for terrain.vert, pushed via vkCmdPushConstants immediately before each
 *	chunk's draw call.
 */
struct TerrainPushConstants {
    /*! 0-1 blend progress between this chunk's `from` and `to` geometry. */
    float blendFactor;

    /*! Whether this chunk is currently blending (`from` is valid). */
    uint32_t isBlending;
};

struct UniformBufferFrag {
    /*! Storage for the camera's world space position (aligned to 16 bytes) */
    glm::vec4 cameraPosition;

    /*! Illumination properties ka, kd, ks, alpha (in that order)
     *	First three are material coefficients, the last one is specular alpha. */
    glm::vec4 materialProperties;

    /*! Debug visualization toggles: x = drawNormals (N), y = highlightChunkBorders (F3). */
    glm::uvec2 debugToggles;

    /*! Whether the camera is below the water plane; drives the underwater tint in terrain.frag. */
    uint32_t isUnderwater;

    /*! World-space width of one terrain chunk, for the chunk-border-highlight distance check. */
    float chunkWidth;

    /*! Surface roughness in [0, 1]: 0 is sharp/shiny, 1 is broad/matte. See buildGUI's slider. */
    float roughness;

    /*! World-space heights (not scaled by heightScale) where the height-based color gradient
     *	transitions dirt->grass and grass->rock. See g_dirt_to_grass_height_offset/g_grass_to_rock_height_offset. */
    float dirtToGrassHeight;
    float grassToRockHeight;

    /*! Half-width of each color transition above, same units as dirtToGrassHeight. */
    float heightColorTransitionBand;
};

/*!
 *	Matches water.vert's uniform block exactly. No fragment-stage uniforms are needed since
 *	water.frag uses a fixed color.
 */
struct UniformBufferWaterVert {
    glm::mat4 modelMatrix;
    glm::mat4 viewProjMatrix;
};

/*!
 *	This struct contains the data of a directional light.
 *	It matches the definition and sizes of the corresponding GPU-side struct exactly, which is used in shaders.
 */
struct DirectionalLight {
    /*! Light color of this light source */
    glm::vec4 color;

    /*! Light direction of this directional light source */
    glm::vec4 direction;
};

/*!
 *	This struct contains the data of a point light.
 *	It matches the definition and sizes of the corresponding GPU-side struct exactly, which is used in shaders.
 */
struct PointLight {
    /*! Light color of this light source */
    glm::vec4 color;

    /*! Position of this light source in world space */
    glm::vec4 position;

    /*! Attenuation properties of this light source */
    glm::vec4 attenuation;
};

/*!
 *	Holds every GPU resource (pipelines, geometries, uniform buffers, descriptor sets, textures)
 *	that make up the terrain scene (Terrain + dir light).
 */
struct TerrainScene {
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    /*! Pipelines are built lazily, on first use of each (polygon mode, cull mode) combination. */
    VkPipeline pipelines[POLYMODES][CULLMODES];
    std::string vertexShaderPath;
    std::string fragmentShaderPath;
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;

    VkBuffer ub_dirlight;

    VkBuffer ub_terrain_vert;
    VkBuffer ub_terrain_frag;
    VkDescriptorSet ds_terrain;

    ChunkManager chunkManager;

    float heightScale;
    float waterLevel;
    float roughness;
};

/*!
 * A raycast hit result.
 */
struct Hit {
    glm::vec3 point;
    float distance;
};

/*!
 *	Holds every GPU resource that makes up the water plane: one pipeline/descriptor set/uniform
 *	buffer shared by all chunks, and one small baked quad per currently-loaded terrain chunk.
 */
struct WaterScene {
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    VkPipeline pipeline;
    std::string vertexShaderPath;
    std::string fragmentShaderPath;

    VkBuffer ub_water_vert;
    VkDescriptorSet ds_water;

    std::unordered_map<ChunkCoord, WaterChunkGeometry> chunkGeometry;
};

/*!
 *	Everything loadSettings() reads out of settings.ini that main() still needs afterward.
 */
struct AppSettings {
    int window_width;
    int window_height;
    std::string window_title;

    float field_of_view;
    float near_plane_distance;
    float far_plane_distance;
    glm::vec3 camera_position;
    float camera_yaw;
    float camera_pitch;

    bool depthtest;
    float background_r;
    float background_g;
    float background_b;

    float initial_hurst;
    uint32_t initial_seed;
    int initial_grid_size_exponent;
    float initial_height_scale;
    float initial_roughness;
    float initial_water_level;
};

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

/* --------------------------------------------- */
// Helper Function Declarations
/* --------------------------------------------- */
/*!
 *	Reads every startup default out of assets/settings/settings.ini: window/camera/renderer's
 *	one-time setup values (returned in an AppSettings) and the GUI sliders'/ChunkManager's/demo
 *	mode's starting values (written directly into their respective globals — see AppSettings).
 */
AppSettings loadSettings();

/*!
 *	This callback function gets invoked by GLFW whenever a GLFW error occured.
 */
void errorCallbackFromGlfw(int error, const char* description);

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
 *	@param	ref_vector				Reference to the vector that the validation layer name shall be added to, if
 *supported
 */
void addValidationLayerNameToVectorIfSupported(const char* validation_layer_name, std::vector<const char*>& ref_vector);

/*!
 *	Creates the VkInstance: application info, required extensions (getRequiredInstanceExtensions),
 *	and the validation layer if supported.
 */
VkInstance createVulkanInstance();

/*!
 *	Creates the VkSurfaceKHR for the given window.
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
 */
VkPhysicalDevice pickPhysicalDevice(VkInstance vk_instance, VkSurfaceKHR vk_surface);

/*!
 *	Based on the given physical device and the surface, select a queue family which supports both,
 *	graphics and presentation to the given surface. Return the INDEX of an appropriate queue family!
 *	@return		The index of a queue family which supports the required features shall be returned.
 */
uint32_t selectQueueFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Builds the VkDeviceQueueCreateInfo for the given (already-selected, see selectQueueFamilyIndex)
 *	queue family, after sanity-checking that index against vkGetPhysicalDeviceQueueFamilyProperties.
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
 *	Creates the logical device (with the extensions/features this app needs, including detecting
 *	Synchronization2 support — see g_synchronization2_supported) and retrieves its queue.
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
 *	@return		A supported format is returned.
 */
VkSurfaceFormatKHR getSurfaceImageFormat(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Based on the given physical device and the surface, a the physical device's surface capabilites are read and returned.
 *	@return		VkSurfaceCapabilitiesKHR data
 */
VkSurfaceCapabilitiesKHR getPhysicalDeviceSurfaceCapabilities(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Creates the swapchain sized to the surface's actual capabilities (clamping/overwriting
 *	window_width/window_height to match), and retrieves its images.
 */
SwapchainSetup createSwapchain(
    VkPhysicalDevice vk_physical_device,
    VkSurfaceKHR vk_surface,
    VkDevice vk_device,
    uint32_t selected_queue_family_index,
    int& window_width,
    int& window_height
);

std::optional<Hit> raycastTerrain(const glm::vec3& origin, const glm::vec3& direction);

/*!
 *	Creates the depth buffer image used by every pipeline with depth test/write enabled.
 */
VkImage createDepthBuffer(VkPhysicalDevice vk_physical_device, VkDevice vk_device, int width, int height);

/*!
 *	Builds the per-swapchain-image framebuffer composition the framework needs — one entry per
 *	image, each with a color attachment and (if depthtest) a shared depth attachment. Used both for
 *	the framework's initial setup and, identically, inside recreateSwapchainAndDependents.
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
 *	Builds (compiles + creates) the terrain pipeline for one (polygon mode, cull mode) combination,
 *	identified by their indices into kTerrainPolygonModes/kTerrainCullModes.
 */
VkPipeline buildTerrainPipeline(const TerrainScene& scene, size_t polygon_mode_index, size_t cull_mode_index);

/*!
 *	Allocates a new descriptor set of the given layout from the given descriptor pool.
 *	It is not required to cleanup the returned descriptor set explicitly, it will be cleaned up when the descriptor pool is destroyed.
 *	@param	device					Valid handle to the logical device
 *	@param	descriptor_pool			Valid handle to a descriptor pool
 *	@param	descriptor_set_layout	Valid handle to a descriptor set layout
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

/*!
 *	Creates the shared pipeline, uniform buffers, and descriptor set that every loaded chunk is
 *	drawn with, and configures the scene's ChunkManager from params. Chunk geometry itself is
 *	generated/uploaded on demand, driven by the camera — not by this one-time setup call.
 */
TerrainScene setupTerrainScene(
    VkDevice vk_device,
    VkQueue vk_queue,
    uint32_t selected_queue_family_index,
    TerrainParams& params,
    float initial_height_scale,
    float initial_roughness,
    float initial_water_level
);

/*!
 *	Builds a minimal ImGui panel shown while the initial chunks are generating, before the real
 *	terrain scene's own controls exist yet.
 */
void buildLoadingGUI(size_t pendingChunkCount);

/*!
 *	Blocks until every chunk in the initial (2 * viewRadius + 1)^2 window around cameraPos has been
 *	generated and uploaded, while keeping the window responsive (polling events and drawing a
 *	loading screen) for however long that takes. Use this once, at startup, before the main render
 *	loop begins — per-frame streaming (updateLoadedChunks) takes over from there.
 */
void generateTerrainGeometryWithLoadingScreen(VkDevice vk_device, ChunkManager& chunkManager, const glm::vec3& cameraPos);

/*!
 *	Builds the water plane's shared pipeline, uniform buffer, and descriptor set.
 */
WaterScene setupWaterScene(VkDevice vk_device, const TerrainParams& terrain_params);

/*!
 *	This callback function gets invoked by GLFW during glfwPollEvents() if there was
 *	mouse button input that can be processed by our application.
 */
void mouseButtonCallbackFromGlfw(GLFWwindow* glfw_window, int button, int action, int mods);

/*!
 *	This callback function gets invoked by GLFW during glfwPollEvents() if there was
 *	mouse scroll input that can be processed by our application.
 */
void scrollCallbackFromGlfw(GLFWwindow* glfw_window, double xoffset, double yoffset);

/*!
 *	This callback function gets invoked by GLFW during glfwPollEvents() whenever the window's
 *	framebuffer size changes (resize, maximize, restore, or a DPI change moving it to a different
 *	monitor). Only sets a flag — actually recreating the swapchain happens once per frame in the
 *	render loop, where the Vulkan objects it needs are in scope.
 */
void framebufferSizeCallbackFromGlfw(GLFWwindow* glfw_window, int width, int height);

/*!
 *	Function that is invoked by GLFW to handle key events like key presses or key releases.
 *	If the ESC key has been pressed, the window will be marked that it should close.
 */
void handleGlfwKeyCallback(GLFWwindow* glfw_window, int key, int scancode, int action, int mods);

/*!
 *	Recreates the swapchain, depth buffer, and every pipeline against the window's current
 *	framebuffer size. Blocks (via glfwWaitEvents()) while the window is minimized. Takes mutable
 *	references to the Vulkan objects and cameras it needs to replace/update in place, since they
 *	live as locals in main().
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
 *	Handles a pending camera-mode toggle (the `C` key, see g_toggle_camera_requested): swaps between
 *	fly and trackball camera, carrying position/orientation across, and re-syncs the mouse-delta
 *	baseline so the switch doesn't cause a sudden jump in look direction next frame. No-op if no
 *	toggle is pending.
 */
void handleCameraToggleRequest(
    GLFWwindow* window,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera,
    Camera*& activeCamera,
    double& mouse_x,
    double& mouse_y,
    double& mouse_x_last,
    double& mouse_y_last
);

/*!
 *	Draws a fresh uint32_t seed from a properly-seeded std::mt19937, spanning the full uint32_t range.
 */
uint32_t generateRandomSeed();
float generateRandomHurst();
float generateRandomHeightScale();
float generateRandomWaterLevel();

/*!
 *	Applies any pending reseed/Hurst-change/demo-mode change to the terrain scene for this frame:
 *	starts a regeneration when needed (guarded by isRegenerating(), same as the GUI's own guard),
 *	advances demo mode's height-scale/water-level drift, and applies a pending view-radius change.
 *	Does not itself stream chunks in/out — see updateLoadedChunks() for that.
 */
void updateTerrainState(TerrainScene& terrain_scene, double currentFrameTime);

/*!
 *	Creates a small world-space-baked water quad for any newly-loaded terrain chunk, and destroys
 *	it for any chunk that's since been unloaded. Call once per frame, right after
 *	updateLoadedChunks(), outside vklStartRecordingCommands()/vklEndRecordingCommands().
 */
void updateWaterChunks(VkDevice vk_device, WaterScene& scene, const ChunkManager& chunkManager);

/*!
 *	Applies this frame's mouse-look, shift-to-run, trackball strafe/zoom, and fly-mode WASD input to
 *	the active camera. Reads `mouse_x`/`mouse_y` (already updated by the caller for this frame) and
 *	updates `mouse_x_last`/`mouse_y_last` to them, ready for next frame's delta.
 */
void applyCameraInput(
    GLFWwindow* window,
    Camera* activeCamera,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera,
    double mouse_x,
    double mouse_y,
    double& mouse_x_last,
    double& mouse_y_last,
    float dt
);

/*!
 *	Prints `label`, then positions the cursor so the widget that follows always ends flush with the
 *	right edge, `widget_width` wide, regardless of the label's own width — instead of its position (or,
 *	for non-input widgets like Button that ignore SetNextItemWidth, its right edge) drifting depending
 *	on how long each row's label happens to be. Pass kSliderWidth for sliders/inputs; for a Button, pass
 *	its own natural size (ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2.0f) so the
 *	button's actual right edge — not just where a kSliderWidth-wide widget would have started — lands at
 *	the true right edge.
 */
void labelThenRightAlignedWidget(const char* label, float widget_width);

/*!
 * Builds the ImGUI Sidebar
 */
void buildGUI(TerrainScene& scene, const glm::vec3& cameraPosition, const glm::vec3& cameraForward);

/*!
 *	Bind the given descriptor set to use the material it represents for subsequent draw calls
 *	with the given pipeline, and render the given chunk (using its vertex and index buffers).
 *	Record everything into the current command buffer as provided by the framework.
 *	@param	pipeline		Valid handle to a given pipeline which shall be used for drawing.
 *	@param	chunk			The loaded chunk to draw. Which of its two vertex buffers plays the
 *							"from" role is derived internally from `chunk.from`'s validity.
 *	@param	indices_buffer		The shared index buffer — see ChunkManager::sharedIndicesBuffer.
 *	@param	number_of_indices	How many indices to draw from indices_buffer.
 *	@param	material		Valid handle to a descriptor set that refers to resources that contain material properties.
 *	@param	num_instances	How many instances to draw of the given geometry. Default = one single instance.
 */
void drawGeometryWithMaterial(
    VkPipeline pipeline,
    const LoadedChunk& chunk,
    VkBuffer indices_buffer,
    uint32_t number_of_indices,
    VkDescriptorSet material,
    uint32_t num_instances = 1u
);

/*!
 *	Updates the terrain scene's uniform buffers based on the current camera, and records draw calls
 *	for it into the currently recording command buffer. Must be called between
 *	vklStartRecordingCommands() and vklEndRecordingCommands().
 */
void updateAndDrawTerrainScene(VkDevice vk_device, TerrainScene& scene, const Camera* camera, double currentTime);

/*!
 *	Updates the water plane's shared uniform buffer and records one draw call per loaded chunk's
 *	water geometry. Must be called between vklStartRecordingCommands() and vklEndRecordingCommands(),
 *	after the terrain has been drawn.
 */
void updateAndDrawWaterScene(WaterScene& scene, const TerrainScene& terrain_scene, const Camera* camera);

/*!
 *	Destroys all GPU resources owned by the given terrain scene.
 */
void cleanupTerrainScene(VkDevice vk_device, TerrainScene& scene);

/*!
 *	Destroys all GPU resources owned by the given water scene.
 */
void cleanupWaterScene(VkDevice vk_device, WaterScene& scene);

/* ------------------------------------------------ */
// Global State
/* ------------------------------------------------ */

// --- GUI chrome ---
static ImGuiSliderFlags flags = ImGuiSliderFlags_None;
ImGuiWindowFlags window_flags = 0;
bool g_panel_open = true;
static bool g_framebuffer_resized = false;

// --- Camera / input ---
static bool g_dragging = false;
static bool g_strafing = false;
static float g_scroll_delta = 0.0f;
static bool g_toggle_camera = false;
static bool g_toggle_camera_requested = false;
static float g_camera_speed = 5.0f;
// Input feel, one-time startup values (settings.ini's [camera] section).
static float g_mouse_sensitivity = 0.005f;
static float g_scroll_sensitivity = 0.5f;

// --- Rendering toggles ---
/*!
 *	0 ... fill polygons
 *	1 ... wireframe mode
 */
static int g_polygon_mode_index = 0;

/*!
 *	0 ... no culling
 *	1 ... cull back faces
 *	2 ... cull front faces
 */
static int g_culling_index = 0;

static bool g_draw_normals = false;
static bool g_highlight_chunk_borders = false;

// --- Terrain GUI sliders (live-adjustable; see buildGUI) ---
static float g_hurst = 0.8f;
static bool g_hurst_changed = false;
static bool g_reseed_requested = false;
static int g_chunk_view_radius = 16;
static bool g_chunk_view_radius_changed = false;

// --- Demo mode (F4): keeps re-randomizing Hurst/seed, one right after the previous blend
// settles, so the terrain keeps morphing on its own while flying around for a recording.
static bool g_demo_mode = false;
// Height-scale drift: negative is a sentinel meaning "not started yet".
static double g_demo_height_interp_start_time = -1.0;
static float g_demo_height_start = 1.0f;
static float g_demo_height_target = 1.0f;
// Water-level drift: same pattern as height-scale above, its own independent timer.
static double g_demo_water_interp_start_time = -1.0;
static float g_demo_water_start = 0.0f;
static float g_demo_water_target = 0.0f;
// How long the drifts above take to reach each newly picked target.
static double g_demo_height_interp_duration = 1.0;
static double g_demo_water_interp_duration = 1.0;

// --- Directional light, terrain material, and color-gradient/water-offset settings — all
// one-time startup values (settings.ini's [terrain]/[light] sections), not GUI-adjustable.
static glm::vec3 g_dirlight_color = glm::vec3(0.85f, 0.85f, 0.85f);
static glm::vec3 g_dirlight_dir = glm::vec3(0.0f, 1.0f, -1.0f);
static float g_terrain_ka = 0.1f;
static float g_terrain_kd = 0.9f;
static float g_terrain_ks = 0.3f;
static float g_terrain_alpha = 10.0f;
static float g_dirt_to_grass_height_offset = 3.0f;
static float g_grass_to_rock_height_offset = 10.0f;
static float g_height_color_transition_band = 2.0f;
static float g_water_depth_bias = 0.05f;

// --- ChunkManager tuning, also one-time startup values (see ChunkManager.h for what each
// controls).
static double g_blend_duration = 2.0;
static int g_max_destroys_per_frame = 16;
static int g_max_uploads_per_frame = 8;

// --- Vulkan/sync internals ---
/*!
 *	A flag that will be set during initialization code.
 *	If set to true, it will indicate that Synchronization2 is supported and can be used.
 */
static bool g_synchronization2_supported = false;

/*!
 *	Function pointer to the implementation of vkCmdPipelineBarrier2KHR() which is not
 *	loaded by default since it is an extension. We have to get the pointer manually,
 *	and we do that only in the case where Synchronization2 is supported.
 */
PFN_vkCmdPipelineBarrier2KHR g_vkCmdPipelineBarrier2KHR;

/* ------------------------------------------------ */
// Main
/* ------------------------------------------------ */

int main() {
    VKL_LOG(WELCOME_MSG);

    AppSettings settings = loadSettings();
    VKL_LOG("Settings loaded from assets/settings/settings.ini.");

    int window_width = settings.window_width;
    int window_height = settings.window_height;

    // Install a callback function, which gets invoked whenever a GLFW error occurred.
    glfwSetErrorCallback(errorCallbackFromGlfw);

    if (!glfwInit()) {
        VKL_EXIT_WITH_ERROR("Failed to init GLFW");
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    int monitor_x, monitor_y;
    glfwGetMonitorWorkarea(monitor, &monitor_x, &monitor_y, &window_width, &window_height);

    // Set some window settings before creating the window:
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // No need to create a graphics context for Vulkan
    // Made resizable later, via glfwSetWindowAttrib, once the main render loop is ready for it.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = nullptr;
    window = glfwCreateWindow(window_width, window_height, settings.window_title.c_str(), nullptr, nullptr);

    if (!window) {
        VKL_EXIT_WITH_ERROR("No GLFW window created.");
    }
    glfwSetWindowPos(window, monitor_x, monitor_y);
    VKL_LOG("Window created.");

    VkInstance vk_instance = createVulkanInstance();
    VKL_LOG("Vulkan instance created.");

    VkSurfaceKHR vk_surface = createWindowSurface(vk_instance, window);
    VKL_LOG("Window surface created.");

    VkPhysicalDevice vk_physical_device = pickPhysicalDevice(vk_instance, vk_surface);
    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(vk_physical_device, &physical_device_properties);
    VKL_LOG("Physical device selected: " << physical_device_properties.deviceName);

    uint32_t selected_queue_family_index = selectQueueFamilyIndex(vk_physical_device, vk_surface);
    VkDeviceQueueCreateInfo device_queue_create_info = createQueueCreateInfo(vk_physical_device, selected_queue_family_index);
    VKL_LOG("Queue family " << selected_queue_family_index << " selected.");

    VkQueue vk_queue = VK_NULL_HANDLE;
    VkDevice vk_device = createLogicalDeviceAndQueue(vk_physical_device, device_queue_create_info, selected_queue_family_index, vk_queue);
    VKL_LOG("Logical device and queue created.");

    SwapchainSetup swapchain_setup =
        createSwapchain(vk_physical_device, vk_surface, vk_device, selected_queue_family_index, window_width, window_height);
    VkSwapchainKHR vk_swapchain = swapchain_setup.swapchain;
    VkSurfaceFormatKHR surface_format = swapchain_setup.surface_format;
    std::vector<VkImage>& swapchain_image_handles = swapchain_setup.image_handles;
    float aspect_ratio = static_cast<float>(window_width) / static_cast<float>(window_height);
    VKL_LOG("Swapchain created with " << swapchain_image_handles.size() << " images.");

    // Create a camera helper object, positioned/oriented per settings.ini. The trackball camera's
    // target is derived by raycasting the configured position/direction against the terrain,
    // falling back to the origin if that ray doesn't hit the terrain (e.g. looking up).
    FlyCamera flyCamera(settings.field_of_view, aspect_ratio, settings.near_plane_distance, settings.far_plane_distance);
    flyCamera.translate(settings.camera_position);
    flyCamera.rotate(glm::radians(settings.camera_yaw), glm::radians(settings.camera_pitch));
    auto initial_hit = raycastTerrain(flyCamera.getPosition(), flyCamera.getForward());
    TrackballCamera trackballCamera(flyCamera, initial_hit.has_value() ? initial_hit->point : glm::vec3(0.0f, 0.0f, 0.0f));
    Camera* activeCamera = &trackballCamera;
    VKL_LOG("Cameras initialized (starting in trackball mode).");

    VkImage depth_buffer = createDepthBuffer(vk_physical_device, vk_device, window_width, window_height);

    VkClearValue depth_clear_value;
    depth_clear_value.depthStencil.depth = 1.0f;
    depth_clear_value.depthStencil.stencil = 0u;
    VKL_LOG("Depth buffer created (" << window_width << "x" << window_height << ").");

    VkClearValue color_clear_value;
    color_clear_value.color.float32[0] = settings.background_r;
    color_clear_value.color.float32[1] = settings.background_g;
    color_clear_value.color.float32[2] = settings.background_b;
    color_clear_value.color.float32[3] = 1.0f;

    VklSwapchainConfig swapchain_config = buildSwapchainConfig(
        vk_swapchain,
        swapchain_image_handles,
        VkExtent2D{static_cast<uint32_t>(window_width), static_cast<uint32_t>(window_height)},
        surface_format.format,
        swapchain_setup.image_usage,
        color_clear_value,
        settings.depthtest,
        depth_buffer,
        depth_clear_value
    );

    // Init the framework:
    if (!vklInitFramework(vk_instance, vk_surface, vk_physical_device, vk_device, vk_queue, swapchain_config)) {
        VKL_EXIT_WITH_ERROR("Failed to init framework");
    }
    VKL_LOG("Vulkan framework initialized.");

    initImGui(
        window,
        vk_instance,
        vk_physical_device,
        vk_device,
        selected_queue_family_index,
        vk_queue,
        swapchain_setup.surface_capabilities.minImageCount,
        static_cast<uint32_t>(swapchain_image_handles.size())
    );
    VKL_LOG("Dear ImGui initialized.");

    TerrainParams initial_terrain_params;
    initial_terrain_params.hurst = settings.initial_hurst;
    initial_terrain_params.seed = settings.initial_seed;
    initial_terrain_params.gridSizeExponent = settings.initial_grid_size_exponent;
    TerrainScene terrain_scene = setupTerrainScene(
        vk_device,
        vk_queue,
        selected_queue_family_index,
        initial_terrain_params,
        settings.initial_height_scale,
        settings.initial_roughness,
        settings.initial_water_level
    );
    VKL_LOG("Terrain scene set up (hurst=" << initial_terrain_params.hurst << ", seed=" << initial_terrain_params.seed << ").");

    generateTerrainGeometryWithLoadingScreen(vk_device, terrain_scene.chunkManager, activeCamera->getPosition());
    VKL_LOG("Initial terrain chunks loaded (" << terrain_scene.chunkManager.loadedChunks.size() << " chunks).");

    WaterScene water_scene = setupWaterScene(vk_device, initial_terrain_params);
    VKL_LOG("Water scene set up.");

    glfwSetMouseButtonCallback(window, mouseButtonCallbackFromGlfw);

    glfwSetScrollCallback(window, scrollCallbackFromGlfw);

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallbackFromGlfw);

    glfwSetKeyCallback(window, handleGlfwKeyCallback);

    double mouse_x, mouse_x_last, mouse_y, mouse_y_last;
    glfwGetCursorPos(window, &mouse_x_last, &mouse_y_last);
    double lastFrameTime = glfwGetTime();

    vklEnablePipelineHotReloading(window, GLFW_KEY_F5);

    glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_TRUE);

    // Everything recreateSwapchainAndDependents needs that stays fixed for the whole loop below
    SwapchainRecreateContext swapchain_recreate_ctx{
        window,
        vk_instance,
        vk_physical_device,
        vk_device,
        vk_queue,
        selected_queue_family_index,
        vk_surface,
        surface_format,
        settings.depthtest,
        color_clear_value,
        depth_clear_value
    };

    VKL_LOG("Entering render loop.");
    while (!glfwWindowShouldClose(window)) {
        double currentFrameTime = glfwGetTime();
        float dt = static_cast<float>(currentFrameTime - lastFrameTime);
        lastFrameTime = currentFrameTime;

        // Handle user input:
        glfwPollEvents();

        int current_fb_width, current_fb_height;
        glfwGetFramebufferSize(window, &current_fb_width, &current_fb_height);
        if (g_framebuffer_resized || current_fb_width != window_width || current_fb_height != window_height) {
            g_framebuffer_resized = false;
            recreateSwapchainAndDependents(
                swapchain_recreate_ctx,
                vk_swapchain,
                depth_buffer,
                swapchain_image_handles,
                window_width,
                window_height,
                trackballCamera,
                flyCamera
            );
        }

        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        handleCameraToggleRequest(window, trackballCamera, flyCamera, activeCamera, mouse_x, mouse_y, mouse_x_last, mouse_y_last);

        updateTerrainState(terrain_scene, currentFrameTime);
        updateLoadedChunks(terrain_scene.chunkManager, activeCamera->getPosition(), currentFrameTime);
        updateWaterChunks(vk_device, water_scene, terrain_scene.chunkManager);

        applyCameraInput(window, activeCamera, trackballCamera, flyCamera, mouse_x, mouse_y, mouse_x_last, mouse_y_last, dt);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        buildGUI(terrain_scene, activeCamera->getPosition(), activeCamera->getForward());
        ImGui::Render();

        // Wait until we get an image from the swapchain to render into:
        try {
            vklWaitForNextSwapchainImage();
        } catch (const vk::OutOfDateKHRError&) {
            recreateSwapchainAndDependents(
                swapchain_recreate_ctx,
                vk_swapchain,
                depth_buffer,
                swapchain_image_handles,
                window_width,
                window_height,
                trackballCamera,
                flyCamera
            );
            continue;
        }
        vklStartRecordingCommands();

        updateAndDrawTerrainScene(vk_device, terrain_scene, activeCamera, currentFrameTime);
        updateAndDrawWaterScene(water_scene, terrain_scene, activeCamera);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vklGetCurrentCommandBuffer());

        vklEndRecordingCommands();

        // Present rendered image to the screen:
        try {
            vklPresentCurrentSwapchainImage();
        } catch (const vk::OutOfDateKHRError&) {
            recreateSwapchainAndDependents(
                swapchain_recreate_ctx,
                vk_swapchain,
                depth_buffer,
                swapchain_image_handles,
                window_width,
                window_height,
                trackballCamera,
                flyCamera
            );
            continue;
        }
    }
    VKL_LOG("Render loop exited, shutting down.");

    // Wait for all GPU work to finish before cleaning up:
    vkDeviceWaitIdle(vk_device);

    // Cleanup:
    vklDestroyDeviceLocalImageAndItsBackingMemory(depth_buffer);
    cleanupTerrainScene(vk_device, terrain_scene);
    cleanupWaterScene(vk_device, water_scene);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vklDestroyFramework();
    vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);
    vkDestroyDevice(vk_device, nullptr);
    vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
    vkDestroyInstance(vk_instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();

    VKL_LOG("Shutdown complete.");
    return EXIT_SUCCESS;
}

/* --------------------------------------------- */
// Helper Function Definitions
/* --------------------------------------------- */

AppSettings loadSettings() {
    INIReader settings_reader("assets/settings/settings.ini");

    AppSettings settings{};
    settings.window_width = 800;
    settings.window_height = 800;
    settings.window_title = settings_reader.Get("window", "title", WINDOW_TITLE);

    settings.field_of_view = static_cast<float>(settings_reader.GetReal("camera", "fov", 60.0f));
    settings.near_plane_distance = static_cast<float>(settings_reader.GetReal("camera", "near", 0.1f));
    settings.far_plane_distance = static_cast<float>(settings_reader.GetReal("camera", "far", 100.0f));
    settings.camera_position = glm::vec3(
        static_cast<float>(settings_reader.GetReal("camera", "position_x", 0.0f)),
        static_cast<float>(settings_reader.GetReal("camera", "position_y", 0.0f)),
        static_cast<float>(settings_reader.GetReal("camera", "position_z", 0.0f))
    );
    // Same convention as the live fly-camera controls: yaw/pitch = 0 looks along +X; positive
    // pitch looks up, negative looks down.
    settings.camera_yaw = static_cast<float>(settings_reader.GetReal("camera", "yaw", 0.0f));
    settings.camera_pitch = static_cast<float>(settings_reader.GetReal("camera", "pitch", 0.0f));

    bool as_wireframe = settings_reader.GetBoolean("renderer", "wireframe", false);
    if (as_wireframe) {
        g_polygon_mode_index = 1;
    }
    bool with_backface_culling = settings_reader.GetBoolean("renderer", "backface_culling", false);
    if (with_backface_culling) {
        g_culling_index = 1;
    }
    g_draw_normals = settings_reader.GetBoolean("renderer", "normals", false);
    settings.depthtest = settings_reader.GetBoolean("renderer", "depthtest", true);
    settings.background_r = static_cast<float>(settings_reader.GetReal("renderer", "background_r", 0.14));
    settings.background_g = static_cast<float>(settings_reader.GetReal("renderer", "background_g", 0.4));
    settings.background_b = static_cast<float>(settings_reader.GetReal("renderer", "background_b", 0.37));

    // Initial values for the "Terrain Controls" GUI sliders
    settings.initial_hurst = static_cast<float>(settings_reader.GetReal("terrain", "hurst", 0.8));
    settings.initial_seed = static_cast<uint32_t>(settings_reader.GetInteger("terrain", "seed", 1337));
    settings.initial_grid_size_exponent = static_cast<int>(settings_reader.GetInteger("terrain", "grid_size_exponent", 4));
    settings.initial_height_scale = static_cast<float>(settings_reader.GetReal("terrain", "height_scale", 1.0));
    settings.initial_roughness = static_cast<float>(settings_reader.GetReal("terrain", "roughness", 0.6));
    settings.initial_water_level = static_cast<float>(settings_reader.GetReal("terrain", "water_level", 2.0));
    g_camera_speed = static_cast<float>(settings_reader.GetReal("camera", "speed", 5.0));
    g_chunk_view_radius = static_cast<int>(settings_reader.GetInteger("chunks", "view_radius", 16));
    g_mouse_sensitivity = static_cast<float>(settings_reader.GetReal("camera", "mouse_sensitivity", 0.005));
    g_scroll_sensitivity = static_cast<float>(settings_reader.GetReal("camera", "scroll_sensitivity", 0.5));

    g_dirlight_color = glm::vec3(
        static_cast<float>(settings_reader.GetReal("light", "color_r", 0.85)),
        static_cast<float>(settings_reader.GetReal("light", "color_g", 0.85)),
        static_cast<float>(settings_reader.GetReal("light", "color_b", 0.85))
    );
    g_dirlight_dir = glm::vec3(
        static_cast<float>(settings_reader.GetReal("light", "dir_x", 0.0)),
        static_cast<float>(settings_reader.GetReal("light", "dir_y", 1.0)),
        static_cast<float>(settings_reader.GetReal("light", "dir_z", -1.0))
    );

    g_terrain_ka = static_cast<float>(settings_reader.GetReal("terrain", "ka", 0.1));
    g_terrain_kd = static_cast<float>(settings_reader.GetReal("terrain", "kd", 0.9));
    g_terrain_ks = static_cast<float>(settings_reader.GetReal("terrain", "ks", 0.3));
    g_terrain_alpha = static_cast<float>(settings_reader.GetReal("terrain", "alpha", 10.0));
    g_dirt_to_grass_height_offset = static_cast<float>(settings_reader.GetReal("terrain", "dirt_to_grass_height_offset", 3.0));
    g_grass_to_rock_height_offset = static_cast<float>(settings_reader.GetReal("terrain", "grass_to_rock_height_offset", 10.0));
    g_height_color_transition_band = static_cast<float>(settings_reader.GetReal("terrain", "height_color_transition_band", 2.0));
    g_water_depth_bias = static_cast<float>(settings_reader.GetReal("terrain", "water_depth_bias", 0.05));
    g_blend_duration = settings_reader.GetReal("terrain", "blend_duration", 2.0);

    g_max_destroys_per_frame = static_cast<int>(settings_reader.GetInteger("chunks", "max_destroys_per_frame", 16));
    g_max_uploads_per_frame = static_cast<int>(settings_reader.GetInteger("chunks", "max_uploads_per_frame", 8));

    g_demo_height_interp_duration = settings_reader.GetReal("demo", "height_interp_duration", 1.0);
    g_demo_water_interp_duration = settings_reader.GetReal("demo", "water_interp_duration", 1.0);

    return settings;
}

void errorCallbackFromGlfw(int error, const char* description) { std::cout << "GLFW error " << error << ": " << description << std::endl; }

void addInstanceExtensionToVectorIfSupported(const char* extension_name, std::vector<const char*>& ref_vector) {
    VkResult result;

    uint32_t instance_extension_count;
    result = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr);
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    std::vector<VkExtensionProperties> available_instance_extensions(instance_extension_count);
    result = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, available_instance_extensions.data());
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    for (const VkExtensionProperties& available_extension : available_instance_extensions) {
        if (strcmp(available_extension.extensionName, extension_name) == 0) {
            ref_vector.push_back(extension_name);
            return;
        }
    }
}

std::vector<const char*> getRequiredInstanceExtensions() {
    std::vector<const char*> required_extensions;

    uint32_t glfw_instance_extensions_count;
    const char** glfw_instance_extensions_names = glfwGetRequiredInstanceExtensions(&glfw_instance_extensions_count);
    for (uint32_t i = 0; i < glfw_instance_extensions_count; ++i) {
        addInstanceExtensionToVectorIfSupported(glfw_instance_extensions_names[i], required_extensions);
    }

    uint32_t framework_instance_extensions_count;
    const char** framework_instance_extensions_names = vklGetRequiredInstanceExtensions(&framework_instance_extensions_count);
    for (uint32_t i = 0; i < framework_instance_extensions_count; ++i) {
        addInstanceExtensionToVectorIfSupported(framework_instance_extensions_names[i], required_extensions);
    }
#ifdef __APPLE__
    addInstanceExtensionToVectorIfSupported(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, required_extensions);
#endif

    return required_extensions;
}

void addValidationLayerNameToVectorIfSupported(const char* validation_layer_name, std::vector<const char*>& ref_vector) {
    VkResult result;

    uint32_t layer_count;
    result = vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    std::vector<VkLayerProperties> available_layers(layer_count);
    result = vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    for (const VkLayerProperties& available_layer : available_layers) {
        if (strcmp(available_layer.layerName, validation_layer_name) == 0) {
            // Found the validation layer => Add it to the vector:
            ref_vector.push_back(validation_layer_name);
            return;
        }
    }
}

VkInstance createVulkanInstance() {
    VkApplicationInfo application_info = {};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pEngineName = "Kimsis_Engine_Terrain_generator";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 2023, 9, 1);
    application_info.pApplicationName = "Kimsis_Engine_Terrain";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 2023, 9, 19);
    application_info.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instance_create_info = {};
    instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.pApplicationInfo = &application_info;

    // A vector to hold all our requested instance extensions:
    std::vector<const char*> instance_extensions = getRequiredInstanceExtensions();

    instance_create_info.enabledExtensionCount = instance_extensions.size();
    instance_create_info.ppEnabledExtensionNames = instance_extensions.data();

    // A vector to hold all our requested validation layers, add standard validation, and set info in the VkInstanceCreateInfo struct:
    std::vector<const char*> enabled_layer_names;
    addValidationLayerNameToVectorIfSupported("VK_LAYER_KHRONOS_validation", enabled_layer_names);
    instance_create_info.enabledLayerCount = enabled_layer_names.size();
    instance_create_info.ppEnabledLayerNames = enabled_layer_names.data();
#ifdef __APPLE__
    instance_create_info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstance vk_instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&instance_create_info, nullptr, &vk_instance);
    VKL_CHECK_VULKAN_RESULT(result);
    if (!vk_instance) {
        VKL_EXIT_WITH_ERROR("No VkInstance created or handle not assigned.");
    }
    return vk_instance;
}

VkSurfaceKHR createWindowSurface(VkInstance vk_instance, GLFWwindow* window) {
    VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
    VkResult result = glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface);
    VKL_CHECK_VULKAN_RESULT(result);
    if (!vk_surface) {
        VKL_EXIT_WITH_ERROR("No VkSurfaceKHR created or handle not assigned.");
    }
    return vk_surface;
}

uint32_t selectPhysicalDeviceIndex(const VkPhysicalDevice* physical_devices, uint32_t physical_device_count, VkSurfaceKHR surface) {
    // Iterate over all the physical devices and select one that satisfies all our requirements.
    // Our requirements are:
    //  - Must support a queue that must have both, graphics and presentation capabilities
    for (uint32_t physical_device_index = 0u; physical_device_index < physical_device_count; ++physical_device_index) {
        // Check if fillModeNonSolid is supported
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(physical_devices[physical_device_index], &features);
        if (VK_TRUE != features.fillModeNonSolid) {
            continue; // This physical device does not support it => look for a different one
        }

        // Get the number of different queue families:
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[physical_device_index], &queue_family_count, nullptr);

        // Get the queue families' data:
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[physical_device_index], &queue_family_count, queue_families.data());

        for (uint32_t queue_family_index = 0u; queue_family_index < queue_family_count; ++queue_family_index) {
            // If this physical device supports a queue family which supports both, graphics and presentation
            //  => select this physical device
            if ((queue_families[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                // This queue supports graphics! Let's see if it also supports presentation:
                VkBool32 presentation_supported;
                vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[physical_device_index], queue_family_index, surface, &presentation_supported);

                if (VK_TRUE == presentation_supported) {
                    // We've found a suitable physical device
                    return physical_device_index;
                }
            }
        }
    }
    VKL_EXIT_WITH_ERROR("Unable to find a suitable physical device that supports graphics and presentation on the same queue.");
}

uint32_t selectPhysicalDeviceIndex(const std::vector<VkPhysicalDevice>& physical_devices, VkSurfaceKHR surface) {
    return selectPhysicalDeviceIndex(physical_devices.data(), static_cast<uint32_t>(physical_devices.size()), surface);
}

VkPhysicalDevice pickPhysicalDevice(VkInstance vk_instance, VkSurfaceKHR vk_surface) {
    // Query the number of physical devices:
    uint32_t physical_devices_count;
    vkEnumeratePhysicalDevices(vk_instance, &physical_devices_count, nullptr);

    if (physical_devices_count == 0) {
        VKL_EXIT_WITH_ERROR("Vulkan does not recognize any physical devices.");
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_devices_count);
    vkEnumeratePhysicalDevices(vk_instance, &physical_devices_count, physical_devices.data());

    uint32_t selected_physical_device_index = selectPhysicalDeviceIndex(physical_devices, vk_surface);
    VkPhysicalDevice vk_physical_device = physical_devices[selected_physical_device_index];
    if (!vk_physical_device) {
        VKL_EXIT_WITH_ERROR("No VkPhysicalDevice selected or handle not assigned.");
    }
    return vk_physical_device;
}

uint32_t selectQueueFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    // Get the number of different queue families for the given physical device:
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

    // Get the queue families' data:
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    for (uint32_t queue_family_index = 0u; queue_family_index < queue_family_count; ++queue_family_index) {
        if ((queue_families[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            // This queue supports graphics! Let's see if it also supports presentation:
            VkBool32 presentation_supported;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index, surface, &presentation_supported);

            if (VK_TRUE == presentation_supported) {
                // We've found a suitable queue family on the given physical device and for the given surface
                //  => return its INDEX:
                return queue_family_index;
            }
        }
    }
    VKL_EXIT_WITH_ERROR("Unable to find a suitable queue family that supports graphics and presentation on the same queue.");
}

VkDeviceQueueCreateInfo createQueueCreateInfo(VkPhysicalDevice vk_physical_device, uint32_t selected_queue_family_index) {
    // Fixed storage duration so the returned struct's pQueuePriorities pointer stays valid for as
    // long as the caller keeps using it (a plain local array here would dangle once this function
    // returns) — safe since the priority value itself never varies.
    static constexpr std::array<float, 1> kQueuePriorities = {1.0f};

    VkDeviceQueueCreateInfo device_queue_create_info = {};
    device_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    device_queue_create_info.queueFamilyIndex = selected_queue_family_index;
    device_queue_create_info.queueCount = 1u;
    device_queue_create_info.pQueuePriorities = kQueuePriorities.data();

    // Sanity check if we have selected a valid queue family index:
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, nullptr);
    if (selected_queue_family_index >= queue_family_count) {
        VKL_EXIT_WITH_ERROR("Invalid queue family index selected.");
    }
    return device_queue_create_info;
}

void addDeviceExtensionToVectorIfSupported(const char* extension_name, VkPhysicalDevice physical_device, std::vector<const char*>& ref_vector) {
    VkResult result;

    uint32_t extensions_count;
    result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, nullptr);
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    std::vector<VkExtensionProperties> available_extensions(extensions_count);
    result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, available_extensions.data());
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    for (const VkExtensionProperties& available_extension : available_extensions) {
        if (strcmp(available_extension.extensionName, extension_name) == 0) {
            ref_vector.push_back(extension_name);
            return;
        }
    }
}

VkDevice createLogicalDeviceAndQueue(
    VkPhysicalDevice vk_physical_device,
    const VkDeviceQueueCreateInfo& device_queue_create_info,
    uint32_t selected_queue_family_index,
    VkQueue& out_vk_queue
) {
    std::vector<const char*> device_extensions;
    addDeviceExtensionToVectorIfSupported(VK_KHR_SWAPCHAIN_EXTENSION_NAME, vk_physical_device, device_extensions);
    addDeviceExtensionToVectorIfSupported(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, vk_physical_device, device_extensions);
    // Looks like SDK 1.2.170 also requires
    addDeviceExtensionToVectorIfSupported(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, vk_physical_device, device_extensions);
    // See if this device supports Synchronization2, and if so, set a flag to indicate that we are going to use Synchronization2:
    if (std::find(std::begin(device_extensions), std::end(device_extensions), std::string(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) !=
        std::end(device_extensions)) {
        g_synchronization2_supported = true;
    }
#ifdef __APPLE__
    addDeviceExtensionToVectorIfSupported("VK_KHR_portability_subset", vk_physical_device, device_extensions);
#endif

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // Add information about queues to be created, and extensions to be enabled:
    device_create_info.queueCreateInfoCount = 1u;
    device_create_info.pQueueCreateInfos = &device_queue_create_info;
    device_create_info.enabledExtensionCount = device_extensions.size();
    device_create_info.ppEnabledExtensionNames = device_extensions.data();

    VkPhysicalDeviceFeatures enabled_physical_device_features = {};
    enabled_physical_device_features.fillModeNonSolid = VK_TRUE;
    device_create_info.pEnabledFeatures = &enabled_physical_device_features;

    // Enable Synchronization2 and hook it into the pNext chain:
    VkPhysicalDeviceSynchronization2FeaturesKHR physical_device_sync2_features = {};
    physical_device_sync2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
    physical_device_sync2_features.synchronization2 = VK_TRUE;
    if (g_synchronization2_supported) {
        device_create_info.pNext = &physical_device_sync2_features;
    }

    // Create the logical device
    VkDevice vk_device = VK_NULL_HANDLE;
    VkResult result = vkCreateDevice(vk_physical_device, &device_create_info, nullptr, &vk_device);
    VKL_CHECK_VULKAN_RESULT(result);

    if (g_synchronization2_supported) {
        auto* procAddr = vkGetDeviceProcAddr(vk_device, "vkCmdPipelineBarrier2KHR");
        if (procAddr == nullptr) {
            // Couldn't get the function pointer to vkCmdPipelineBarrier2KHR
            g_synchronization2_supported = false;
        } else {
            g_vkCmdPipelineBarrier2KHR = reinterpret_cast<PFN_vkCmdPipelineBarrier2KHR>(procAddr);
        }
    }
    if (!vk_device) {
        VKL_EXIT_WITH_ERROR("No VkDevice created or handle not assigned.");
    }

    // Get the handle of the queue that was requested:
    vkGetDeviceQueue(vk_device, selected_queue_family_index, 0u, &out_vk_queue);
    if (!out_vk_queue) {
        VKL_EXIT_WITH_ERROR("No VkQueue selected or handle not assigned.");
    }
    return vk_device;
}

VkSurfaceFormatKHR getSurfaceImageFormat(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    VkResult result;

    uint32_t surface_format_count;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, nullptr);
    VKL_CHECK_VULKAN_ERROR(result);

    std::vector<VkSurfaceFormatKHR> surface_formats(surface_format_count);
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, surface_formats.data());
    VKL_CHECK_VULKAN_ERROR(result);

    if (surface_formats.empty()) {
        VKL_EXIT_WITH_ERROR("Unable to find supported surface formats.");
    }

    // Prefer a RGB8/sRGB format; If we are unable to find such, just return any:
    for (const VkSurfaceFormatKHR& f : surface_formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }

    return surface_formats[0];
}

VkSurfaceCapabilitiesKHR getPhysicalDeviceSurfaceCapabilities(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    VkSurfaceCapabilitiesKHR surface_capabilities;
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surface_capabilities);
    VKL_CHECK_VULKAN_ERROR(result);
    return surface_capabilities;
}

SwapchainSetup createSwapchain(
    VkPhysicalDevice vk_physical_device,
    VkSurfaceKHR vk_surface,
    VkDevice vk_device,
    uint32_t selected_queue_family_index,
    int& window_width,
    int& window_height
) {
    SwapchainSetup setup{};
    setup.surface_format = getSurfaceImageFormat(vk_physical_device, vk_surface);
    setup.surface_capabilities = getPhysicalDeviceSurfaceCapabilities(vk_physical_device, vk_surface);

    // Clamp to what the surface actually reports; the window manager doesn't always honor the
    // requested width/height exactly.
    if (setup.surface_capabilities.currentExtent.width != UINT32_MAX) {
        window_width = static_cast<int>(setup.surface_capabilities.currentExtent.width);
        window_height = static_cast<int>(setup.surface_capabilities.currentExtent.height);
    } else {
        window_width = static_cast<int>(std::clamp(
            static_cast<uint32_t>(window_width),
            setup.surface_capabilities.minImageExtent.width,
            setup.surface_capabilities.maxImageExtent.width
        ));
        window_height = static_cast<int>(std::clamp(
            static_cast<uint32_t>(window_height),
            setup.surface_capabilities.minImageExtent.height,
            setup.surface_capabilities.maxImageExtent.height
        ));
    }

    std::vector<uint32_t> queueFamilyIndices = {selected_queue_family_index};

    // Build the swapchain config struct:
    VkSwapchainCreateInfoKHR swapchain_create_info = {};
    swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_create_info.surface = vk_surface;
    swapchain_create_info.minImageCount = setup.surface_capabilities.minImageCount;
    swapchain_create_info.imageArrayLayers = 1u;
    swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_create_info.preTransform = setup.surface_capabilities.currentTransform;
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_create_info.clipped = VK_TRUE;
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_create_info.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
    swapchain_create_info.pQueueFamilyIndices = queueFamilyIndices.data();
    swapchain_create_info.imageFormat = setup.surface_format.format;
    swapchain_create_info.imageColorSpace = setup.surface_format.colorSpace;
    swapchain_create_info.imageExtent = VkExtent2D{static_cast<uint32_t>(window_width), static_cast<uint32_t>(window_height)};
    swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;

    // Create the swapchain:
    VkResult result = vkCreateSwapchainKHR(vk_device, &swapchain_create_info, nullptr, &setup.swapchain);
    VKL_CHECK_VULKAN_RESULT(result);
    if (!setup.swapchain) {
        VKL_EXIT_WITH_ERROR("No VkSwapchainKHR created or handle not assigned.");
    }
    setup.image_usage = swapchain_create_info.imageUsage;

    // Query how many swapchain images we got:
    uint32_t swapchain_image_count;
    vkGetSwapchainImagesKHR(vk_device, setup.swapchain, &swapchain_image_count, nullptr);

    // Retrieve the swapchain images:
    setup.image_handles.resize(swapchain_image_count);
    vkGetSwapchainImagesKHR(vk_device, setup.swapchain, &swapchain_image_count, setup.image_handles.data());

    return setup;
}

std::optional<Hit> raycastTerrain(const glm::vec3& origin, const glm::vec3& direction) {
    constexpr float kGroundPlaneZ = 0.0f;
    if (std::abs(direction.z) < 1e-6f) return std::nullopt;
    float t = (kGroundPlaneZ - origin.z) / direction.z;
    if (t < 0.0f) return std::nullopt;
    glm::vec3 point = origin + t * direction;
    return Hit{point, t};
}

VkImage createDepthBuffer(VkPhysicalDevice vk_physical_device, VkDevice vk_device, int width, int height) {
    return vklCreateDeviceLocalImageWithBackingMemory(
        vk_physical_device,
        vk_device,
        width,
        height,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

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
) {
    VklSwapchainConfig swapchain_config = {};
    swapchain_config.swapchainHandle = vk_swapchain;
    swapchain_config.imageExtent = image_extent;
    for (const VkImage& img : swapchain_image_handles) {
        VklSwapchainFramebufferComposition framebufferComposition;
        framebufferComposition.colorAttachmentImageDetails.imageHandle = img;
        framebufferComposition.colorAttachmentImageDetails.imageFormat = image_format;
        framebufferComposition.colorAttachmentImageDetails.imageUsage = image_usage;
        framebufferComposition.colorAttachmentImageDetails.clearValue = color_clear_value;
        if (depthtest) {
            // If we also set the data of the depth buffer, our framebuffer will consist of two images:
            framebufferComposition.depthAttachmentImageDetails.imageHandle = depth_buffer;
            framebufferComposition.depthAttachmentImageDetails.imageFormat = VK_FORMAT_D32_SFLOAT;
            framebufferComposition.depthAttachmentImageDetails.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            framebufferComposition.depthAttachmentImageDetails.clearValue = depth_clear_value;
        }
        swapchain_config.swapchainImages.push_back(framebufferComposition);
    }
    return swapchain_config;
}

void initImGui(
    GLFWwindow* window,
    VkInstance vk_instance,
    VkPhysicalDevice vk_physical_device,
    VkDevice vk_device,
    uint32_t selected_queue_family_index,
    VkQueue vk_queue,
    uint32_t min_image_count,
    uint32_t image_count
) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, /*install_callbacks=*/false);

    ImGui_ImplVulkan_InitInfo imgui_init_info = {};
    imgui_init_info.ApiVersion = VK_API_VERSION_1_1;
    imgui_init_info.Instance = vk_instance;
    imgui_init_info.PhysicalDevice = vk_physical_device;
    imgui_init_info.Device = vk_device;
    imgui_init_info.QueueFamily = selected_queue_family_index;
    imgui_init_info.Queue = vk_queue;
    imgui_init_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
    imgui_init_info.MinImageCount = min_image_count;
    imgui_init_info.ImageCount = image_count;
    imgui_init_info.PipelineInfoMain.RenderPass = vklGetRenderpass();
    imgui_init_info.PipelineInfoMain.Subpass = 0u;
    imgui_init_info.MinAllocationSize = 1024u * 1024u;
    ImGui_ImplVulkan_Init(&imgui_init_info);
}

/* --------------------------------------------- */
// Build the terrain pipeline for given polygon and cull modes
/* --------------------------------------------- */

VkPipeline buildTerrainPipeline(const TerrainScene& scene, size_t polygon_mode_index, size_t cull_mode_index) {
    VklGraphicsPipelineConfig pipeline_config{
        scene.vertexShaderPath.c_str(),
        scene.fragmentShaderPath.c_str(),
        {
            VkVertexInputBindingDescription{0u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
            VkVertexInputBindingDescription{1u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
            VkVertexInputBindingDescription{2u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
            VkVertexInputBindingDescription{3u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX},
        },
        {
            VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
            VkVertexInputAttributeDescription{1u, 1u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
            VkVertexInputAttributeDescription{2u, 2u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
            VkVertexInputAttributeDescription{3u, 3u, VK_FORMAT_R32G32B32_SFLOAT, 0u},
        },
        /* --------------------------------------------- */
        // Wireframe Mode
        /* --------------------------------------------- */
        kTerrainPolygonModes[polygon_mode_index],
        /* --------------------------------------------- */
        // Back-face Culling
        /* --------------------------------------------- */
        kTerrainCullModes[cull_mode_index],
        scene.descriptorSetLayoutBindings,
        /* enableAlphaBlending: */ false,
        {VkPushConstantRange{VK_SHADER_STAGE_VERTEX_BIT, 0u, sizeof(TerrainPushConstants)}},
    };
    return vklCreateGraphicsPipeline(pipeline_config);
}

VkDescriptorSet allocDescriptorSet(VkDevice device, VkDescriptorPool descriptor_pool, VkDescriptorSetLayout descriptor_set_layout) {
    VkResult result;

    // Prepare allocation info and allocate:
    VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
    descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_set_allocate_info.descriptorPool = descriptor_pool;
    descriptor_set_allocate_info.descriptorSetCount = 1u;
    descriptor_set_allocate_info.pSetLayouts = &descriptor_set_layout;

    VkDescriptorSet descriptor_set;
    result = vkAllocateDescriptorSets(device, &descriptor_set_allocate_info, &descriptor_set);
    VKL_CHECK_VULKAN_RESULT(result);

    if (result < VK_SUCCESS) {
        VKL_EXIT_WITH_ERROR("Allocating a new descriptor set from the given pool and of the given layout failed.");
    }

    return descriptor_set;
}

void writeDescriptorSet(VkDevice device, VkDescriptorSet descriptor_set, VkBuffer vert_buffer, VkBuffer frag_buffer) {
    VkDescriptorBufferInfo vert_descriptor_buffer_info = {};
    vert_descriptor_buffer_info.buffer = vert_buffer;
    vert_descriptor_buffer_info.offset = static_cast<VkDeviceSize>(0);
    vert_descriptor_buffer_info.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo frag_descriptor_buffer_info = {};
    frag_descriptor_buffer_info.buffer = frag_buffer;
    frag_descriptor_buffer_info.offset = static_cast<VkDeviceSize>(0);
    frag_descriptor_buffer_info.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> writes = {
        VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            /* dstBinding: */ 0u,
            0u,
            1u,
            /* descriptorType: */ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            nullptr,
            /* pBufferInfo: */ &vert_descriptor_buffer_info,
            nullptr
        },
        VkWriteDescriptorSet{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            /* dstBinding: */ 1u,
            0u,
            1u,
            /* descriptorType: */ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            nullptr,
            /* pBufferInfo: */ &frag_descriptor_buffer_info,
            nullptr
        },
    };

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
}

void writeDescriptorSet(
    VkDevice device,
    VkDescriptorSet descriptor_set,
    VkBuffer vert_buffer,
    VkBuffer frag_buffer,
    VkBuffer directional_light_data
) {
    writeDescriptorSet(device, descriptor_set, vert_buffer, frag_buffer);

    VkDescriptorBufferInfo dirlight_buffer_info = {};
    dirlight_buffer_info.buffer = directional_light_data;
    dirlight_buffer_info.offset = static_cast<VkDeviceSize>(0);
    dirlight_buffer_info.range = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> writes = {VkWriteDescriptorSet{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        descriptor_set,
        /* dstBinding: */ 2u,
        0u,
        1u,
        /* descriptorType: */ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        nullptr,
        /* pBufferInfo: */ &dirlight_buffer_info,
        nullptr
    }};

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
}

TerrainScene setupTerrainScene(
    VkDevice vk_device,
    VkQueue vk_queue,
    uint32_t selected_queue_family_index,
    TerrainParams& params,
    float initial_height_scale,
    float initial_roughness,
    float initial_water_level
) {
    TerrainScene scene{};
    scene.heightScale = initial_height_scale;
    scene.roughness = initial_roughness;
    scene.waterLevel = initial_water_level;
    scene.chunkManager.baseParams = params;
    scene.chunkManager.viewRadius = g_chunk_view_radius;
    scene.chunkManager.blendDuration = g_blend_duration;
    scene.chunkManager.maxDestroysPerFrame = g_max_destroys_per_frame;
    scene.chunkManager.maxUploadsPerFrame = g_max_uploads_per_frame;
    g_hurst = params.hurst;

    /* --------------------------------------------- */
    // Create a Custom Graphics Pipeline
    /* --------------------------------------------- */
    scene.descriptorSetLayoutBindings = {
        VkDescriptorSetLayoutBinding{0u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    scene.vertexShaderPath = gcgFindShaderFile("assets/shaders/terrain.vert");
    scene.fragmentShaderPath = gcgFindShaderFile("assets/shaders/terrain.frag");

    /* --------------------------------------------- */
    // Interaction
    /* --------------------------------------------- */
    scene.pipelines[g_polygon_mode_index][g_culling_index] = buildTerrainPipeline(scene, g_polygon_mode_index, g_culling_index);

    /* --------------------------------------------- */
    // Allocate and Write Descriptors
    /* --------------------------------------------- */
    std::vector<VkDescriptorPoolSize> pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3u}};

    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.maxSets = 1u;
    descriptor_pool_create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    descriptor_pool_create_info.pPoolSizes = pool_sizes.data();

    VkResult result = vkCreateDescriptorPool(vk_device, &descriptor_pool_create_info, nullptr, &scene.descriptor_pool);
    VKL_CHECK_VULKAN_RESULT(result);

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
    descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    // We can reuse the same layout description that we have passed to graphics pipeline creation:
    descriptor_set_layout_create_info.bindingCount = static_cast<uint32_t>(scene.descriptorSetLayoutBindings.size());
    descriptor_set_layout_create_info.pBindings = scene.descriptorSetLayoutBindings.data();

    result = vkCreateDescriptorSetLayout(vk_device, &descriptor_set_layout_create_info, nullptr, &scene.descriptor_set_layout);
    VKL_CHECK_VULKAN_RESULT(result);

    VkDeviceSize num_dirlights = 1;
    scene.ub_dirlight = vklCreateHostCoherentBufferWithBackingMemory(
        sizeof(DirectionalLight) * num_dirlights,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    DirectionalLight directional_light = {glm::vec4(g_dirlight_color, 0.0f), glm::normalize(glm::vec4(g_dirlight_dir, 0.0f))};
    vklCopyDataIntoHostCoherentBuffer(scene.ub_dirlight, &directional_light, sizeof(DirectionalLight));

    scene.ub_terrain_vert = vklCreateHostCoherentBufferWithBackingMemory(
        sizeof(UniformBufferVert),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    scene.ub_terrain_frag = vklCreateHostCoherentBufferWithBackingMemory(
        sizeof(UniformBufferFrag),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    scene.ds_terrain = allocDescriptorSet(vk_device, scene.descriptor_pool, scene.descriptor_set_layout);
    writeDescriptorSet(vk_device, scene.ds_terrain, scene.ub_terrain_vert, scene.ub_terrain_frag, scene.ub_dirlight);

    return scene;
}

void buildLoadingGUI(size_t pendingChunkCount) {
    const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin(
        "Loading",
        nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove
    );
    ImGui::Text("Generating terrain... (%zu chunks remaining)", pendingChunkCount);
    ImGui::End();
}

static size_t chunksStillGenerating(const ChunkManager& chunkManager) {
    return chunkManager.pendingChunks.size() + chunkManager.readyForNormals.size() + chunkManager.pendingNormals.size();
}

void generateTerrainGeometryWithLoadingScreen(VkDevice vk_device, ChunkManager& chunkManager, const glm::vec3& cameraPos) {
    updateLoadedChunks(chunkManager, cameraPos, glfwGetTime());
    while (chunksStillGenerating(chunkManager) > 0) {
        glfwPollEvents();

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        buildLoadingGUI(chunksStillGenerating(chunkManager));
        ImGui::Render();

        vklWaitForNextSwapchainImage();
        vklStartRecordingCommands();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vklGetCurrentCommandBuffer());
        vklEndRecordingCommands();
        vklPresentCurrentSwapchainImage();

        updateLoadedChunks(chunkManager, cameraPos, glfwGetTime());
    }
}

WaterScene setupWaterScene(VkDevice vk_device, const TerrainParams& terrain_params) {
    WaterScene scene{};

    scene.vertexShaderPath = gcgFindShaderFile("assets/shaders/water.vert");
    scene.fragmentShaderPath = gcgFindShaderFile("assets/shaders/water.frag");

    std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings = {
        VkDescriptorSetLayoutBinding{0u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
    };

    VklGraphicsPipelineConfig pipeline_config{
        scene.vertexShaderPath.c_str(),
        scene.fragmentShaderPath.c_str(),
        {VkVertexInputBindingDescription{0u, sizeof(float) * 3, VK_VERTEX_INPUT_RATE_VERTEX}},
        {VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, 0u}},
        VK_POLYGON_MODE_FILL,
        VK_CULL_MODE_NONE,
        descriptor_set_layout_bindings,
        /* enableAlphaBlending: */ true,
    };
    scene.pipeline = vklCreateGraphicsPipeline(pipeline_config);

    std::vector<VkDescriptorPoolSize> pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u}};
    VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
    descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_create_info.maxSets = 1u;
    descriptor_pool_create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    descriptor_pool_create_info.pPoolSizes = pool_sizes.data();
    VkResult result = vkCreateDescriptorPool(vk_device, &descriptor_pool_create_info, nullptr, &scene.descriptor_pool);
    VKL_CHECK_VULKAN_RESULT(result);

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
    descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_set_layout_create_info.bindingCount = static_cast<uint32_t>(descriptor_set_layout_bindings.size());
    descriptor_set_layout_create_info.pBindings = descriptor_set_layout_bindings.data();
    result = vkCreateDescriptorSetLayout(vk_device, &descriptor_set_layout_create_info, nullptr, &scene.descriptor_set_layout);
    VKL_CHECK_VULKAN_RESULT(result);

    scene.ub_water_vert = vklCreateHostCoherentBufferWithBackingMemory(
        sizeof(UniformBufferWaterVert),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    scene.ds_water = allocDescriptorSet(vk_device, scene.descriptor_pool, scene.descriptor_set_layout);

    VkDescriptorBufferInfo vert_buffer_info = {};
    vert_buffer_info.buffer = scene.ub_water_vert;
    vert_buffer_info.offset = static_cast<VkDeviceSize>(0);
    vert_buffer_info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        nullptr,
        scene.ds_water,
        /* dstBinding: */ 0u,
        0u,
        1u,
        /* descriptorType: */ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        nullptr,
        /* pBufferInfo: */ &vert_buffer_info,
        nullptr
    };
    vkUpdateDescriptorSets(vk_device, 1u, &write, 0u, nullptr);

    // chunkGeometry is populated lazily by updateWaterChunks() as chunks load.
    return scene;
}

/*!
 *	This callback function gets invoked by GLFW during glfwPollEvents() if there was
 *	mouse button input that can be processed by our application.
 */
void mouseButtonCallbackFromGlfw(GLFWwindow* glfw_window, int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(glfw_window, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        g_dragging = true;
    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        g_dragging = false;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        g_strafing = true;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        g_strafing = false;
    }
}

/*!
 *	This callback function gets invoked by GLFW during glfwPollEvents() if there was
 *	mouse scroll input that can be processed by our application.
 */
void scrollCallbackFromGlfw(GLFWwindow* glfw_window, double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(glfw_window, xoffset, yoffset);
    if (ImGui::GetIO().WantCaptureMouse) return;

    g_scroll_delta += static_cast<float>(yoffset);
}

void framebufferSizeCallbackFromGlfw(GLFWwindow* glfw_window, int width, int height) { g_framebuffer_resized = true; }

void handleGlfwKeyCallback(GLFWwindow* glfw_window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(glfw_window, key, scancode, action, mods);
    if (ImGui::GetIO().WantCaptureKeyboard) return;

    if (action != GLFW_RELEASE) return;
    if (key == GLFW_KEY_ESCAPE) {
        glfwSetWindowShouldClose(glfw_window, true);
    }
    /* --------------------------------------------- */
    // Interaction
    /* --------------------------------------------- */
    if (key == GLFW_KEY_F1) {
        g_polygon_mode_index = 1 - g_polygon_mode_index;
    }
    if (key == GLFW_KEY_F2) {
        g_culling_index = (g_culling_index + 1) % 3;
    }
    if (key == GLFW_KEY_F3) {
        g_highlight_chunk_borders = !g_highlight_chunk_borders;
    }
    if (key == GLFW_KEY_N) {
        g_draw_normals = !g_draw_normals;
    }
    if (key == GLFW_KEY_C) {
        g_toggle_camera_requested = true;
    }
    if (key == GLFW_KEY_R) {
        g_reseed_requested = true;
    }
    if (key == GLFW_KEY_F4) {
        g_demo_mode = !g_demo_mode;
    }
}

void recreateSwapchainAndDependents(
    const SwapchainRecreateContext& ctx,
    VkSwapchainKHR& vk_swapchain,
    VkImage& depth_buffer,
    std::vector<VkImage>& swapchain_image_handles,
    int& window_width,
    int& window_height,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera
) {
    int fb_width = 0, fb_height = 0;
    glfwGetFramebufferSize(ctx.window, &fb_width, &fb_height);
    while (fb_width == 0 || fb_height == 0) {
        // Minimized: block until the window is restored.
        glfwGetFramebufferSize(ctx.window, &fb_width, &fb_height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(ctx.vk_device);

    vklDestroyDeviceLocalImageAndItsBackingMemory(depth_buffer);
    VkSwapchainKHR old_swapchain = vk_swapchain;
    vklDestroyFramework();

    VkSurfaceCapabilitiesKHR new_surface_capabilities = getPhysicalDeviceSurfaceCapabilities(ctx.vk_physical_device, ctx.vk_surface);
    VkExtent2D new_extent;
    if (new_surface_capabilities.currentExtent.width != UINT32_MAX) {
        new_extent = new_surface_capabilities.currentExtent;
    } else {
        new_extent.width =
            std::clamp(static_cast<uint32_t>(fb_width), new_surface_capabilities.minImageExtent.width, new_surface_capabilities.maxImageExtent.width);
        new_extent.height = std::clamp(
            static_cast<uint32_t>(fb_height),
            new_surface_capabilities.minImageExtent.height,
            new_surface_capabilities.maxImageExtent.height
        );
    }

    VkSwapchainCreateInfoKHR new_swapchain_create_info = {};
    new_swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    new_swapchain_create_info.surface = ctx.vk_surface;
    new_swapchain_create_info.minImageCount = new_surface_capabilities.minImageCount;
    new_swapchain_create_info.imageArrayLayers = 1u;
    new_swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    new_swapchain_create_info.preTransform = new_surface_capabilities.currentTransform;
    new_swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    new_swapchain_create_info.clipped = VK_TRUE;
    new_swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    new_swapchain_create_info.queueFamilyIndexCount = 1u;
    new_swapchain_create_info.pQueueFamilyIndices = &ctx.selected_queue_family_index;
    new_swapchain_create_info.imageFormat = ctx.surface_format.format;
    new_swapchain_create_info.imageColorSpace = ctx.surface_format.colorSpace;
    new_swapchain_create_info.imageExtent = new_extent;
    new_swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    new_swapchain_create_info.oldSwapchain = old_swapchain;

    VkResult swap_result = vkCreateSwapchainKHR(ctx.vk_device, &new_swapchain_create_info, nullptr, &vk_swapchain);
    VKL_CHECK_VULKAN_RESULT(swap_result);
    vkDestroySwapchainKHR(ctx.vk_device, old_swapchain, nullptr);

    uint32_t new_swapchain_image_count;
    vkGetSwapchainImagesKHR(ctx.vk_device, vk_swapchain, &new_swapchain_image_count, nullptr);
    swapchain_image_handles.resize(new_swapchain_image_count);
    vkGetSwapchainImagesKHR(ctx.vk_device, vk_swapchain, &new_swapchain_image_count, swapchain_image_handles.data());

    depth_buffer = createDepthBuffer(ctx.vk_physical_device, ctx.vk_device, static_cast<int>(new_extent.width), static_cast<int>(new_extent.height));

    VklSwapchainConfig new_swapchain_config = buildSwapchainConfig(
        vk_swapchain,
        swapchain_image_handles,
        new_extent,
        new_swapchain_create_info.imageFormat,
        new_swapchain_create_info.imageUsage,
        ctx.color_clear_value,
        ctx.depthtest,
        depth_buffer,
        ctx.depth_clear_value
    );

    if (!vklInitFramework(ctx.vk_instance, ctx.vk_surface, ctx.vk_physical_device, ctx.vk_device, ctx.vk_queue, new_swapchain_config)) {
        VKL_EXIT_WITH_ERROR("Failed to reinit framework after window resize");
    }

    // Rebuilds every registered pipeline against the fresh render pass and extent:
    vklHotReloadPipelines();

    window_width = static_cast<int>(new_extent.width);
    window_height = static_cast<int>(new_extent.height);
    float new_aspect_ratio = static_cast<float>(new_extent.width) / static_cast<float>(new_extent.height);
    trackballCamera.setAspectRatio(new_aspect_ratio);
    flyCamera.setAspectRatio(new_aspect_ratio);
}

void handleCameraToggleRequest(
    GLFWwindow* window,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera,
    Camera*& activeCamera,
    double& mouse_x,
    double& mouse_y,
    double& mouse_x_last,
    double& mouse_y_last
) {
    if (!g_toggle_camera_requested) return;
    g_toggle_camera_requested = false;

    if (!g_toggle_camera) {
        // Fly camera
        flyCamera = FlyCamera(trackballCamera);
        activeCamera = &flyCamera;
        g_toggle_camera = true;
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    } else {
        // Trackball camera
        auto hit = raycastTerrain(flyCamera.getPosition(), flyCamera.getForward());
        if (hit.has_value()) {
            trackballCamera = TrackballCamera(flyCamera, hit->point);
            activeCamera = &trackballCamera;
            g_toggle_camera = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
    glfwGetCursorPos(window, &mouse_x, &mouse_y);
    mouse_x_last = mouse_x;
    mouse_y_last = mouse_y;
}

uint32_t generateRandomSeed() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<uint32_t> dist;
    return dist(rng);
}

// Inset from the slider's full [0,1] range to avoid the visually-degenerate extremes (near-flat at
// 1, near-white-noise at 0).
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

// Inset from the slider's full [-25, 25] range — comfortably varied without drifting the water
// plane absurdly far from where the terrain actually sits.
float generateRandomWaterLevel() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> dist(-15.0f, 15.0f);
    return dist(rng);
}

void updateTerrainState(TerrainScene& terrain_scene, double currentFrameTime) {
    // Both triggers are ignored while a previous regeneration is still resolving or blending.
    if (g_reseed_requested) {
        g_reseed_requested = false;
        if (!isRegenerating(terrain_scene.chunkManager)) {
            terrain_scene.chunkManager.baseParams.seed = generateRandomSeed();
            invalidateAllLoadedChunks(terrain_scene.chunkManager);
        }
    }

    if (g_hurst_changed) {
        g_hurst_changed = false;
        if (!isRegenerating(terrain_scene.chunkManager) && g_hurst != terrain_scene.chunkManager.baseParams.hurst) {
            terrain_scene.chunkManager.baseParams.hurst = g_hurst;
            invalidateAllLoadedChunks(terrain_scene.chunkManager);
        } else {
            g_hurst = terrain_scene.chunkManager.baseParams.hurst;
        }
    }

    // Demo mode (F4): fires the instant the previous regeneration's blend has fully settled.
    if (g_demo_mode && !isRegenerating(terrain_scene.chunkManager)) {
        g_hurst = generateRandomHurst();
        terrain_scene.chunkManager.baseParams.hurst = g_hurst;
        terrain_scene.chunkManager.baseParams.seed = generateRandomSeed();
        invalidateAllLoadedChunks(terrain_scene.chunkManager);
    }

    // Demo mode's height scale: independent of the above — picks a new random target every second
    // and smoothly interpolates toward it.
    if (g_demo_mode) {
        if (g_demo_height_interp_start_time < 0.0) {
            // First activation: interpolate from whatever the height scale currently is.
            g_demo_height_start = terrain_scene.heightScale;
            g_demo_height_target = generateRandomHeightScale();
            g_demo_height_interp_start_time = currentFrameTime;
        }
        double elapsed = currentFrameTime - g_demo_height_interp_start_time;
        if (elapsed >= g_demo_height_interp_duration) {
            g_demo_height_start = g_demo_height_target;
            g_demo_height_target = generateRandomHeightScale();
            g_demo_height_interp_start_time = currentFrameTime;
            elapsed = 0.0;
        }
        float t = static_cast<float>(glm::clamp(elapsed / g_demo_height_interp_duration, 0.0, 1.0));
        terrain_scene.heightScale = glm::mix(g_demo_height_start, g_demo_height_target, t);
    }

    // Demo mode's water level: same independent-timer drift pattern as height scale above.
    if (g_demo_mode) {
        if (g_demo_water_interp_start_time < 0.0) {
            g_demo_water_start = terrain_scene.waterLevel;
            g_demo_water_target = generateRandomWaterLevel();
            g_demo_water_interp_start_time = currentFrameTime;
        }
        double elapsed = currentFrameTime - g_demo_water_interp_start_time;
        if (elapsed >= g_demo_water_interp_duration) {
            g_demo_water_start = g_demo_water_target;
            g_demo_water_target = generateRandomWaterLevel();
            g_demo_water_interp_start_time = currentFrameTime;
            elapsed = 0.0;
        }
        float t = static_cast<float>(glm::clamp(elapsed / g_demo_water_interp_duration, 0.0, 1.0));
        terrain_scene.waterLevel = glm::mix(g_demo_water_start, g_demo_water_target, t);
    }

    if (g_chunk_view_radius_changed) {
        g_chunk_view_radius_changed = false;
        terrain_scene.chunkManager.viewRadius = g_chunk_view_radius;
    }
}

void updateWaterChunks(VkDevice vk_device, WaterScene& scene, const ChunkManager& chunkManager) {
    // Create a water quad for any newly-loaded terrain chunk.
    for (auto& entry : chunkManager.loadedChunks) {
        const ChunkCoord& coord = entry.first;
        if (scene.chunkGeometry.count(coord)) continue;
        scene.chunkGeometry[coord] = buildWaterChunkGeometry(coord, chunkManager.baseParams);
    }

    // Destroy the water quad for any chunk that's no longer loaded. Wait for the GPU to finish first
    // — a previous frame's command buffer may still be referencing these buffers.
    bool destroyed_any = false;
    for (auto it = scene.chunkGeometry.begin(); it != scene.chunkGeometry.end();) {
        if (!chunkManager.loadedChunks.count(it->first)) {
            if (!destroyed_any) {
                vkDeviceWaitIdle(vk_device);
                destroyed_any = true;
            }
            destroyWaterChunkGeometry(it->second);
            it = scene.chunkGeometry.erase(it);
        } else {
            ++it;
        }
    }
}

void applyCameraInput(
    GLFWwindow* window,
    Camera* activeCamera,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera,
    double mouse_x,
    double mouse_y,
    double& mouse_x_last,
    double& mouse_y_last,
    float dt
) {
    float delta_x = mouse_x - mouse_x_last;
    float delta_y = mouse_y - mouse_y_last;
    float yawDelta = delta_x * g_mouse_sensitivity;
    float pitchDelta = -delta_y * g_mouse_sensitivity;
    if (g_toggle_camera || g_dragging) activeCamera->rotate((g_toggle_camera ? -1 : 1) * yawDelta, pitchDelta);

    bool shiftHeld = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
    activeCamera->setSpeed(g_camera_speed * (shiftHeld ? 2.0f : 1.0f));

    if (g_strafing && !g_toggle_camera) {
        glm::vec3 right = trackballCamera.getRight();
        glm::vec3 camUp = trackballCamera.getUp();
        glm::vec3 worldDelta = (-delta_x * right + delta_y * camUp) * trackballCamera.kPanSensitivity;
        trackballCamera.translate(worldDelta);
    }

    if (!g_toggle_camera && g_scroll_delta != 0.0f) {
        trackballCamera.zoom(g_scroll_delta * g_scroll_sensitivity);
    }
    g_scroll_delta = 0.0f;

    if (g_toggle_camera && !ImGui::GetIO().WantCaptureKeyboard) {
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) flyCamera.moveForward(dt);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) flyCamera.moveBackward(dt);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) flyCamera.moveLeft(dt);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) flyCamera.moveRight(dt);
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) flyCamera.moveUp(dt);
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) flyCamera.moveDown(dt);
    }

    mouse_x_last = mouse_x;
    mouse_y_last = mouse_y;
}

void labelThenRightAlignedWidget(const char* label, float widget_width) {
    ImGui::Spacing();
    float right_edge_x = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", label);
    ImGui::SameLine();
    ImGui::SetCursorPosX(right_edge_x - widget_width);
    ImGui::SetNextItemWidth(widget_width);
}

void buildGUI(TerrainScene& scene, const glm::vec3& cameraPosition, const glm::vec3& cameraForward) {
    const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    const ImGuiSliderFlags flags_for_sliders = (flags & ~ImGuiSliderFlags_WrapAround);

    if (!ImGui::Begin("Terrain Settings", &g_panel_open, window_flags)) {
        ImGui::End();
        return;
    }

    size_t generating_count = chunksStillGenerating(scene.chunkManager);
    bool is_regenerating = isRegenerating(scene.chunkManager);
    std::string generation_status_string = generating_count > 0 ? "Generating (" + std::to_string(generating_count) + " chunks)"
                                           : is_regenerating    ? "Blending..."
                                                                : "Generated";
    const char* generation_status_text = generation_status_string.c_str();
    float generation_status_offset = (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(generation_status_text).x) * 0.5f;
    if (generation_status_offset > 0.0f) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + generation_status_offset);
    }
    ImGui::Text("%s", generation_status_text);

    ImGui::BeginDisabled(is_regenerating);
    labelThenRightAlignedWidget("Hurst Exponent", kSliderWidth);
    ImGui::SliderFloat("##hurst", &g_hurst, 0.0f, 1.0f, "%f", flags_for_sliders);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        g_hurst_changed = true;
    }
    ImGui::EndDisabled();

    labelThenRightAlignedWidget("Height Scale", kSliderWidth);
    ImGui::SliderFloat("##heightScale", &scene.heightScale, 0.1f, 5.0f, "%f", flags_for_sliders);

    labelThenRightAlignedWidget("Roughness", kSliderWidth);
    ImGui::SliderFloat("##roughness", &scene.roughness, 0.0f, 1.0f, "%f", flags_for_sliders);

    labelThenRightAlignedWidget("Water level", kSliderWidth);
    ImGui::SliderFloat("##waterLevel", &scene.waterLevel, -25.0f, 25.0f, "%f", flags_for_sliders);

    labelThenRightAlignedWidget("Camera Speed", kSliderWidth);
    ImGui::SliderFloat("##cameraSpeed", &g_camera_speed, 1.0f, 100.0f, "%f", flags_for_sliders);

    labelThenRightAlignedWidget("View Radius", kSliderWidth);
    ImGui::SliderInt("##viewRadius", &g_chunk_view_radius, 1, 32, "%d", flags_for_sliders);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        g_chunk_view_radius_changed = true;
    }

    std::string seed_label = "Current seed: " + std::to_string(scene.chunkManager.baseParams.seed);
    float reseed_button_width = ImGui::CalcTextSize("Reseed").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    labelThenRightAlignedWidget(seed_label.c_str(), reseed_button_width);
    ImGui::BeginDisabled(is_regenerating);
    if (ImGui::Button("Reseed")) {
        g_reseed_requested = true;
    }
    ImGui::EndDisabled();

    ImGui::Text("Camera Mode: %s", g_toggle_camera ? "Fly" : "Trackball");
    ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", cameraPosition.x, cameraPosition.y, cameraPosition.z);
    ImGui::Text("Camera Forward:  (%.2f, %.2f, %.2f)", cameraForward.x, cameraForward.y, cameraForward.z);

    ImGui::Separator();
    ImGui::Text("Trackball Camera:");
    ImGui::Text("Left-click drag: Orbit camera");
    ImGui::Text("Right-click drag: Pan camera");
    ImGui::Text("Scroll: Zoom in/out");

    ImGui::Separator();
    ImGui::Text("Fly Camera:");
    ImGui::Text("W: Move forward");
    ImGui::Text("S: Move backward");
    ImGui::Text("A: Strafe left");
    ImGui::Text("D: Strafe right");
    ImGui::Text("Space: Move up");
    ImGui::Text("Ctrl: Move down");

    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::Text("C: Toggle fly/trackball camera");
    ImGui::Text("F1: Toggle wireframe mode");
    ImGui::Text("F2: Cycle face culling mode");
    ImGui::Text("F3: Toggle chunk-boundary debug overlay");
    ImGui::Text("F5: Hot-reload shaders");
    ImGui::Text("N: Toggle normals debug view");
    ImGui::Text("R: Reseed terrain");
    ImGui::Text("Esc: Quit application");
    ImGui::Text("Shift: Double camera speed");
    std::string demo_label = std::string("F4: Demo mode: ") + (g_demo_mode ? "ON" : "OFF");
    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "%s", demo_label.c_str());

    ImGui::End();
}

void drawGeometryWithMaterial(
    VkPipeline pipeline,
    const LoadedChunk& chunk,
    VkBuffer indices_buffer,
    uint32_t number_of_indices,
    VkDescriptorSet material,
    uint32_t num_instances
) {
    /* --------------------------------------------- */
    // Command Buffer Recording
    /* --------------------------------------------- */

    // Get the current command buffer:
    VkCommandBuffer cb = vklGetCurrentCommandBuffer();

    // Record binding the descriptor set for subsequent draw calls into the command buffer:
    VkPipelineLayout pipeline_layout = vklGetLayoutForPipeline(pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0u, 1u, &material, 0u, nullptr);

    vklCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    // Not blending: `from` is empty, so substitute `to` in its place.
    bool is_blending = chunk.from.vertexBuffer != VK_NULL_HANDLE;
    const Geometry& geometry_from = is_blending ? chunk.from : chunk.to;
    // Positions and normals share one combined buffer per Geometry — the same VkBuffer is bound
    // twice here, once per offset, which Vulkan allows.
    VkBuffer vertex_buffers[4] = {geometry_from.vertexBuffer, chunk.to.vertexBuffer, geometry_from.vertexBuffer, chunk.to.vertexBuffer};
    VkDeviceSize offsets[4] = {0, 0, geometry_from.normalsOffset, chunk.to.normalsOffset};
    vkCmdBindVertexBuffers(cb, 0u, 4u, vertex_buffers, offsets);

    vkCmdBindIndexBuffer(cb, indices_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, number_of_indices, num_instances, 0u, 0u, 0u);
}

void updateAndDrawTerrainScene(VkDevice vk_device, TerrainScene& scene, const Camera* camera, double currentTime) {
    // Shared, scene-level uniforms: identical for every chunk, since chunk position is already
    // baked into each chunk's own vertex data.
    UniformBufferVert ub_vert_data;
    ub_vert_data.modelMatrix = glm::scale(glm::mat4{1.0f}, glm::vec3(1.0f, 1.0f, scene.heightScale));
    ub_vert_data.modelMatrixForNormals = glm::scale(glm::mat4{1.0f}, glm::vec3(1.0f, 1.0f, 1 / scene.heightScale));
    ub_vert_data.viewProjMatrix = camera->getViewProjectionMatrix();
    vklCopyDataIntoHostCoherentBuffer(scene.ub_terrain_vert, &ub_vert_data, sizeof(UniformBufferVert));

    UniformBufferFrag ub_frag_data;
    ub_frag_data.cameraPosition = glm::vec4{camera->getPosition(), 1.0f};
    ub_frag_data.materialProperties = {g_terrain_ka, g_terrain_kd, g_terrain_ks, g_terrain_alpha};
    ub_frag_data.debugToggles = glm::uvec2{g_draw_normals ? 1u : 0u, g_highlight_chunk_borders ? 1u : 0u};
    ub_frag_data.isUnderwater = camera->getPosition().z < scene.waterLevel * scene.heightScale ? 1 : 0;
    int gridSize = (1 << scene.chunkManager.baseParams.gridSizeExponent) + 1;
    ub_frag_data.chunkWidth = static_cast<float>((gridSize - 1) * scene.chunkManager.baseParams.spacing);
    ub_frag_data.roughness = scene.roughness;
    ub_frag_data.dirtToGrassHeight = scene.waterLevel + g_dirt_to_grass_height_offset;
    ub_frag_data.grassToRockHeight = scene.waterLevel + g_grass_to_rock_height_offset;
    ub_frag_data.heightColorTransitionBand = g_height_color_transition_band;
    vklCopyDataIntoHostCoherentBuffer(scene.ub_terrain_frag, &ub_frag_data, sizeof(UniformBufferFrag));

    VkPipeline& selected_pipeline = scene.pipelines[g_polygon_mode_index][g_culling_index];
    if (selected_pipeline == VK_NULL_HANDLE) {
        selected_pipeline = buildTerrainPipeline(scene, g_polygon_mode_index, g_culling_index);
    }

    VkCommandBuffer cb = vklGetCurrentCommandBuffer();
    VkPipelineLayout pipeline_layout = vklGetLayoutForPipeline(selected_pipeline);

    // One draw call per loaded chunk, all against the same pipeline/descriptor set above — only
    // the vertex/index buffers and the push-constant blend state differ per chunk.
    for (auto& entry : scene.chunkManager.loadedChunks) {
        const LoadedChunk& chunk = entry.second;
        bool is_blending = chunk.from.vertexBuffer != VK_NULL_HANDLE;
        float blend_factor =
            is_blending ? glm::clamp(static_cast<float>((currentTime - chunk.blendStartTime) / scene.chunkManager.blendDuration), 0.0f, 1.0f) : 1.0f;

        TerrainPushConstants push_constants{blend_factor, is_blending ? 1u : 0u};
        vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0u, sizeof(TerrainPushConstants), &push_constants);

        drawGeometryWithMaterial(
            selected_pipeline,
            chunk,
            scene.chunkManager.sharedIndicesBuffer,
            scene.chunkManager.sharedNumberOfIndices,
            scene.ds_terrain
        );
    }
}

void updateAndDrawWaterScene(WaterScene& scene, const TerrainScene& terrain_scene, const Camera* camera) {
    UniformBufferWaterVert ub_data;
    ub_data.modelMatrix =
        glm::translate(glm::mat4{1.0f}, glm::vec3(0.0f, 0.0f, terrain_scene.waterLevel * terrain_scene.heightScale + g_water_depth_bias));
    ub_data.viewProjMatrix = camera->getViewProjectionMatrix();
    vklCopyDataIntoHostCoherentBuffer(scene.ub_water_vert, &ub_data, sizeof(UniformBufferWaterVert));

    VkCommandBuffer cb = vklGetCurrentCommandBuffer();
    VkPipelineLayout pipeline_layout = vklGetLayoutForPipeline(scene.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0u, 1u, &scene.ds_water, 0u, nullptr);
    vklCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline);

    // One draw call per loaded chunk's water tile, all against the same pipeline/descriptor set
    // above — only the vertex/index buffers differ per chunk.
    for (auto& entry : scene.chunkGeometry) {
        const WaterChunkGeometry& geometry = entry.second;
        VkBuffer vertex_buffers[1] = {geometry.positionsBuffer};
        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(cb, 0u, 1u, vertex_buffers, offsets);
        vkCmdBindIndexBuffer(cb, geometry.indicesBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, geometry.numberOfIndices, 1u, 0u, 0u, 0u);
    }
}

void cleanupTerrainScene(VkDevice vk_device, TerrainScene& scene) {
    vkDestroyDescriptorSetLayout(vk_device, scene.descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, scene.descriptor_pool, nullptr);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_terrain_vert);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_terrain_frag);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_dirlight);

    for (size_t i = 0; i < POLYMODES; ++i) {
        for (size_t j = 0; j < CULLMODES; ++j) {
            vklDestroyGraphicsPipeline(scene.pipelines[i][j]);
        }
    }
    cleanupChunkManager(scene.chunkManager);
}

void cleanupWaterScene(VkDevice vk_device, WaterScene& scene) {
    for (auto& entry : scene.chunkGeometry) {
        destroyWaterChunkGeometry(entry.second);
    }
    vkDestroyDescriptorSetLayout(vk_device, scene.descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, scene.descriptor_pool, nullptr);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_water_vert);
    vklDestroyGraphicsPipeline(scene.pipeline);
}