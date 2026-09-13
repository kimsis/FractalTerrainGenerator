/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <vector>
#include <vulkan/vulkan.hpp> // for vk::OutOfDateKHRError, thrown by the framework's internal vulkan-hpp calls on resize

#include "Camera/Camera.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "terrain/ChunkManager.h"
#include "terrain/DiamondSquareGenerator.h"
#include "terrain/Geometry.h"
#include "utils/PathUtils.h"
#include "utils/Utils.h"

#undef min
#undef max

constexpr char WELCOME_MSG[] = ":::::: WELCOME TO GCG 2025 ::::::";
constexpr char WINDOW_TITLE[] = "GCG 2025";

constexpr float BACKGROUND_R = 0.14f;
constexpr float BACKGROUND_G = 0.4f;
constexpr float BACKGROUND_B = 0.37f;

constexpr glm::vec4 DIRLIGHT_COLOR = glm::vec4(0.85f, 0.85f, 0.85f, 0.0f);
constexpr glm::vec4 DIRLIGHT_DIR = glm::vec4(0.0f, 1.0f, -1.0f, 0.0f);

constexpr float CORNELL_KA = 0.1f;
constexpr float CORNELL_KD = 0.9f;
constexpr float CORNELL_KS = 0.3f;
constexpr float CORNELL_ALPHA = 10.0f;

constexpr size_t POLYMODES = 2;
constexpr size_t CULLMODES = 3;
constexpr VkPolygonMode kTerrainPolygonModes[POLYMODES] = {VK_POLYGON_MODE_FILL, VK_POLYGON_MODE_LINE};
constexpr VkCullModeFlags kTerrainCullModes[CULLMODES] = {VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT, VK_CULL_MODE_FRONT_BIT};

/*! Fixed width every GUI slider is drawn at, used by labelThenRightAlignedWidget. */
constexpr float kSliderWidth = 200.0f;

static ImGuiSliderFlags flags = ImGuiSliderFlags_None;
ImGuiWindowFlags window_flags = 0;
bool g_panel_open = true;

/* --------------------------------------------- */
// Helper Function Declarations
/* --------------------------------------------- */
/*!
 *	This callback function gets invoked by GLFW whenever a GLFW error occured.
 */
void errorCallbackFromGlfw(int error, const char* description);

/*!
 *	Function that is invoked by GLFW to handle key events like key presses or key releases.
 *	If the ESC key has been pressed, the window will be marked that it should close.
 */
void handleGlfwKeyCallback(GLFWwindow* glfw_window, int key, int scancode, int action, int mods);

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
 *	Based on the given physical device and the surface, select a queue family which supports both,
 *	graphics and presentation to the given surface. Return the INDEX of an appropriate queue family!
 *	@return		The index of a queue family which supports the required features shall be returned.
 */
uint32_t selectQueueFamilyIndex(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Add extension_name to the target vector ref_vector if the extension is supported on this system.
 *	@param	extension_name		The instance extension that shall be added to ref_vector
 *	@param	ref_vector			Reference to the vector that the instance extension shall be added to, if supported
 */
void addInstanceExtensionToVectorIfSupported(const char* extension_name, std::vector<const char*>& ref_vector);

/*!
 *	Add validation_layer_name to the target vector ref_vector if the validation layer is supported on this system:
 *	@param	validation_layer_name	The validation layer name that shall be added to ref_vector
 *	@param	ref_vector				Reference to the vector that the validation layer name shall be added to, if
 *supported
 */
void addValidationLayerNameToVectorIfSupported(const char* validation_layer_name, std::vector<const char*>& ref_vector);

/*!
 *	Add extension_name to the target vector ref_vector if the extension is supported by the given physical device.
 *	@param	extension_name		The device extension that shall be added to ref_vector
 *	@param	physical_device		The physical device handle which must support the given extension
 *	@param	ref_vector			Reference to the vector that the device extension shall be added to, if supported
 */
void addDeviceExtensionToVectorIfSupported(const char* extension_name, VkPhysicalDevice physical_device, std::vector<const char*>& ref_vector);

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
 *	Based on the given physical device and the surface, a the physical device's surface capabilites are read and returned.
 *	@return		VkSurfaceCapabilitiesKHR data
 */
VkSurfaceCapabilitiesKHR getPhysicalDeviceSurfaceCapabilities(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Based on the given physical device and the surface, a supported surface image format
 *	which can be used for the framebuffer's attachment formats is searched and returned.
 *	@return		A supported format is returned.
 */
VkSurfaceFormatKHR getSurfaceImageFormat(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

/*!
 *	Based on the given physical device and the surface, return its surface transform flag.
 *	This can be used to set the swap chain to the same configuration as the surface's current transform.
 *	@return		The surface capabilities' currentTransform value is returned, which is suitable for swap chain config.
 */
VkSurfaceTransformFlagBitsKHR getSurfaceTransform(VkPhysicalDevice physical_device, VkSurfaceKHR surface);

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

    /*! 0-1 float for the smooth transition between hurst/seed changes */
    float blendFactor;

    /*! isBlend toggle, to avoid degenerate mixing of the same geometry in shaders. */
    uint32_t isBlending;
};

struct UniformBufferFrag {
    /*! Storage for the camera's world space position (aligned to 16 bytes) */
    glm::vec4 cameraPosition;

    /*! Illumination properties ka, kd, ks, alpha (in that order)
     *	First three are material coefficients, the last one is specular alpha. */
    glm::vec4 materialProperties;

    /*! Debug toggle for the normal-visualization branch in terrain.frag. */
    uint32_t drawNormals;

    /*! Whether the camera is currently below the water plane; drives the underwater tint in
     *  terrain.frag. A 4-byte type, not a native bool, for the same std140 reason as drawNormals. */
    uint32_t isUnderwater;
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
    Geometry terrain_geometry_from;
    Geometry terrain_geometry_to;
    TerrainParams terrainParams;
    float heightScale;
    float waterLevel;

    float blendStartTime;
    float blendDuration = 1.0f;

    std::future<GeometryData> pendingTerrainGeneration;
};

/*!
 * A raycast hit result.
 */
struct Hit {
    glm::vec3 point;
    float distance;
};

/*!
 *	Holds every GPU resource that makes up the water plane: a single flat quad, built once at
 *	startup and never regenerated, since its footprint (the terrain's fixed XY extent) never changes.
 */
struct WaterScene {
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    VkPipeline pipeline;
    std::string vertexShaderPath;
    std::string fragmentShaderPath;

    VkBuffer positionsBuffer;
    VkBuffer indicesBuffer;
    uint32_t numberOfIndices;

    VkBuffer ub_water_vert;
    VkDescriptorSet ds_water;
};

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
 *	Recreates the swapchain, depth buffer, and every pipeline against the window's current
 *	framebuffer size. Blocks (via glfwWaitEvents()) while the window is minimized. Takes mutable
 *	references to the Vulkan objects and cameras it needs to replace/update in place, since they
 *	live as locals in main().
 */
void recreateSwapchainAndDependents(
    GLFWwindow* window,
    VkInstance vk_instance,
    VkPhysicalDevice vk_physical_device,
    VkDevice vk_device,
    VkQueue vk_queue,
    uint32_t selected_queue_family_index,
    VkSurfaceKHR vk_surface,
    VkSurfaceFormatKHR surface_format,
    bool depthtest,
    VkClearValue color_clear_value,
    VkClearValue depth_clear_value,
    VkSwapchainKHR& vk_swapchain,
    VkImage& depth_buffer,
    std::vector<VkImage>& swapchain_image_handles,
    int& window_width,
    int& window_height,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera
);

/*!
 *	Bind the given descriptor se to use the material it represents for subsequent draw calls
 *	with the given pipeline, and render the given geometry (using its vertex and index buffers).
 *	Record everything into the current command buffer as provided by the framework.
 *	@param	pipeline		Valid handle to a given pipeline which shall be used for drawing.
 *	@param	geometry_from		Reference to a geometry_from object containing the buffers to be used for drawing.
 *	@param	material		Valid handle to a descriptor set that refers to resources that contain material properties.
 *	@param	num_instances	How many instances to draw of the given geometry. Default = one single instance.
 */
void drawGeometryWithMaterial(
    VkPipeline pipeline,
    const Geometry& geometry_from,
    const Geometry& geometry_to,
    VkDescriptorSet material,
    uint32_t num_instances = 1u
);

/*!
 *	Builds (compiles + creates) the terrain pipeline for one (polygon mode, cull mode) combination,
 *	identified by their indices into kTerrainPolygonModes/kTerrainCullModes. Shader compilation isn't
 *	cached by the framework, so this is deliberately only called once per combination, on demand.
 */
VkPipeline buildTerrainPipeline(const TerrainScene& scene, size_t polygon_mode_index, size_t cull_mode_index);

/*!
 *	Creates every pipeline, geometry, uniform buffer, descriptor set, and texture that the terrain
 *	scene consists of. Takes already-generated terrain geometry data rather than generating it itself,
 *	so the (potentially slow) CPU generation can happen elsewhere — e.g. on a background thread while
 *	a loading screen keeps the window responsive — before this is called.
 */
TerrainScene setupTerrainScene(
    VkDevice vk_device,
    VkQueue vk_queue,
    uint32_t selected_queue_family_index,
    const GeometryData& terrain_geometry_data,
    TerrainParams& params
);

/*!
 *	Builds a minimal ImGui panel shown while terrain is generating in the background, before the
 *	real terrain scene (and its controls) exist yet.
 */
void buildLoadingGUI();

/*!
 *	Kicks off terrain generation for the given params on a background thread and returns immediately
 *	with a future for the result — does not block, does not show any loading UI. Use this for silent
 *	background regeneration (e.g. after a Hurst change), where the currently-displayed terrain should
 *	keep rendering normally until the new one is ready to swap in.
 */
std::future<GeometryData> startTerrainGeneration(const TerrainParams& params);

/*!
 *	Generates terrain geometry for the given params, blocking the caller until it's done, while keeping
 *	the window responsive (polling events and drawing a loading screen) for however long that takes.
 *	Built on top of startTerrainGeneration — use this specifically when there's nothing else to show
 *	yet (e.g. the very first, initial generation at startup).
 */
GeometryData generateTerrainGeometryWithLoadingScreen(const TerrainParams& params);

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
 *	Draws a fresh uint32_t seed from a properly-seeded std::mt19937, spanning the full uint32_t range.
 *	Not rand(): rand()'s range is only guaranteed to be at least [0, 32767], it's implicitly-seeded
 *	global state (producing the same sequence every run unless srand() is called), and has well-known
 *	statistical weaknesses in its low-order bits.
 */
uint32_t generateRandomSeed();

/*!
 * Builds the ImGUI Sidebar
 */
void buildGUI(TerrainScene& scene, const glm::vec3& cameraPosition, const glm::vec3& cameraForward);

/*!
 *	Updates the terrain scene's uniform buffers based on the current camera, and records draw calls
 *	for it into the currently recording command buffer. Must be called between
 *	vklStartRecordingCommands() and vklEndRecordingCommands().
 */
void updateAndDrawTerrainScene(VkDevice vk_device, TerrainScene& scene, const Camera* camera);

/*!
 *	Destroys all GPU resources owned by the given terrain scene.
 */
void cleanupTerrainScene(VkDevice vk_device, TerrainScene& scene);
std::optional<Hit> raycastTerrain(const TerrainScene& terrainScene, const glm::vec3& origin, const glm::vec3& direction);

/*!
 *	Builds the water plane's single quad (spanning the terrain's fixed XY footprint at local z=0),
 *	pipeline, uniform buffer, and descriptor set.
 */
WaterScene setupWaterScene(VkDevice vk_device, const TerrainParams& terrain_params);

/*!
 *	Updates the water plane's uniform buffer (model matrix built from the terrain scene's current
 *	waterLevel/heightScale) and records its draw call. Must be called between
 *	vklStartRecordingCommands() and vklEndRecordingCommands(), after the terrain has been drawn.
 */
void updateAndDrawWaterScene(WaterScene& scene, const TerrainScene& terrain_scene, const Camera* camera);

/*!
 *	Destroys all GPU resources owned by the given water scene.
 */
void cleanupWaterScene(VkDevice vk_device, WaterScene& scene);

static bool g_dragging = false;
static bool g_strafing = false;
static float g_scroll_delta = 0.0f;
static bool g_framebuffer_resized = false;

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
static bool g_toggle_camera = false;
static bool g_toggle_camera_requested = false;
static bool g_reseed_requested = false;
static float g_camera_speed = 5.0f;
static int g_chunk_view_radius = 8;
static bool g_chunk_view_radius_changed = false;

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

int main(int argc, char** argv) {
    VKL_LOG(WELCOME_MSG);

    CMDLineArgs cmdline_args;
    gcgParseArgs(cmdline_args, argc, argv);

    /* --------------------------------------------- */
    // Load Settings From File
    /* --------------------------------------------- */

    int window_width = 800;
    int window_height = 800;
    std::string window_title = "Task 0";
    INIReader window_reader("assets/settings/window.ini");

    window_title = window_reader.Get("window", "title", WINDOW_TITLE);
    std::string init_camera_filepath = "assets/settings/camera_terrain.ini";
    if (cmdline_args.init_camera) {
        init_camera_filepath = cmdline_args.init_camera_filepath;
    }
    INIReader camera_reader(init_camera_filepath);

    float field_of_view = static_cast<float>(camera_reader.GetReal("camera", "fov", 60.0f));
    float near_plane_distance = static_cast<float>(camera_reader.GetReal("camera", "near", 0.1f));
    float far_plane_distance = static_cast<float>(camera_reader.GetReal("camera", "far", 100.0f));
    float aspect_ratio = static_cast<float>(window_width) / static_cast<float>(window_height);
    glm::vec3 camera_position(
        static_cast<float>(camera_reader.GetReal("camera", "position_x", 0.0f)),
        static_cast<float>(camera_reader.GetReal("camera", "position_y", 0.0f)),
        static_cast<float>(camera_reader.GetReal("camera", "position_z", 0.0f))
    );
    // Same convention as the live fly-camera controls: yaw/pitch = 0 looks along +X; positive
    // pitch looks up, negative looks down.
    float camera_yaw = static_cast<float>(camera_reader.GetReal("camera", "yaw", 0.0f));
    float camera_pitch = static_cast<float>(camera_reader.GetReal("camera", "pitch", 0.0f));
    std::string init_renderer_filepath = "assets/settings/renderer_standard.ini";
    if (cmdline_args.init_renderer) {
        init_renderer_filepath = cmdline_args.init_renderer_filepath;
    }
    INIReader renderer_reader(init_renderer_filepath);
    bool as_wireframe = renderer_reader.GetBoolean("renderer", "wireframe", false);
    if (as_wireframe) {
        g_polygon_mode_index = 1;
    }
    bool with_backface_culling = renderer_reader.GetBoolean("renderer", "backface_culling", false);
    if (with_backface_culling) {
        g_culling_index = 1;
    }
    g_draw_normals = renderer_reader.GetBoolean("renderer", "normals", false);
    bool depthtest = renderer_reader.GetBoolean("renderer", "depthtest", true);

    // Install a callback function, which gets invoked whenever a GLFW error occurred.
    glfwSetErrorCallback(errorCallbackFromGlfw);

    /* --------------------------------------------- */
    // Create a Window with GLFW
    /* --------------------------------------------- */
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
    window = glfwCreateWindow(window_width, window_height, window_title.c_str(), nullptr, nullptr);

    if (!window) {
        VKL_LOG("If your program reaches this point, that means two things:");
        VKL_LOG("1) Project setup was successful. Everything is working fine.");
        VKL_LOG("2) You haven't implemented Subtask 1.2, which is creating a window with GLFW.");
        VKL_EXIT_WITH_ERROR("No GLFW window created.");
    }
    glfwSetWindowPos(window, monitor_x, monitor_y);
    VKL_LOG("Subtask 1.2 done.");

    VkResult result;
    VkInstance vk_instance = VK_NULL_HANDLE;
    VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
    VkPhysicalDevice vk_physical_device = VK_NULL_HANDLE;
    VkDevice vk_device = VK_NULL_HANDLE;
    VkQueue vk_queue = VK_NULL_HANDLE;
    VkSwapchainKHR vk_swapchain = VK_NULL_HANDLE;

    /* --------------------------------------------- */
    // Create a Vulkan Instance
    /* --------------------------------------------- */
    VkApplicationInfo application_info = {};                     // Zero-initialize every member
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // Set this struct instance's type
    application_info.pEngineName = "GCG_VK_Library";             // Set some properties...
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 2023, 9, 1);
    application_info.pApplicationName = "GCG_VK_Solution";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 2023, 9, 19);
    application_info.apiVersion = VK_API_VERSION_1_1; // Your system needs to support this Vulkan API version.

    VkInstanceCreateInfo instance_create_info = {};                      // Zero-initialize every member
    instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; // Set the struct's type
    instance_create_info.pApplicationInfo = &application_info;

    // A vector to hold all our requested instance extensions:
    std::vector<const char*> instance_extensions = getRequiredInstanceExtensions();

    // Set info in the VkInstanceCreateInfo struct:
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

    // Create the instance
    result = vkCreateInstance(&instance_create_info, nullptr, &vk_instance);
    VKL_CHECK_VULKAN_RESULT(result);
    if (!vk_instance) {
        VKL_EXIT_WITH_ERROR("No VkInstance created or handle not assigned.");
    }
    VKL_LOG("Subtask 1.3 done.");

    /* --------------------------------------------- */
    // Create a Vulkan Window Surface
    /* --------------------------------------------- */
    result = glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface);
    VKL_CHECK_VULKAN_RESULT(result);
    if (!vk_surface) {
        VKL_EXIT_WITH_ERROR("No VkSurfaceKHR created or handle not assigned.");
    }
    VKL_LOG("Subtask 1.4 done.");

    /* --------------------------------------------- */
    // Pick a Physical Device
    /* --------------------------------------------- */
    // Query the number of physical devices:
    uint32_t physical_devices_count;
    vkEnumeratePhysicalDevices(vk_instance, &physical_devices_count, nullptr);

    if (physical_devices_count == 0) {
        VKL_EXIT_WITH_ERROR("Vulkan does not recognize any physical devices.");
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_devices_count);
    vkEnumeratePhysicalDevices(vk_instance, &physical_devices_count, physical_devices.data());

    uint32_t selected_physical_device_index = selectPhysicalDeviceIndex(physical_devices, vk_surface);
    vk_physical_device = physical_devices[selected_physical_device_index];
    if (!vk_physical_device) {
        VKL_EXIT_WITH_ERROR("No VkPhysicalDevice selected or handle not assigned.");
    }
    VKL_LOG("Subtask 1.5 done.");

    /* --------------------------------------------- */
    // Select a Queue Family
    /* --------------------------------------------- */
    std::array<float, 1> queue_priorities = {1.0f};

    uint32_t selected_queue_family_index = selectQueueFamilyIndex(vk_physical_device, vk_surface);
    VkDeviceQueueCreateInfo device_queue_create_info = {};
    device_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    device_queue_create_info.queueFamilyIndex = selected_queue_family_index;
    device_queue_create_info.queueCount = 1u;
    device_queue_create_info.pQueuePriorities = queue_priorities.data();

    // Sanity check if we have selected a valid queue family index:
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device, &queue_family_count, nullptr);
    if (selected_queue_family_index >= queue_family_count) {
        VKL_EXIT_WITH_ERROR("Invalid queue family index selected.");
    }
    VKL_LOG("Subtask 1.6 done.");

    /* --------------------------------------------- */
    // Create a Logical Device and Get Queue
    /* --------------------------------------------- */
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
    result = vkCreateDevice(vk_physical_device, &device_create_info, nullptr, &vk_device);
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
    vkGetDeviceQueue(vk_device, selected_queue_family_index, 0u, &vk_queue);
    if (!vk_queue) {
        VKL_EXIT_WITH_ERROR("No VkQueue selected or handle not assigned.");
    }
    VKL_LOG("Subtask 1.7 done.");

    /* --------------------------------------------- */
    // Create a Swapchain
    /* --------------------------------------------- */
    uint32_t queueFamilyIndexCount = 0u;
    std::vector<uint32_t> queueFamilyIndices;
    VkSurfaceFormatKHR surface_format = getSurfaceImageFormat(vk_physical_device, vk_surface);
    queueFamilyIndices.push_back(selected_queue_family_index);
    queueFamilyIndexCount = 1u;
    VkSurfaceCapabilitiesKHR surface_capabilities = getPhysicalDeviceSurfaceCapabilities(vk_physical_device, vk_surface);
    // Clamp to what the surface actually reports; the window manager doesn't always honor the
    // requested width/height exactly.
    if (surface_capabilities.currentExtent.width != UINT32_MAX) {
        window_width = static_cast<int>(surface_capabilities.currentExtent.width);
        window_height = static_cast<int>(surface_capabilities.currentExtent.height);
    } else {
        window_width = static_cast<int>(
            std::clamp(static_cast<uint32_t>(window_width), surface_capabilities.minImageExtent.width, surface_capabilities.maxImageExtent.width)
        );
        window_height = static_cast<int>(
            std::clamp(static_cast<uint32_t>(window_height), surface_capabilities.minImageExtent.height, surface_capabilities.maxImageExtent.height)
        );
    }
    aspect_ratio = static_cast<float>(window_width) / static_cast<float>(window_height);
    // Build the swapchain config struct:
    VkSwapchainCreateInfoKHR swapchain_create_info = {};
    swapchain_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_create_info.surface = vk_surface;
    swapchain_create_info.minImageCount = surface_capabilities.minImageCount;
    swapchain_create_info.imageArrayLayers = 1u;
    swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        swapchain_create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    } else {
        std::cout << "Warning: Automatic Testing might fail, VK_IMAGE_USAGE_TRANSFER_SRC_BIT image usage is not supported" << std::endl;
    }
    swapchain_create_info.preTransform = surface_capabilities.currentTransform;
    swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_create_info.clipped = VK_TRUE;
    swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_create_info.queueFamilyIndexCount = queueFamilyIndexCount;
    swapchain_create_info.pQueueFamilyIndices = queueFamilyIndices.data();
    swapchain_create_info.imageFormat = surface_format.format;
    swapchain_create_info.imageColorSpace = surface_format.colorSpace;
    swapchain_create_info.imageExtent = VkExtent2D{static_cast<uint32_t>(window_width), static_cast<uint32_t>(window_height)};
    swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;

    // Create the swapchain:
    result = vkCreateSwapchainKHR(vk_device, &swapchain_create_info, nullptr, &vk_swapchain);
    VKL_CHECK_VULKAN_RESULT(result);
    if (!vk_swapchain) {
        VKL_EXIT_WITH_ERROR("No VkSwapchainKHR created or handle not assigned.");
    }

    // Query how many swapchain images we got:
    uint32_t swapchain_image_count;
    vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &swapchain_image_count, nullptr);

    // Retrieve the swapchain images:
    std::vector<VkImage> swapchain_image_handles(swapchain_image_count);
    vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &swapchain_image_count, swapchain_image_handles.data());
    VKL_LOG("Subtask 1.8 done.");

    /* --------------------------------------------- */
    // Depth Test
    /* --------------------------------------------- */
    VkImage depth_buffer = vklCreateDeviceLocalImageWithBackingMemory(
        vk_physical_device,
        vk_device,
        window_width,
        window_height,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    );

    VkClearValue depth_clear_value;
    depth_clear_value.depthStencil.depth = 1.0f;
    depth_clear_value.depthStencil.stencil = 0u;

    /* --------------------------------------------- */
    // Init Framework
    /* --------------------------------------------- */

    // Gather swapchain config as required by the framework:
    VklSwapchainConfig swapchain_config = {};

    VkClearValue color_clear_value;
    color_clear_value.color.float32[0] = BACKGROUND_R;
    color_clear_value.color.float32[1] = BACKGROUND_G;
    color_clear_value.color.float32[2] = BACKGROUND_B;
    color_clear_value.color.float32[3] = 1.0f;

    swapchain_config.swapchainHandle = vk_swapchain;
    swapchain_config.imageExtent = swapchain_create_info.imageExtent;
    for (const VkImage& img : swapchain_image_handles) {
        VklSwapchainFramebufferComposition framebufferComposition;
        framebufferComposition.colorAttachmentImageDetails.imageHandle = img;
        framebufferComposition.colorAttachmentImageDetails.imageFormat = swapchain_create_info.imageFormat;
        framebufferComposition.colorAttachmentImageDetails.imageUsage = swapchain_create_info.imageUsage;
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

    // Init the framework:
    if (!vklInitFramework(vk_instance, vk_surface, vk_physical_device, vk_device, vk_queue, swapchain_config)) {
        VKL_EXIT_WITH_ERROR("Failed to init framework");
    }
    VKL_LOG("Subtask 1.9 done.");

    /* --------------------------------------------- */
    // Dear ImGui: init
    /* --------------------------------------------- */
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, /*install_callbacks=*/false);

    ImGui_ImplVulkan_InitInfo imgui_init_info = {};
    imgui_init_info.ApiVersion = application_info.apiVersion;
    imgui_init_info.Instance = vk_instance;
    imgui_init_info.PhysicalDevice = vk_physical_device;
    imgui_init_info.Device = vk_device;
    imgui_init_info.QueueFamily = selected_queue_family_index;
    imgui_init_info.Queue = vk_queue;
    imgui_init_info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
    imgui_init_info.MinImageCount = surface_capabilities.minImageCount;
    imgui_init_info.ImageCount = swapchain_image_count;
    imgui_init_info.PipelineInfoMain.RenderPass = vklGetRenderpass();
    imgui_init_info.PipelineInfoMain.Subpass = 0u;
    imgui_init_info.MinAllocationSize = 1024u * 1024u;
    ImGui_ImplVulkan_Init(&imgui_init_info);

    /* --------------------------------------------- */
    // Set up the Scene
    /* --------------------------------------------- */
    TerrainParams initial_terrain_params;
    GeometryData initial_terrain_geometry = generateTerrainGeometryWithLoadingScreen(initial_terrain_params);

    TerrainScene terrain_scene =
        setupTerrainScene(vk_device, vk_queue, selected_queue_family_index, initial_terrain_geometry, initial_terrain_params);

    int initial_terrain_size = (1 << initial_terrain_params.gridSizeExponent) + 1;
    terrain_scene.waterLevel = (initial_terrain_geometry.positions[0].z + initial_terrain_geometry.positions[initial_terrain_size - 1].z +
                                initial_terrain_geometry.positions[(initial_terrain_size - 1) * initial_terrain_size].z +
                                initial_terrain_geometry.positions[initial_terrain_size * initial_terrain_size - 1].z) /
                               4.0f;

    WaterScene water_scene = setupWaterScene(vk_device, initial_terrain_params);

    ChunkManager chunk_manager;
    chunk_manager.baseParams.gridSizeExponent = 4; // size=17 (16x16 cells), small/fast per chunk
    chunk_manager.baseParams.hurst = initial_terrain_params.hurst;
    chunk_manager.baseParams.seed = initial_terrain_params.seed;
    chunk_manager.baseParams.initialVariance = initial_terrain_params.initialVariance;
    chunk_manager.viewRadius = g_chunk_view_radius; // subsequent changes applied on slider release, below

    /* --------------------------------------------- */
    // Camera
    /* --------------------------------------------- */

    // Create a camera helper object, positioned/oriented per camera_terrain.ini. The trackball
    // camera's target is derived by raycasting the configured position/direction against the
    // terrain (same conversion used for live trackball<->fly switching further below), falling
    // back to the origin if that ray doesn't hit the terrain (e.g. looking up).
    FlyCamera flyCamera(field_of_view, aspect_ratio, near_plane_distance, far_plane_distance);
    flyCamera.translate(camera_position);
    flyCamera.rotate(glm::radians(camera_yaw), glm::radians(camera_pitch));
    auto initial_hit = raycastTerrain(terrain_scene, flyCamera.getPosition(), flyCamera.getForward());
    TrackballCamera trackballCamera(flyCamera, initial_hit.has_value() ? initial_hit->point : glm::vec3(0.0f, 0.0f, 0.0f));
    Camera* activeCamera = &trackballCamera;

    // Callback function for handling mouse button events:
    glfwSetMouseButtonCallback(window, mouseButtonCallbackFromGlfw);

    // Callback function for handling mouse scroll events:
    glfwSetScrollCallback(window, scrollCallbackFromGlfw);

    // Callback function for handling window resize events:
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallbackFromGlfw);

    /* --------------------------------------------- */
    // Set-up the Render Loop
    // Register a Key Callback
    /* --------------------------------------------- */

    glfwSetKeyCallback(window, handleGlfwKeyCallback);

    double mouse_x, mouse_x_last, mouse_y, mouse_y_last;
    glfwGetCursorPos(window, &mouse_x_last, &mouse_y_last);
    double lastFrameTime = glfwGetTime();

    vklEnablePipelineHotReloading(window, GLFW_KEY_F5);

    glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_TRUE);

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
                window,
                vk_instance,
                vk_physical_device,
                vk_device,
                vk_queue,
                selected_queue_family_index,
                vk_surface,
                surface_format,
                depthtest,
                color_clear_value,
                depth_clear_value,
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
        if (g_toggle_camera_requested) {
            g_toggle_camera_requested = false;
            if (!g_toggle_camera) {
                // Fly camera
                flyCamera = FlyCamera(trackballCamera);
                activeCamera = &flyCamera;
                g_toggle_camera = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            } else {
                // Trackball camera
                auto hit = raycastTerrain(terrain_scene, flyCamera.getPosition(), flyCamera.getForward());
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

        if (g_reseed_requested) {
            g_reseed_requested = false;
            if (!terrain_scene.pendingTerrainGeneration.valid()) {
                terrain_scene.terrainParams.seed = generateRandomSeed();
                terrain_scene.pendingTerrainGeneration = startTerrainGeneration(terrain_scene.terrainParams);
            }
        }

        if (g_chunk_view_radius_changed) {
            g_chunk_view_radius_changed = false;
            chunk_manager.viewRadius = g_chunk_view_radius;
        }
        updateLoadedChunks(chunk_manager, activeCamera->getPosition());
        {
            static size_t last_loaded = SIZE_MAX;
            static size_t last_pending = SIZE_MAX;
            if (chunk_manager.loadedChunks.size() != last_loaded || chunk_manager.pendingChunks.size() != last_pending) {
                last_loaded = chunk_manager.loadedChunks.size();
                last_pending = chunk_manager.pendingChunks.size();
                ChunkCoord cc = cameraToChunkCoord(activeCamera->getPosition(), chunk_manager.baseParams);
                glm::vec3 camPos = activeCamera->getPosition();
                fprintf(
                    stderr,
                    "DIAGNOSTIC: pos=(%.1f,%.1f,%.1f) camMode=%s cameraChunk=(%d,%d) loaded=%zu pending=%zu\n",
                    camPos.x,
                    camPos.y,
                    camPos.z,
                    g_toggle_camera ? "fly" : "trackball",
                    cc.cx,
                    cc.cy,
                    last_loaded,
                    last_pending
                );
            }
        }

        float delta_x = mouse_x - mouse_x_last;
        float delta_y = mouse_y - mouse_y_last;
        constexpr float kMouseSensitivity = 0.005f;
        float yawDelta = delta_x * kMouseSensitivity;
        float pitchDelta = -delta_y * kMouseSensitivity;
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
            trackballCamera.zoom(g_scroll_delta * trackballCamera.kScrollSensitivity);
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

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        // ImGui::ShowDemoWindow();
        buildGUI(terrain_scene, activeCamera->getPosition(), activeCamera->getForward());
        ImGui::Render();

        // Wait until we get an image from the swapchain to render into:
        try {
            vklWaitForNextSwapchainImage();
        } catch (const vk::OutOfDateKHRError&) {
            recreateSwapchainAndDependents(
                window,
                vk_instance,
                vk_physical_device,
                vk_device,
                vk_queue,
                selected_queue_family_index,
                vk_surface,
                surface_format,
                depthtest,
                color_clear_value,
                depth_clear_value,
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

        updateAndDrawTerrainScene(vk_device, terrain_scene, activeCamera);
        updateAndDrawWaterScene(water_scene, terrain_scene, activeCamera);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vklGetCurrentCommandBuffer());

        vklEndRecordingCommands();

        // Present rendered image to the screen:
        try {
            vklPresentCurrentSwapchainImage();
        } catch (const vk::OutOfDateKHRError&) {
            recreateSwapchainAndDependents(
                window,
                vk_instance,
                vk_physical_device,
                vk_device,
                vk_queue,
                selected_queue_family_index,
                vk_surface,
                surface_format,
                depthtest,
                color_clear_value,
                depth_clear_value,
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

        if (cmdline_args.run_headless) {
            uint32_t idx = vklGetCurrentSwapChainImageIndex();
            std::string screenshot_filename = "screenshot";
            if (cmdline_args.set_filename) {
                screenshot_filename = cmdline_args.filename;
            }
            gcgSaveScreenshot(
                screenshot_filename,
                swapchain_image_handles[idx],
                window_width,
                window_height,
                surface_format.format,
                vk_device,
                vk_physical_device,
                vk_queue,
                selected_queue_family_index
            );
            break;
        }
    }

    // Wait for all GPU work to finish before cleaning up:
    vkDeviceWaitIdle(vk_device);

    // Cleanup:
    vklDestroyDeviceLocalImageAndItsBackingMemory(depth_buffer);
    cleanupTerrainScene(vk_device, terrain_scene);
    cleanupWaterScene(vk_device, water_scene);
    cleanupChunkManager(chunk_manager);

    /* --------------------------------------------- */
    // Dear ImGui: shutdown
    /* --------------------------------------------- */
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    /* --------------------------------------------- */
    // Cleanup
    /* --------------------------------------------- */
    vklDestroyFramework();
    vkDestroySwapchainKHR(vk_device, vk_swapchain, nullptr);
    vkDestroyDevice(vk_device, nullptr);
    vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
    vkDestroyInstance(vk_instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}

/* --------------------------------------------- */
// Helper Function Definitions
/* --------------------------------------------- */

void errorCallbackFromGlfw(int error, const char* description) { std::cout << "GLFW error " << error << ": " << description << std::endl; }

void recreateSwapchainAndDependents(
    GLFWwindow* window,
    VkInstance vk_instance,
    VkPhysicalDevice vk_physical_device,
    VkDevice vk_device,
    VkQueue vk_queue,
    uint32_t selected_queue_family_index,
    VkSurfaceKHR vk_surface,
    VkSurfaceFormatKHR surface_format,
    bool depthtest,
    VkClearValue color_clear_value,
    VkClearValue depth_clear_value,
    VkSwapchainKHR& vk_swapchain,
    VkImage& depth_buffer,
    std::vector<VkImage>& swapchain_image_handles,
    int& window_width,
    int& window_height,
    TrackballCamera& trackballCamera,
    FlyCamera& flyCamera
) {
    int fb_width = 0, fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    while (fb_width == 0 || fb_height == 0) {
        // Minimized: block until the window is restored.
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(vk_device);

    vklDestroyDeviceLocalImageAndItsBackingMemory(depth_buffer);
    VkSwapchainKHR old_swapchain = vk_swapchain;
    vklDestroyFramework();

    VkSurfaceCapabilitiesKHR new_surface_capabilities = getPhysicalDeviceSurfaceCapabilities(vk_physical_device, vk_surface);
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
    new_swapchain_create_info.surface = vk_surface;
    new_swapchain_create_info.minImageCount = new_surface_capabilities.minImageCount;
    new_swapchain_create_info.imageArrayLayers = 1u;
    new_swapchain_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (new_surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        new_swapchain_create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    new_swapchain_create_info.preTransform = new_surface_capabilities.currentTransform;
    new_swapchain_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    new_swapchain_create_info.clipped = VK_TRUE;
    new_swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    new_swapchain_create_info.queueFamilyIndexCount = 1u;
    new_swapchain_create_info.pQueueFamilyIndices = &selected_queue_family_index;
    new_swapchain_create_info.imageFormat = surface_format.format;
    new_swapchain_create_info.imageColorSpace = surface_format.colorSpace;
    new_swapchain_create_info.imageExtent = new_extent;
    new_swapchain_create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    new_swapchain_create_info.oldSwapchain = old_swapchain;

    VkResult swap_result = vkCreateSwapchainKHR(vk_device, &new_swapchain_create_info, nullptr, &vk_swapchain);
    VKL_CHECK_VULKAN_RESULT(swap_result);
    vkDestroySwapchainKHR(vk_device, old_swapchain, nullptr);

    uint32_t new_swapchain_image_count;
    vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &new_swapchain_image_count, nullptr);
    swapchain_image_handles.resize(new_swapchain_image_count);
    vkGetSwapchainImagesKHR(vk_device, vk_swapchain, &new_swapchain_image_count, swapchain_image_handles.data());

    depth_buffer = vklCreateDeviceLocalImageWithBackingMemory(
        vk_physical_device,
        vk_device,
        static_cast<int>(new_extent.width),
        static_cast<int>(new_extent.height),
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    );

    VklSwapchainConfig new_swapchain_config = {};
    new_swapchain_config.swapchainHandle = vk_swapchain;
    new_swapchain_config.imageExtent = new_extent;
    for (const VkImage& img : swapchain_image_handles) {
        VklSwapchainFramebufferComposition framebufferComposition;
        framebufferComposition.colorAttachmentImageDetails.imageHandle = img;
        framebufferComposition.colorAttachmentImageDetails.imageFormat = new_swapchain_create_info.imageFormat;
        framebufferComposition.colorAttachmentImageDetails.imageUsage = new_swapchain_create_info.imageUsage;
        framebufferComposition.colorAttachmentImageDetails.clearValue = color_clear_value;
        if (depthtest) {
            framebufferComposition.depthAttachmentImageDetails.imageHandle = depth_buffer;
            framebufferComposition.depthAttachmentImageDetails.imageFormat = VK_FORMAT_D32_SFLOAT;
            framebufferComposition.depthAttachmentImageDetails.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            framebufferComposition.depthAttachmentImageDetails.clearValue = depth_clear_value;
        }
        new_swapchain_config.swapchainImages.push_back(framebufferComposition);
    }

    if (!vklInitFramework(vk_instance, vk_surface, vk_physical_device, vk_device, vk_queue, new_swapchain_config)) {
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
    if (key == GLFW_KEY_N) {
        g_draw_normals = !g_draw_normals;
    }
    if (key == GLFW_KEY_C) {
        g_toggle_camera_requested = true;
    }
    if (key == GLFW_KEY_R) {
        g_reseed_requested = true;
    }
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

std::vector<const char*> getRequiredInstanceExtensions() {
    std::vector<const char*> required_extensions;

    // Query extensions which are required by GLFW, adding each one only if it is supported:
    uint32_t glfw_instance_extensions_count;
    const char** glfw_instance_extensions_names = glfwGetRequiredInstanceExtensions(&glfw_instance_extensions_count);
    for (uint32_t i = 0; i < glfw_instance_extensions_count; ++i) {
        addInstanceExtensionToVectorIfSupported(glfw_instance_extensions_names[i], required_extensions);
    }

    // Query extensions which are required by Vulkan Launchpad, adding each one only if it is supported:
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

void addInstanceExtensionToVectorIfSupported(const char* extension_name, std::vector<const char*>& ref_vector) {
    VkResult result;

    // Query how many instance extensions there are:
    uint32_t instance_extension_count;
    result = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, nullptr);
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    // Get all the instance extension names/properties there are:
    std::vector<VkExtensionProperties> available_instance_extensions(instance_extension_count);
    result = vkEnumerateInstanceExtensionProperties(nullptr, &instance_extension_count, available_instance_extensions.data());
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    for (const VkExtensionProperties& available_extension : available_instance_extensions) {
        if (strcmp(available_extension.extensionName, extension_name) == 0) {
            // Found the extension => Add it to the vector:
            ref_vector.push_back(extension_name);
            return;
        }
    }
}

void addValidationLayerNameToVectorIfSupported(const char* validation_layer_name, std::vector<const char*>& ref_vector) {
    VkResult result;

    // Query how many validation layers there are:
    uint32_t layer_count;
    result = vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    // Get all the validation layer names/properties there are:
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

void addDeviceExtensionToVectorIfSupported(const char* extension_name, VkPhysicalDevice physical_device, std::vector<const char*>& ref_vector) {
    VkResult result;

    // Query how many device extensions there are:
    uint32_t extensions_count;
    result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, nullptr);
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    // Get all the device extensions:
    std::vector<VkExtensionProperties> available_extensions(extensions_count);
    result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extensions_count, available_extensions.data());
    VKL_CHECK_VULKAN_ERROR(result);
    VKL_RETURN_ON_ERROR(result);

    for (const VkExtensionProperties& available_extension : available_extensions) {
        if (strcmp(available_extension.extensionName, extension_name) == 0) {
            // Found the extension => Add it to the vector:
            ref_vector.push_back(extension_name);
            return;
        }
    }
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

VkSurfaceTransformFlagBitsKHR getSurfaceTransform(VkPhysicalDevice physical_device, VkSurfaceKHR surface) {
    return getPhysicalDeviceSurfaceCapabilities(physical_device, surface).currentTransform;
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

void drawGeometryWithMaterial(
    VkPipeline pipeline,
    const Geometry& geometry_from,
    const Geometry& geometry_to,
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
    VkBuffer vertex_buffers[4] = {geometry_from.positionsBuffer, geometry_to.positionsBuffer, geometry_from.normalsBuffer, geometry_to.normalsBuffer};
    VkDeviceSize offsets[4] = {0, 0, 0, 0};
    vkCmdBindVertexBuffers(cb, 0u, 4u, vertex_buffers, offsets);

    vkCmdBindIndexBuffer(cb, geometry_to.indicesBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, geometry_to.numberOfIndices, num_instances, 0u, 0u, 0u);
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
    };
    return vklCreateGraphicsPipeline(pipeline_config);
}

TerrainScene setupTerrainScene(
    VkDevice vk_device,
    VkQueue vk_queue,
    uint32_t selected_queue_family_index,
    const GeometryData& terrain_geometry_data,
    TerrainParams& params
) {
    TerrainScene scene{};
    scene.terrainParams = params;
    scene.heightScale = 1.0f;

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
    DirectionalLight directional_light = {DIRLIGHT_COLOR, glm::normalize(DIRLIGHT_DIR)};
    vklCopyDataIntoHostCoherentBuffer(scene.ub_dirlight, &directional_light, sizeof(DirectionalLight));

    scene.terrain_geometry_to = createAndUploadIntoGpuMemory(terrain_geometry_data);
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

void updateAndDrawTerrainScene(VkDevice vk_device, TerrainScene& scene, const Camera* camera) {
    if (scene.pendingTerrainGeneration.valid() &&
        scene.pendingTerrainGeneration.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        scene.blendStartTime = glfwGetTime();
        GeometryData new_terrain_geometry_data = scene.pendingTerrainGeneration.get();
        vkDeviceWaitIdle(vk_device);

        int size = (1 << scene.terrainParams.gridSizeExponent) + 1;
        scene.waterLevel = (new_terrain_geometry_data.positions[0].z + new_terrain_geometry_data.positions[size - 1].z +
                            new_terrain_geometry_data.positions[(size - 1) * size].z + new_terrain_geometry_data.positions[size * size - 1].z) /
                           4.0f;

        if (scene.terrain_geometry_from.positionsBuffer != VK_NULL_HANDLE) {
            destroyGeometryGpuMemory(scene.terrain_geometry_from);
        }
        scene.terrain_geometry_from = scene.terrain_geometry_to;
        scene.terrain_geometry_to = createAndUploadIntoGpuMemory(new_terrain_geometry_data);
    }

    UniformBufferVert ub_vert_data;
    ub_vert_data.modelMatrix = glm::scale(glm::mat4{1.0f}, glm::vec3(1.0f, 1.0f, scene.heightScale));
    ub_vert_data.modelMatrixForNormals = glm::scale(glm::mat4{1.0f}, glm::vec3(1.0f, 1.0f, 1 / scene.heightScale));
    ub_vert_data.viewProjMatrix = camera->getViewProjectionMatrix();
    ub_vert_data.blendFactor = glm::clamp((static_cast<float>(glfwGetTime()) - scene.blendStartTime) / scene.blendDuration, 0.0f, 1.0f);
    if (scene.terrain_geometry_from.positionsBuffer != VK_NULL_HANDLE && ub_vert_data.blendFactor >= 1.0f) {
        destroyGeometryGpuMemory(scene.terrain_geometry_from);
        scene.terrain_geometry_from = Geometry{};
    }
    ub_vert_data.isBlending = scene.terrain_geometry_from.positionsBuffer != VK_NULL_HANDLE;
    vklCopyDataIntoHostCoherentBuffer(scene.ub_terrain_vert, &ub_vert_data, sizeof(UniformBufferVert));

    UniformBufferFrag ub_frag_data;
    ub_frag_data.cameraPosition = glm::vec4{camera->getPosition(), 1.0f};
    ub_frag_data.materialProperties = {CORNELL_KA, CORNELL_KD, CORNELL_KS, CORNELL_ALPHA};
    ub_frag_data.drawNormals = g_draw_normals ? 1 : 0;
    ub_frag_data.isUnderwater = camera->getPosition().z < scene.waterLevel * scene.heightScale ? 1 : 0;
    vklCopyDataIntoHostCoherentBuffer(scene.ub_terrain_frag, &ub_frag_data, sizeof(UniformBufferFrag));

    VkPipeline& selected_pipeline = scene.pipelines[g_polygon_mode_index][g_culling_index];
    if (selected_pipeline == VK_NULL_HANDLE) {
        selected_pipeline = buildTerrainPipeline(scene, g_polygon_mode_index, g_culling_index);
    }
    bool has_from = scene.terrain_geometry_from.positionsBuffer != VK_NULL_HANDLE;
    drawGeometryWithMaterial(
        selected_pipeline,
        has_from ? scene.terrain_geometry_from : scene.terrain_geometry_to,
        scene.terrain_geometry_to,
        scene.ds_terrain
    );
}

void cleanupTerrainScene(VkDevice vk_device, TerrainScene& scene) {
    vkDestroyDescriptorSetLayout(vk_device, scene.descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, scene.descriptor_pool, nullptr);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_terrain_vert);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_terrain_frag);
    if (scene.terrain_geometry_from.positionsBuffer != VK_NULL_HANDLE) {
        destroyGeometryGpuMemory(scene.terrain_geometry_from);
    }
    destroyGeometryGpuMemory(scene.terrain_geometry_to);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_dirlight);

    for (size_t i = 0; i < POLYMODES; ++i) {
        for (size_t j = 0; j < CULLMODES; ++j) {
            vklDestroyGraphicsPipeline(scene.pipelines[i][j]);
        }
    }
}

WaterScene setupWaterScene(VkDevice vk_device, const TerrainParams& terrain_params) {
    WaterScene scene{};

    int size = (1 << terrain_params.gridSizeExponent) + 1;
    float half_extent = (size / 2.0f) * terrain_params.spacing;
    std::vector<glm::vec3> positions = {
        {-half_extent, -half_extent, 0.0f},
        {half_extent, -half_extent, 0.0f},
        {half_extent, half_extent, 0.0f},
        {-half_extent, half_extent, 0.0f},
    };
    std::vector<uint32_t> indices = {0u, 1u, 2u, 0u, 2u, 3u};

    scene.positionsBuffer = vklCreateHostCoherentBufferAndUploadData(
        positions.data(),
        positions.size() * sizeof(glm::vec3),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    );
    scene.indicesBuffer = vklCreateHostCoherentBufferAndUploadData(
        indices.data(),
        indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    );
    scene.numberOfIndices = static_cast<uint32_t>(indices.size());

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

    return scene;
}

void updateAndDrawWaterScene(WaterScene& scene, const TerrainScene& terrain_scene, const Camera* camera) {
    UniformBufferWaterVert ub_data;
    ub_data.modelMatrix = glm::translate(glm::mat4{1.0f}, glm::vec3(0.0f, 0.0f, terrain_scene.waterLevel * terrain_scene.heightScale));
    ub_data.viewProjMatrix = camera->getViewProjectionMatrix();
    vklCopyDataIntoHostCoherentBuffer(scene.ub_water_vert, &ub_data, sizeof(UniformBufferWaterVert));

    VkCommandBuffer cb = vklGetCurrentCommandBuffer();
    VkPipelineLayout pipeline_layout = vklGetLayoutForPipeline(scene.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0u, 1u, &scene.ds_water, 0u, nullptr);

    vklCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, scene.pipeline);
    VkBuffer vertex_buffers[1] = {scene.positionsBuffer};
    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(cb, 0u, 1u, vertex_buffers, offsets);

    vkCmdBindIndexBuffer(cb, scene.indicesBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, scene.numberOfIndices, 1u, 0u, 0u, 0u);
}

void cleanupWaterScene(VkDevice vk_device, WaterScene& scene) {
    vkDestroyDescriptorSetLayout(vk_device, scene.descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(vk_device, scene.descriptor_pool, nullptr);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.ub_water_vert);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.positionsBuffer);
    vklDestroyHostCoherentBufferAndItsBackingMemory(scene.indicesBuffer);
    vklDestroyGraphicsPipeline(scene.pipeline);
}

void buildLoadingGUI() {
    const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::Begin(
        "Loading",
        nullptr,
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove
    );
    ImGui::Text("Generating terrain...");
    ImGui::End();
}

std::future<GeometryData> startTerrainGeneration(const TerrainParams& params) {
    return std::async(std::launch::async, generateTerrainGeometry, params);
}

GeometryData generateTerrainGeometryWithLoadingScreen(const TerrainParams& params) {
    std::future<GeometryData> terrain_geometry_future = startTerrainGeneration(params);

    while (terrain_geometry_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        glfwPollEvents();

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        buildLoadingGUI();
        ImGui::Render();

        vklWaitForNextSwapchainImage();
        vklStartRecordingCommands();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vklGetCurrentCommandBuffer());
        vklEndRecordingCommands();
        vklPresentCurrentSwapchainImage();
    }

    return terrain_geometry_future.get();
}

uint32_t generateRandomSeed() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_int_distribution<uint32_t> dist;
    return dist(rng);
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

    bool generation_in_progress = scene.pendingTerrainGeneration.valid();
    const char* generation_status_text = generation_in_progress ? "Generating" : "Generated";
    float generation_status_offset = (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(generation_status_text).x) * 0.5f;
    if (generation_status_offset > 0.0f) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + generation_status_offset);
    }
    ImGui::Text("%s", generation_status_text);

    labelThenRightAlignedWidget("Hurst Exponent", kSliderWidth);
    ImGui::BeginDisabled(generation_in_progress);
    ImGui::SliderFloat("##hurst", &scene.terrainParams.hurst, 0.0f, 1.0f, "%f", flags_for_sliders);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        scene.pendingTerrainGeneration = startTerrainGeneration(scene.terrainParams);
    }
    ImGui::EndDisabled();

    labelThenRightAlignedWidget("Height Scale", kSliderWidth);
    ImGui::SliderFloat("##heightScale", &scene.heightScale, 0.000001f, 5.0f, "%f", flags_for_sliders);

    labelThenRightAlignedWidget("Water level", kSliderWidth);
    ImGui::SliderFloat("##waterLevel", &scene.waterLevel, -25.0f, 25.0f, "%f", flags_for_sliders);

    labelThenRightAlignedWidget("Camera Speed", kSliderWidth);
    ImGui::SliderFloat("##cameraSpeed", &g_camera_speed, 1.0f, 25.0f, "%f", flags_for_sliders);

    labelThenRightAlignedWidget("View Radius", kSliderWidth);
    ImGui::SliderInt("##viewRadius", &g_chunk_view_radius, 1, 32, "%d", flags_for_sliders);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        g_chunk_view_radius_changed = true;
    }

    std::string seed_label = "Current seed: " + std::to_string(scene.terrainParams.seed);
    float reseed_button_width = ImGui::CalcTextSize("Reseed").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    labelThenRightAlignedWidget(seed_label.c_str(), reseed_button_width);
    ImGui::BeginDisabled(generation_in_progress);
    if (ImGui::Button("Reseed")) {
        scene.terrainParams.seed = generateRandomSeed();
        scene.pendingTerrainGeneration = startTerrainGeneration(scene.terrainParams);
    }
    ImGui::EndDisabled();

    ImGui::Text("Camera Mode: %s", g_toggle_camera ? "Fly" : "Trackball");
    ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", cameraPosition.x, cameraPosition.y, cameraPosition.z);
    ImGui::Text("Camera Forward:  (%.2f, %.2f, %.2f)", cameraForward.x, cameraForward.y, cameraForward.z);

    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::Text("C: Toggle fly/trackball camera");
    ImGui::Text("F1: Toggle wireframe mode");
    ImGui::Text("F2: Cycle face culling mode");
    ImGui::Text("N: Toggle normals debug view");
    ImGui::Text("R: Reseed terrain");
    ImGui::Text("Esc: Quit application");
    ImGui::Text("Shift: Double camera speed");

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

    ImGui::End();
}

std::optional<Hit> raycastTerrain(const TerrainScene& terrainScene, const glm::vec3& origin, const glm::vec3& direction) {
    constexpr float kGroundPlaneZ = 0.0f;
    if (std::abs(direction.z) < 1e-6f) return std::nullopt;
    float t = (kGroundPlaneZ - origin.z) / direction.z;
    if (t < 0.0f) return std::nullopt;
    glm::vec3 point = origin + t * direction;
    return Hit{point, t};
}
