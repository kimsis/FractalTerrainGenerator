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

#include "algorithms/DiamondSquareGenerator.h"
#include "Camera/Camera.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "lighting/SceneLighting.h"
#include "terrain/ChunkCoord.h"
#include "terrain/ChunkManager.h"
#include "terrain/TerrainGeometry.h"
#include "terrain/TerrainScene.h"
#include "utils/AppSettings.h"
#include "utils/INIReader.h"
#include "utils/PathUtils.h"
#include "utils/VulkanSetup.h"
#include "water/WaterGeometry.h"
#include "water/WaterScene.h"

#undef min
#undef max

/* ------------------------------------------------ */
// Constants
/* ------------------------------------------------ */
constexpr char WELCOME_MSG[] = ":::::: WELCOME TO MY TERRAIN GENERATOR ::::::";
constexpr char WINDOW_TITLE[] = "Fractal Terrain Generator";

/*! Fixed width every GUI slider is drawn at, used by labelThenRightAlignedWidget. */
constexpr float kSliderWidth = 200.0f;

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

/* --------------------------------------------- */
// Build the terrain pipeline for given polygon and cull modes
/* --------------------------------------------- */

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

