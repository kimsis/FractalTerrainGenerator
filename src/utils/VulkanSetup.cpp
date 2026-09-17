#include "VulkanSetup.h"

#include <VulkanLaunchpad.h>

#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>

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

    std::vector<const char*> instance_extensions = getRequiredInstanceExtensions();
    instance_create_info.enabledExtensionCount = instance_extensions.size();
    instance_create_info.ppEnabledExtensionNames = instance_extensions.data();

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
    for (uint32_t physical_device_index = 0u; physical_device_index < physical_device_count; ++physical_device_index) {
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(physical_devices[physical_device_index], &features);
        if (VK_TRUE != features.fillModeNonSolid) {
            continue;
        }

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[physical_device_index], &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[physical_device_index], &queue_family_count, queue_families.data());

        for (uint32_t queue_family_index = 0u; queue_family_index < queue_family_count; ++queue_family_index) {
            if ((queue_families[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                VkBool32 presentation_supported;
                vkGetPhysicalDeviceSurfaceSupportKHR(physical_devices[physical_device_index], queue_family_index, surface, &presentation_supported);

                if (VK_TRUE == presentation_supported) {
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
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    for (uint32_t queue_family_index = 0u; queue_family_index < queue_family_count; ++queue_family_index) {
        if ((queue_families[queue_family_index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            VkBool32 presentation_supported;
            vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index, surface, &presentation_supported);

            if (VK_TRUE == presentation_supported) {
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
    static constexpr std::array<float, 1> QUEUE_PRIORITIES = {1.0f};

    VkDeviceQueueCreateInfo device_queue_create_info = {};
    device_queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    device_queue_create_info.queueFamilyIndex = selected_queue_family_index;
    device_queue_create_info.queueCount = 1u;
    device_queue_create_info.pQueuePriorities = QUEUE_PRIORITIES.data();

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
    bool synchronization2_supported = false;

    std::vector<const char*> device_extensions;
    addDeviceExtensionToVectorIfSupported(VK_KHR_SWAPCHAIN_EXTENSION_NAME, vk_physical_device, device_extensions);
    addDeviceExtensionToVectorIfSupported(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, vk_physical_device, device_extensions);
    // Required in addition to Synchronization2 since SDK 1.2.170.
    addDeviceExtensionToVectorIfSupported(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, vk_physical_device, device_extensions);
    if (std::find(std::begin(device_extensions), std::end(device_extensions), std::string(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) !=
        std::end(device_extensions)) {
        synchronization2_supported = true;
    }
#ifdef __APPLE__
    addDeviceExtensionToVectorIfSupported("VK_KHR_portability_subset", vk_physical_device, device_extensions);
#endif

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1u;
    device_create_info.pQueueCreateInfos = &device_queue_create_info;
    device_create_info.enabledExtensionCount = device_extensions.size();
    device_create_info.ppEnabledExtensionNames = device_extensions.data();

    VkPhysicalDeviceFeatures enabled_physical_device_features = {};
    enabled_physical_device_features.fillModeNonSolid = VK_TRUE;
    device_create_info.pEnabledFeatures = &enabled_physical_device_features;

    VkPhysicalDeviceSynchronization2FeaturesKHR physical_device_sync2_features = {};
    physical_device_sync2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
    physical_device_sync2_features.synchronization2 = VK_TRUE;
    if (synchronization2_supported) {
        device_create_info.pNext = &physical_device_sync2_features;
    }

    VkDevice vk_device = VK_NULL_HANDLE;
    VkResult result = vkCreateDevice(vk_physical_device, &device_create_info, nullptr, &vk_device);
    VKL_CHECK_VULKAN_RESULT(result);

    if (synchronization2_supported) {
        // Nothing in this app issues Synchronization2 barriers, so the resolved pointer itself isn't
        // kept — this just confirms the extension is actually usable, not just reported as present.
        auto* procAddr = vkGetDeviceProcAddr(vk_device, "vkCmdPipelineBarrier2KHR");
        if (procAddr == nullptr) {
            synchronization2_supported = false;
        }
    }
    if (!vk_device) {
        VKL_EXIT_WITH_ERROR("No VkDevice created or handle not assigned.");
    }

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

    // Prefer sRGB8; fall back to whatever's available.
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

    VkResult result = vkCreateSwapchainKHR(vk_device, &swapchain_create_info, nullptr, &setup.swapchain);
    VKL_CHECK_VULKAN_RESULT(result);
    if (!setup.swapchain) {
        VKL_EXIT_WITH_ERROR("No VkSwapchainKHR created or handle not assigned.");
    }
    setup.image_usage = swapchain_create_info.imageUsage;

    uint32_t swapchain_image_count;
    vkGetSwapchainImagesKHR(vk_device, setup.swapchain, &swapchain_image_count, nullptr);

    setup.image_handles.resize(swapchain_image_count);
    vkGetSwapchainImagesKHR(vk_device, setup.swapchain, &swapchain_image_count, setup.image_handles.data());

    return setup;
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

    vklHotReloadPipelines();

    window_width = static_cast<int>(new_extent.width);
    window_height = static_cast<int>(new_extent.height);
    float new_aspect_ratio = static_cast<float>(new_extent.width) / static_cast<float>(new_extent.height);
    trackballCamera.setAspectRatio(new_aspect_ratio);
    flyCamera.setAspectRatio(new_aspect_ratio);
}

VkDescriptorSet allocDescriptorSet(VkDevice device, VkDescriptorPool descriptor_pool, VkDescriptorSetLayout descriptor_set_layout) {
    VkResult result;

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
