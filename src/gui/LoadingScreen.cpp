#include "LoadingScreen.h"

#include <VulkanLaunchpad.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <GLFW/glfw3.h>

size_t chunksStillGenerating(const ChunkManager& chunkManager) {
    return chunkManager.pendingChunks.size() + chunkManager.readyForNormals.size() + chunkManager.pendingNormals.size();
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
