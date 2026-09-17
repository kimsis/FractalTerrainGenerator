cmake_minimum_required(VERSION 3.14)
project(VulkanLaunchpadStarter)

#================================#
# Compiler                       #
#================================#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/VulkanLaunchpad/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

#================================#
# Local patches for VulkanLaunchpad #
#================================#
# VulkanLaunchpad is a third-party submodule we can't commit fixes to directly;
# local fixes are carried as patches (see patches/) and re-applied here so every
# checkout gets them without a manual step. Safe on repeat configures - already
# applied patches are detected and skipped by the script itself. Implemented as
# a CMake script (not a shell script) so it works identically on Windows/macOS/Linux.
include("${PROJECT_SOURCE_DIR}/patches/ApplyPatches.cmake")


#================================#
# VulkanLaunchpad (dependency)   #
#================================#
add_subdirectory(external/VulkanLaunchpad)

#================================#
# Dear ImGui (GUI dependency)    #
#================================#
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui")

#================================#
# VulkanLaunchpadStarter         #
#================================#
add_executable(${PROJECT_NAME}
    src/Main.cpp
    src/utils/AppSettings.h
    src/utils/VulkanSetup.h
    src/utils/VulkanSetup.cpp
    src/utils/MathUtils.h
    src/utils/RandomUtils.h
    src/utils/RandomUtils.cpp
    src/gui/ImGuiUtils.h
    src/gui/ImGuiUtils.cpp
    src/gui/LoadingScreen.h
    src/gui/LoadingScreen.cpp
    src/camera/Camera.cpp
    src/terrain/TerrainGeometry.h
    src/terrain/TerrainGeometry.cpp
    src/terrain/TerrainScene.h
    src/terrain/TerrainScene.cpp
    src/water/WaterGeometry.h
    src/water/WaterGeometry.cpp
    src/water/WaterScene.h
    src/water/WaterScene.cpp
    src/algorithms/DiamondSquareGenerator.h
    src/algorithms/DiamondSquareGenerator.cpp
    src/terrain/ChunkCoord.h
    src/terrain/ChunkManager.h
    src/terrain/ChunkManager.cpp
    src/lighting/SceneLighting.h
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    VulkanLaunchpad
)
add_dependencies(${PROJECT_NAME} VulkanLaunchpad)
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

#================================#
# IDE specific setup              #
#================================#
if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
elseif(CMAKE_GENERATOR MATCHES "Xcode")
    set(CMAKE_XCODE_GENERATE_SCHEME ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_GENERATE_SCHEME ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES XCODE_SCHEME_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY XCODE_STARTUP_PROJECT ${PROJECT_NAME})
endif()