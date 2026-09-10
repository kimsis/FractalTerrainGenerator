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
# applied patches are detected and skipped by the script itself.
find_program(BASH_PROGRAM bash)
if(BASH_PROGRAM)
    execute_process(COMMAND ${BASH_PROGRAM} "${PROJECT_SOURCE_DIR}/patches/apply-patches.sh"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE PATCH_APPLY_RESULT
                    OUTPUT_VARIABLE PATCH_APPLY_OUTPUT
                    ERROR_VARIABLE PATCH_APPLY_ERROR)
    if(NOT PATCH_APPLY_RESULT EQUAL "0")
        message(WARNING "Failed to apply local VulkanLaunchpad patches (see patches/apply-patches.sh):\n${PATCH_APPLY_OUTPUT}${PATCH_APPLY_ERROR}")
    else()
        message(STATUS "${PATCH_APPLY_OUTPUT}")
    endif()
else()
    message(WARNING "bash not found - could not auto-apply patches/*.patch to VulkanLaunchpad. Run patches/apply-patches.sh manually (e.g. via Git Bash on Windows).")
endif()


#================================#
# VulkanLaunchpad (dependency)   #
#================================#
add_subdirectory(external/VulkanLaunchpad)

#================================#
# GCG Framework (prebuilt lib)   #
#================================#
if(WIN32)
    set(GCG_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/windows")
    set(GCG_LIB_DEBUG "${GCG_LIB_DIR}/GCG_VK_Lib_Debug.lib")
    set(GCG_LIB_RELEASE "${GCG_LIB_DIR}/GCG_VK_Lib_Release.lib")
elseif(APPLE)
    set(GCG_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/macos")
    set(GCG_LIB_DEBUG "${GCG_LIB_DIR}/libGCG_VK_Lib_Debug.a")
    set(GCG_LIB_RELEASE "${GCG_LIB_DIR}/libGCG_VK_Lib_Release.a")
else()
    set(GCG_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/linux")
    set(GCG_LIB_DEBUG "${GCG_LIB_DIR}/libGCG_VK_Lib_Debug.a")
    set(GCG_LIB_RELEASE "${GCG_LIB_DIR}/libGCG_VK_Lib_Release.a")
endif()

#================================#
# Dear ImGui (GUI dependency)    #
#================================#
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/imgui")

#================================#
# VulkanLaunchpadStarter         #
#================================#
add_executable(${PROJECT_NAME}
    src/Main.cpp
    src/Camera.h
    src/utils/VulkanHelpers.h
    src/utils/VulkanHelpers.cpp
    src/utils/Utils.h
    src/utils/MathUtils.h
    src/terrain/Geometry.h
    src/terrain/Geometry.cpp
    src/terrain/DiamondSquareGenerator.h
    src/terrain/DiamondSquareGenerator.cpp
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
    $<$<CONFIG:Debug>:${GCG_LIB_DEBUG}>
    $<$<NOT:$<CONFIG:Debug>>:${GCG_LIB_RELEASE}>
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