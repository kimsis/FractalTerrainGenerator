# Applies local patches to the VulkanLaunchpad submodule, using only `git` (no shell) so this
# works identically on Linux/macOS/Windows. These fix real bugs in that upstream framework (e.g. a
# Vulkan validation error), but since it's a third-party repo (cg-tuwien/VulkanLaunchpad) we don't
# have write access to, the fix can't be committed there directly - only carried here as a patch
# and re-applied to every fresh checkout.
#
# Safe to re-run: a patch that's already applied is detected and skipped. Included automatically
# from the root CMakeLists.txt's configure step, so normally you don't need to run this by hand.
#
# Expects GIT_EXECUTABLE and PROJECT_SOURCE_DIR to already be set by the including CMakeLists.txt.

set(SUBMODULE_DIR "${PROJECT_SOURCE_DIR}/external/VulkanLaunchpad")
set(PATCH_DIR "${PROJECT_SOURCE_DIR}/patches")

file(GLOB PATCH_FILES "${PATCH_DIR}/*.patch")
list(SORT PATCH_FILES)

foreach(PATCH_FILE ${PATCH_FILES})
    get_filename_component(PATCH_NAME "${PATCH_FILE}" NAME)

    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --check "${PATCH_FILE}"
        WORKING_DIRECTORY "${SUBMODULE_DIR}"
        RESULT_VARIABLE PATCH_CHECK_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )

    if(PATCH_CHECK_RESULT EQUAL 0)
        message(STATUS "Applying ${PATCH_NAME}...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply "${PATCH_FILE}"
            WORKING_DIRECTORY "${SUBMODULE_DIR}"
            RESULT_VARIABLE PATCH_APPLY_RESULT
        )
        if(NOT PATCH_APPLY_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to apply ${PATCH_NAME} even though the check succeeded.")
        endif()
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply --reverse --check "${PATCH_FILE}"
            WORKING_DIRECTORY "${SUBMODULE_DIR}"
            RESULT_VARIABLE PATCH_REVERSE_CHECK_RESULT
            OUTPUT_QUIET ERROR_QUIET
        )
        if(PATCH_REVERSE_CHECK_RESULT EQUAL 0)
            message(STATUS "${PATCH_NAME} already applied, skipping.")
        else()
            message(FATAL_ERROR
                "${PATCH_NAME} does not apply cleanly and isn't already applied. "
                "VulkanLaunchpad may have changed upstream in a conflicting way - review manually."
            )
        endif()
    endif()
endforeach()
