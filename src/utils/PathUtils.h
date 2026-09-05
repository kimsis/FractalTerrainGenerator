#pragma once
/*
 * Copyright 2023 Vienna University of Technology.
 * Institute of Computer Graphics and Algorithms.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 */

#include <filesystem>
#include <fstream>
#include <string>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#define PATH_MAX 1337
#else
#include <limits.h>
#include <unistd.h>

#include <vector>
#endif
#include <VulkanLaunchpad.h>

extern std::string gcgLoadShaderFilePath(const std::string& filePath);

inline std::filesystem::path gcgGetExecutableDir() {
    char buffer[PATH_MAX];
#if defined(__APPLE__)
    uint32_t size = PATH_MAX;
    if (_NSGetExecutablePath(buffer, &size) == -1) {
        VKL_EXIT_WITH_ERROR("Error retrieving executable path on MacOS!");
    }
#elif defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
    DWORD ret = GetModuleFileNameA(NULL, buffer, PATH_MAX);
    if (ret == 0 || ret == PATH_MAX) {
        VKL_EXIT_WITH_ERROR("Error retrieving executable path on Windows!");
    }
#else
    ssize_t ret = readlink("/proc/self/exe", buffer, PATH_MAX);
    if (ret == -1) {
        VKL_EXIT_WITH_ERROR("Error retrieving executable path on Unix-like OS!");
    }
#endif
    std::filesystem::path exec_path(buffer);
    return exec_path.parent_path();
}

inline bool gcgFileExists(const std::string& filePath) {
    std::ifstream file(filePath);
    return file.good();
}

inline std::string gcgFindFileInParentDir(std::filesystem::path currentDir, const std::string& targetFile, std::vector<std::string>& candidates) {
    // Construct the full file path
    std::filesystem::path targetPath = currentDir / targetFile;

    // Check if file exists
    if (gcgFileExists(targetPath.string())) {
        candidates.push_back(targetPath.string());
    }

    // Stop either at root directory, or if we encounter a CMakeLists.txt
    auto cMakeListsExists = gcgFileExists((currentDir / "CMakeLists.txt").string());

    if (!cMakeListsExists && currentDir.parent_path() != currentDir) {
        return gcgFindFileInParentDir(currentDir.parent_path(), targetFile, candidates);
    }

    if (candidates.size() == 0) {
        return "";
    } else if (candidates.size() == 1) {
        return candidates[0];
    } else {
        VKL_WARNING("Ambiguous asset file path: '" << targetFile << "'");
        VKL_WARNING("Found this path at multiple locations. Don't know which one shall be used. Candidates are:");
        for (const auto& candidate : candidates) {
            VKL_LOG("  - " + candidate);
        }
        VKL_EXIT_WITH_ERROR("Remove all duplicates to fix this!");
    }
}

inline std::string gcgFindFileInParentDir(const std::string& targetFile) {
    std::vector<std::string> candidates;
    return gcgFindFileInParentDir(gcgGetExecutableDir(), targetFile, candidates);
}

inline std::string gcgFindTextureFile(const std::string& targetFile) {
    std::string texturePathResult = gcgFindFileInParentDir(targetFile);
    if (texturePathResult.empty()) {
        VKL_EXIT_WITH_ERROR("Could not find texture file: " << targetFile);
    }
    return texturePathResult;
}

inline std::vector<std::string> gcgFindTextureFiles(const std::vector<std::string>& targetFiles) {
    std::vector<std::string> texturePaths;

    for (const auto& targetFile : targetFiles) {
        std::string texturePath = gcgFindTextureFile(targetFile);
        texturePaths.push_back(texturePath);
    }

    return texturePaths;
}

inline std::string gcgFindShaderFile(const std::string& targetFile) {
    std::string shaderPathResult = gcgFindFileInParentDir(targetFile);
    if (shaderPathResult.empty()) {
        VKL_EXIT_WITH_ERROR("Could not find shader file: " << targetFile);
    }
    return shaderPathResult;
}

template <size_t N, size_t M>
inline std::vector<std::vector<std::string>> gcgFindAllShaderFiles(const char* shaders[N][M]) {
    std::vector<std::vector<std::string>> allShaderPaths;

    // Loop through each illumination mode
    for (int i = 0; i < N; i++) {
        std::vector<std::string> shaderPathsForMode;

        // Loop through each file for the current mode
        for (int j = 0; j < M; j++) {
            // Call the function to find the shader file and add the result to the vector
            shaderPathsForMode.push_back(gcgFindShaderFile(shaders[i][j]));
        }

        // Add the paths for the current mode to the main vector
        allShaderPaths.push_back(shaderPathsForMode);
    }

    return allShaderPaths;
}
