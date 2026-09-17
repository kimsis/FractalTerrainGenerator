#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <string>

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
