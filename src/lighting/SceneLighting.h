#pragma once

#include <glm/glm.hpp>

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
 *	This struct contains the data of a directional light.
 *	It matches the definition and sizes of the corresponding GPU-side struct exactly, which is used in shaders.
 */
struct DirectionalLight {
    /*! Light color of this light source */
    glm::vec4 color;

    /*! Light direction of this directional light source */
    glm::vec4 direction;
};
