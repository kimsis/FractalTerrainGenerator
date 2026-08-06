#version 450
/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texture_coordinates;

layout (binding = 0) uniform UniformBuffer {
	vec4 color;
	mat4 modelMatrix;
	mat4 modelMatrixForNormals;
	mat4 viewProjMatrix;
	vec4 cameraPosition;
	vec4 illumination; // ka, kd, ks, alpha
	ivec4 userInput;
} ub_data;

layout (binding = 1) uniform DirectionalLight {
	vec4 color;
	vec4 direction;
} dl_data;

layout (binding = 2) uniform PointLight {
	vec4 color;
	vec4 position;
	vec4 attenuation;
} pl_data;

layout (location = 0) out VertexData {
	vec3 positionWorld;
	vec3 normalWorld;
	vec2 textureCoordinates;
} vert_out;

void main() {
	vec4 position_world = ub_data.modelMatrix * vec4(in_position, 1);
	vert_out.positionWorld = position_world.xyz;
	gl_Position = ub_data.viewProjMatrix * position_world;
	
	vert_out.normalWorld = mat3(ub_data.modelMatrixForNormals) * in_normal;
	vert_out.textureCoordinates = in_texture_coordinates;
}
