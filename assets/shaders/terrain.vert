#version 450

layout (location = 0) in vec3 in_position_from;
layout (location = 1) in vec3 in_position_to;
layout (location = 2) in vec3 in_normal_from;
layout (location = 3) in vec3 in_normal_to;

layout (binding = 0) uniform UniformBufferVert {
	mat4 modelMatrix;
	mat4 modelMatrixForNormals;
	mat4 viewProjMatrix;
	float blendFactor;
} ub_data;

layout (location = 0) out VertexData {
	vec4 position_world;
	vec3 normal_world;
} vert_out;


void main() {
	vec4 position_world_from = ub_data.modelMatrix * vec4(in_position_from, 1);
	vec4 position_world_to = ub_data.modelMatrix * vec4(in_position_to, 1);
	vert_out.position_world = mix(position_world_from, position_world_to, ub_data.blendFactor);

	vec3 normal_world_from = mat3(ub_data.modelMatrixForNormals) * in_normal_from;
	vec3 normal_world_to = mat3(ub_data.modelMatrixForNormals) * in_normal_to;
	vert_out.normal_world = normalize(mix(normal_world_from, normal_world_to, ub_data.blendFactor));

	gl_Position = ub_data.viewProjMatrix * vert_out.position_world;
}
