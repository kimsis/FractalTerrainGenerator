#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

layout (binding = 0) uniform UniformBuffer {
	mat4 modelMatrix;
	mat4 modelMatrixForNormals;
	mat4 viewProjMatrix;
	vec4 cameraPosition;
	vec4 materialProperties; // ka, kd, ks, alpha
	ivec4 userInput;
} ub_data;

layout (location = 0) out VertexData {
	vec4 position_world;
	vec3 normal_world;
} vert_out;


void main() {
	vert_out.position_world = ub_data.modelMatrix * vec4(in_position, 1);
	vert_out.normal_world = mat3(ub_data.modelMatrixForNormals) * in_normal;
	gl_Position = ub_data.viewProjMatrix * vert_out.position_world;
}
