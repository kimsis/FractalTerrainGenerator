#version 450

layout (location = 0) in vec3 in_position;

layout (binding = 0) uniform UniformBufferWaterVert {
	mat4 modelMatrix;
	mat4 viewProjMatrix;
} ub_data;

void main() {
	gl_Position = ub_data.viewProjMatrix * ub_data.modelMatrix * vec4(in_position, 1.0);
}
