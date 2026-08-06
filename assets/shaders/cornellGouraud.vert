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
layout (location = 2) in vec3 in_color;

layout (binding = 0) uniform UniformBuffer {
	vec4 color;
	mat4 modelMatrix;
	mat4 modelMatrixForNormals;
	mat4 viewProjMatrix;
	vec4 cameraPosition;
	vec4 materialProperties; // ka, kd, ks, alpha
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
	vec3 color;
} vert_out;

vec3 phong(vec3 n, vec3 l, vec3 v, vec3 diffuseC, float diffuseF, vec3 specularC, float specularF, float alpha, bool attenuate, vec3 attenuation) {
	float d = length(l);
	l = normalize(l);
	float att = 1.0;	
	if(attenuate) {
		att = 1.0f / (attenuation.x + d * attenuation.y + d * d * attenuation.z);
	}
	vec3 r = reflect(-l, n);
	return (diffuseF * diffuseC * max(0, dot(n, l)) + specularF * specularC * pow(max(0, dot(r, v)), alpha)) * att; 
}

void main() {

	vec3 normal_world = mat3(ub_data.modelMatrixForNormals) * in_normal;
	vec4 position_world = ub_data.modelMatrix * vec4(in_position, 1);
	gl_Position = ub_data.viewProjMatrix * position_world;

	vec3 n = normalize(normal_world);
	vec3 v = normalize(ub_data.cameraPosition.xyz - position_world.xyz);
	
	// Use a different illumination depending on whether we see the inside or outside of the Cornell Box:
	if (dot(n, v) > 0.0) {
		// Assume that the Cornell Box is emissive inside:
		vert_out.color = in_color;

		if (ub_data.userInput[0] == 1) { // toggle normals
        	vec3 scaledNormal = 0.5 * n + 0.5;
        	vert_out.color = vec3(pow(scaledNormal.x, 2.2), pow(scaledNormal.y, 2.2), pow(scaledNormal.z, 2.2));
    	}
		
		return;
	}
	// else, i.e. when viewed from the outside, use the ordinarily shaded color.
	// But attention: We have to invert the normal, because we'd like to illuminate the back faces:
	n = -n;

	// Start with ambient illumination contribution:
	vec3 color = in_color * ub_data.materialProperties[0];

	float diffuseF  = ub_data.materialProperties[1];
	float specularF = ub_data.materialProperties[2];
	float specularA = ub_data.materialProperties[3];
	
	// Add directional light's contribution:
	color += phong(
		n, 
		-dl_data.direction.xyz, 
		v, 
		dl_data.color.rgb * in_color, diffuseF, 
		dl_data.color.rgb,                specularF, specularA,
		false, vec3(1.0)
	);
			
	// Add point light's contribution:
	color += phong(
		n, 
		pl_data.position.xyz - position_world.xyz, 
		v, 
		pl_data.color.rgb * in_color, diffuseF, 
		pl_data.color.rgb,                specularF, specularA,
		true, pl_data.attenuation.xyz
	);

	vert_out.color = color;

	if (ub_data.userInput[0] == 1) { // toggle normals
		vec3 scaledNormal = 0.5 * n + 0.5;
        vert_out.color = vec3(pow(scaledNormal.x, 2.2), pow(scaledNormal.y, 2.2), pow(scaledNormal.z, 2.2));
    }
}
