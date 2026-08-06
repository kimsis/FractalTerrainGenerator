#version 450
/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
 
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

layout (binding = 3) uniform sampler2D diffuse_texture;

layout (location = 0) in VertexData {
	vec3 positionWorld;
	vec3 normalWorld;
	vec2 textureCoordinates;
} frag_in;

layout (location = 0) out vec4 out_color;

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

// Gets the reflected color value from a a certain position, from 
// a certain direction INSIDE of a cornell box of size 3 which is 
// positioned at the origin.
// positionWS:  Position inside the cornell box for which to get the
//              reflected color value for from -directionWS.
// directionWS: Outgoing direction vector (from positionWS towards the
//              outside) for which to get the reflected color value for.
vec3 getCornellBoxReflectionColor(vec3 positionWS, vec3 directionWS) {
	vec3 P0 = positionWS;
	vec3 V  = normalize(directionWS);

	const float boxSize = 1.5;
	vec4[5] planes = {
		vec4(-1.0,  0.0,  0.0, -boxSize), // left
		vec4( 1.0,  0.0,  0.0, -boxSize), // right
		vec4( 0.0,  1.0,  0.0, -boxSize), // top
		vec4( 0.0, -1.0,  0.0, -boxSize), // bottom
		vec4( 0.0,  0.0, -1.0, -boxSize)  // back
	};
	vec3[5] colors = {
		vec3(0.49, 0.06, 0.22),    // left
		vec3(0.0, 0.13, 0.31),    // right
		vec3(0.96, 0.93, 0.85), // top
		vec3(0.64, 0.64, 0.64), // bottom
		vec3(0.76, 0.74, 0.68)  // back
	};

	for (int i = 0; i < 5; ++i) {
		vec3  N = planes[i].xyz;
		float d = planes[i].w;
		float denom = dot(V, N);
		if (denom <= 0) continue;
		float t = -(dot(P0, N) + d)/denom;
		vec3  P = P0 + t*V;
		float q = boxSize + 0.01;
		if (P.x > -q && P.x < q && P.y > -q && P.y < q && P.z > -q && P.z < q) {
			return colors[i];
		}
	}
	return vec3(0.0, 0.0, 0.0);
}

// Computes the reflection direction for an incident vector I about normal N, 
// and clamps the reflection to a maximum of 180, i.e. the reflection vector
// will always lie within the hemisphere around normal N.
// Aside from clamping, this function produces the same result as GLSL's reflect function.
vec3 clampedReflect(vec3 I, vec3 N)
{
	return I - 2.0 * min(dot(N, I), 0.0) * N;
}

vec3 fresnelSchlick (float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow (1.0 - cosTheta, 5.0);
}

void main() {	
	vec3 n = normalize(frag_in.normalWorld);
	vec3 v = normalize(frag_in.positionWorld - ub_data.cameraPosition.xyz);
	vec3 R = normalize(clampedReflect(v, n));
	vec3 reflectionColor = getCornellBoxReflectionColor(frag_in.positionWorld, R);
	vec3 F0 = vec3(0.1); // <-- some kind of plastic
	vec3 reflectivity = fresnelSchlick(dot(n, -v), F0);
	vec3 diffuseColor = texture(diffuse_texture, frag_in.textureCoordinates).rgb;
	
	// Start with ambient illumination contribution:
	vec3 color = diffuseColor * ub_data.illumination[0];
	
	// Add directional light's contribution:
	color += phong(
		n, 
		-dl_data.direction.xyz, 
		-v, 
		dl_data.color.rgb * diffuseColor, ub_data.illumination[1], 
		dl_data.color.rgb,                ub_data.illumination[2], 
		ub_data.illumination[3], 
		false, vec3(1.0)
	);
			
	// Add point light's contribution
	color += phong(
		n, 
		pl_data.position.xyz - frag_in.positionWorld, 
		-v, 
		pl_data.color.rgb * diffuseColor, ub_data.illumination[1], 
		pl_data.color.rgb,                ub_data.illumination[2], 
		ub_data.illumination[3], 
		true, pl_data.attenuation.xyz
	);

	// Write color for the current fragment:
	out_color = vec4(color, 1.0);
	if (ub_data.userInput[1] == 1) { // toggle fresnel
		color =  mix(color, reflectionColor, reflectivity);
        out_color = vec4(color, 1.0);
    }

	if (ub_data.userInput[0] == 1) { // toggle normals
        vec3 scaledNormal = 0.5 * n + 0.5;
        out_color = vec4(pow(scaledNormal.x, 2.2), pow(scaledNormal.y, 2.2), pow(scaledNormal.z, 2.2), 1.0);
    }
	if (ub_data.userInput[2] == 1) { // toggle texcoords
        out_color = vec4(frag_in.textureCoordinates, 0, 1);
    }
}

