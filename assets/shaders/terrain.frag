#version 450
 

layout (location = 0) in VertexData {
	vec4 position_world;
	vec3 normal_world;
} frag_in;

layout (binding = 0) uniform UniformBuffer {
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

void main() {
	
	vec3 n = normalize(frag_in.normal_world);
	vec3 v = normalize(ub_data.cameraPosition.xyz - frag_in.position_world.xyz);
	vec3 baseColor = vec3(0.4, 0.6, 0.3);

	// Start with ambient illumination contribution:
	vec3 color = baseColor * ub_data.materialProperties[0];

	float diffuseF  = ub_data.materialProperties[1];
	float specularF = ub_data.materialProperties[2];
	float specularA = ub_data.materialProperties[3];
	
	// Add directional light's contribution:
	color += phong(
		n, 
		-dl_data.direction.xyz, 
		v, 
		dl_data.color.rgb * baseColor, diffuseF, 
		dl_data.color.rgb,	specularF, specularA,
		false, vec3(1.0)
	);

	out_color = vec4(color, 1.0);

	if (ub_data.userInput[0] == 1) { // toggle normals
		vec3 scaledNormal = 0.5 * n + 0.5;
        out_color = vec4(pow(scaledNormal.x, 2.2), pow(scaledNormal.y, 2.2), pow(scaledNormal.z, 2.2), 1.0);
    }
}
