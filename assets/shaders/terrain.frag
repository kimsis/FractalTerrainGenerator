#version 450
 

layout (location = 0) in VertexData {
	vec4 position_world;
	vec3 normal_world;
} frag_in;

layout (binding = 1) uniform UniformBufferFrag {
	vec4 cameraPosition;
	vec4 materialProperties; // ka, kd, ks, alpha
	uvec2 debugToggles; // x = drawNormals (N), y = highlightChunkBorders (F3)
	bool isUnderwater;
	float chunkWidth;
	float roughness;
	float dirtToGrassHeight;
	float grassToRockHeight;
	float heightColorTransitionBand;
} ub_data;

const float BORDER_HIGHLIGHT_HALF_WIDTH = 0.5; // 1 world unit wide, independent of vertex spacing

// Slopes steeper than this show bare rock regardless of height, blended over a +/-5 degree band to
// avoid a hard, aliased edge where the terrain crosses the threshold.
const float SLOPE_ROCK_ANGLE_DEG = 45.0;
const float SLOPE_ROCK_TRANSITION_DEG = 5.0;

layout (binding = 2) uniform DirectionalLight {
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

	// Height-based color: brown dirt in valleys/underwater, green grass at mid elevation, grey rock
	// on peaks, smoothly blended across dirtToGrassHeight/grassToRockHeight.
	vec3 dirtColor = vec3(0.45, 0.35, 0.20);
	vec3 grassColor = vec3(0.4, 0.6, 0.3);
	vec3 rockColor = vec3(0.22, 0.22, 0.22);
	float height = frag_in.position_world.z;
	vec3 baseColor = mix(dirtColor, grassColor, smoothstep(
		ub_data.dirtToGrassHeight - ub_data.heightColorTransitionBand,
		ub_data.dirtToGrassHeight + ub_data.heightColorTransitionBand,
		height
	));
	baseColor = mix(baseColor, rockColor, smoothstep(
		ub_data.grassToRockHeight - ub_data.heightColorTransitionBand,
		ub_data.grassToRockHeight + ub_data.heightColorTransitionBand,
		height
	));

	// Steep slopes are bare rock regardless of height. dot(n, up) is cos(slopeAngle) and decreases as
	// slope steepens, so the smoothstep edges are the steep/shallow band ends' cosines, in that order.
	float upDot = dot(n, vec3(0.0, 0.0, 1.0));
	float slopeRockFactor = 1.0 - smoothstep(
		cos(radians(SLOPE_ROCK_ANGLE_DEG + SLOPE_ROCK_TRANSITION_DEG)),
		cos(radians(SLOPE_ROCK_ANGLE_DEG - SLOPE_ROCK_TRANSITION_DEG)),
		upDot
	);
	baseColor = mix(baseColor, rockColor, slopeRockFactor);

	// Start with ambient illumination contribution:
	vec3 color = baseColor * ub_data.materialProperties[0];

	float diffuseF  = ub_data.materialProperties[1];
	// Roughness dims and broadens the specular highlight: 0 is the original sharp/shiny look, 1 fully matte.
	float specularF = ub_data.materialProperties[2] * (1.0 - ub_data.roughness);
	float specularA = mix(ub_data.materialProperties[3], 1.0, ub_data.roughness);

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

	if (ub_data.debugToggles.x != 0u) {
		vec3 scaledNormal = 0.5 * n + 0.5;
        out_color = vec4(pow(scaledNormal.x, 2.2), pow(scaledNormal.y, 2.2), pow(scaledNormal.z, 2.2), 1.0);
    }

	if (ub_data.isUnderwater) {
		vec3 underwaterColor = vec3(0.0, 0.15, 0.3);
		out_color = vec4(mix(out_color.rgb, underwaterColor, 0.6), out_color.a);
	}

	if (ub_data.debugToggles.y != 0u) {
		// Chunk boundaries sit at world positions where (worldXY + chunkWidth/2) is an exact
		// multiple of chunkWidth (see DiamondSquareGenerator's world-position formula); measure each
		// fragment's distance to the nearest such multiple, on both axes.
		vec2 shifted = frag_in.position_world.xy + vec2(ub_data.chunkWidth * 0.5);
		vec2 distanceIntoCell = mod(shifted, ub_data.chunkWidth);
		vec2 distanceToBoundary = min(distanceIntoCell, ub_data.chunkWidth - distanceIntoCell);
		if (min(distanceToBoundary.x, distanceToBoundary.y) < BORDER_HIGHLIGHT_HALF_WIDTH) {
			out_color = vec4(1.0, 0.0, 0.0, 1.0);
		}
	}
}
