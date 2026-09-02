#version 450
/*
 * Copyright 2023 TU Wien, Institute of Visual Computing & Human-Centered Technology.
 * This file is part of the GCG Lab Framework and must not be redistributed.
 *
 * Original version created by Lukas Gersthofer and Bernhard Steiner.
 * Vulkan edition created by Johannes Unterguggenberger (junt@cg.tuwien.ac.at).
 */
 
layout (location = 0) in VertexData {
	vec3 color;
} frag_in;

layout (location = 0) out vec4 out_color;

void main() {	
	out_color = vec4(frag_in.color, 1.0);
}
