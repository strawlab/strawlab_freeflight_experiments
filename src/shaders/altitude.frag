/* -*- Mode: C; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#version 120
uniform float edge_height;
varying float z;

void main(void)
{
	if (z < edge_height) {
		gl_FragColor = vec4(0,0,0,1);
	} else {
		gl_FragColor = vec4(1,1,1,1);
	}
}
