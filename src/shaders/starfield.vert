/* -*- Mode: C -*- */
#version 120

uniform float pixelsize;
varying float dist;

void main(void)
{
  vec4 eyePos = gl_ModelViewMatrix * gl_Vertex;
  gl_Position = gl_ProjectionMatrix * eyePos;
  gl_PointSize = pixelsize;
}
