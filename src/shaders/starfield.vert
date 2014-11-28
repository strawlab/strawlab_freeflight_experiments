/* -*- Mode: C -*- */
#version 120

uniform float pixelsize;
//varying float dist; // For fog computation in fragment shader.

void main(void)
{
  vec4 eyePos = gl_ModelViewMatrix * gl_Vertex;
  //dist = distance(eyePos, vec4(0.0, 0.0, 0.0, 0.0));
  gl_Position = gl_ProjectionMatrix * eyePos;
  // gl_PointSize = pixelsize/gl_Position.w; // More natural - size inversely proportional to distance.
  gl_PointSize = pixelsize; // Size is constant in angular terms (independent of distance).
}
