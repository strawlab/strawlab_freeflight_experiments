/* -*- Mode: C -*- */
#version 120

uniform float pixelsize;

void main(void)
{
  vec4 eyePos = gl_ModelViewMatrix * gl_Vertex;
  gl_Position = gl_ProjectionMatrix * eyePos;

  vec3 eye3 = vec3( eyePos.x/ eyePos.w, eyePos.y / eyePos.w, eyePos.z / eyePos.w );

  float dist = length(eye3);
  gl_PointSize = pixelsize/dist;
}
