/* -*- Mode: C; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#version 120

varying float z;

void main(void)
{
  gl_Position=ftransform(); // standard OpenGL vertex transform
  z = gl_Vertex.z;
}
