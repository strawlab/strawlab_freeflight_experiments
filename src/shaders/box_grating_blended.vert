/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#version 120

varying vec2 box_pos;

void main(void)
{
  gl_Position=ftransform(); // standard OpenGL vertex transform
  box_pos = gl_Vertex.xy;
}
