/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#version 120
varying vec4 my_texcoord;

void main(void)
{
  gl_Position=ftransform(); // standard OpenGL vertex transform
  my_texcoord = gl_MultiTexCoord0;
}
