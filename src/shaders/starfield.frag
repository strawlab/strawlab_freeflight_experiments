/* -*- Mode: C -*- */
#version 120

uniform vec3 color;

void main (void)
{
   vec4 result;

   vec2 tex_coord = gl_TexCoord[0].xy;
   tex_coord.y = 1.0-tex_coord.y;
   float d = 2.0*distance(tex_coord.xy, vec2(0.5, 0.5));
   result.a = step(d, 1.0);

   result.rgb = color;

   gl_FragColor = result;
}
