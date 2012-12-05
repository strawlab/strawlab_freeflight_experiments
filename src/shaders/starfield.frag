/* -*- Mode: C -*- */
#version 120

uniform vec3 fog_color;
uniform vec3 color;
varying float dist;

void main (void)
{
   vec4 result;

   vec2 tex_coord = gl_TexCoord[0].xy;
   tex_coord.y = 1.0-tex_coord.y;
   float d = 2.0*distance(tex_coord.xy, vec2(0.5, 0.5));
   result.a = step(d, 1.0);

   float fog_factor = clamp(dist/10.0,0,1);
   result.rgb = mix(color, fog_color, fog_factor);

   gl_FragColor = result;
}
