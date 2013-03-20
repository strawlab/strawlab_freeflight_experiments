/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#version 120
uniform float tex_v_offset;
uniform sampler2D my_texture;

varying vec4 my_texcoord;

void main(void)
{
    vec2 texcoord = vec2( my_texcoord.x, my_texcoord.y+tex_v_offset );
    gl_FragColor = texture2D(my_texture, texcoord);
}
