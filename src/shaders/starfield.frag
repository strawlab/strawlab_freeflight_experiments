/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#version 120
uniform sampler2D star_tex;
void main(void)
{
    gl_FragColor = texture2D(star_tex, gl_PointCoord);
}
