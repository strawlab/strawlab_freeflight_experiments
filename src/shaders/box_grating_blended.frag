/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#version 120

#define M_PI 3.14159265359
#define M_2PI 6.28318530718

#define pedestal 0.5

varying vec2 box_pos;

uniform float phase_position0;
uniform float wavelength0;
uniform float contrast0;
uniform float orientation0;

void main(void)
{

    // Optimization: These values could be precomputed and sent as tex coords.
    float x = box_pos.x;
    float y = box_pos.y;

    float Q0 = cos(orientation0)*x + sin(orientation0)*y;

    float phase0 = Q0/wavelength0 * M_2PI;
    float value0 = contrast0*0.5*sin( phase0 + phase_position0 );

    float value = value0 + pedestal;
    gl_FragColor = vec4(value, value, value, 1.0);
    //gl_FragColor = vec4(x, y, 0.0, 1.0);
}
