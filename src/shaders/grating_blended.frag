/* -*- Mode: C; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#version 120

#define M_PI 3.14159265359
#define M_2PI 6.28318530718

#define pedestal 0.5

varying vec3 cyl_pos;

uniform float phase_position0, phase_position1;
uniform float wavelength0, wavelength1;
uniform float contrast0, contrast1;
uniform float orientation0, orientation1;

void main(void)
{

    // Optimization: These values could be precomputed and sent as tex coords.
    float azimuth = atan(cyl_pos.y, cyl_pos.x);
    float r = sqrt( cyl_pos.x*cyl_pos.x + cyl_pos.y*cyl_pos.y + cyl_pos.z*cyl_pos.z );
    float inclination = acos(cyl_pos.z/r); // 0 at north pole, increases downward to M_PI
    //  float elevation = (M_PI*0.5)-inclination; // 0 at equator, north pole is +M_PI

    float Q0 = cos(orientation0)*azimuth + sin(orientation0)*inclination;
    float Q1 = cos(orientation1)*azimuth + sin(orientation1)*inclination;

    float phase0 = Q0/wavelength0 * M_2PI;
    float value0 = contrast0*0.5*sin( phase0 + phase_position0 );

    float phase1 = Q1/wavelength1 * M_2PI;
    float value1 = contrast1*0.5*sin( phase1 + phase_position1 );

    float value = value0 + value1 + pedestal;
    gl_FragColor = vec4(value, value, value, 1.0);
}
