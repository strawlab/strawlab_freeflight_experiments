#include <math.h>

void pathEllipseData (double *py, double *LFpy, double *LF2py, double *LGLFpy, double zeta1, double zeta2, double a, double b, double delta, double xme, double yme) {
    
	double cz1 = cos(zeta1);
    double cd = cos(delta);
    double sz1 = sin(zeta1);
    double sd = sin(delta);
    
    // ellipse
    py[0] = a * cz1 * cd - b * sz1 * sd + xme;
	py[1] = a * cz1 * sd + b * sz1 * cd + yme;
	LFpy[0] = (-a * sz1 * cd - b * cz1 * sd) * zeta2;
	LFpy[1] = (-a * sz1 * sd + b * cz1 * cd) * zeta2;
	LF2py[0] = -(a * cz1 * cd - b * sz1 * sd) * zeta2 * zeta2;
	LF2py[1] = -(a * cz1 * sd + b * sz1 * cd) * zeta2 * zeta2;
	LGLFpy[0] = 0;
	LGLFpy[1] = 0;
	LGLFpy[2] = -a * sz1 * cd - b * cz1 * sd;
	LGLFpy[3] = -a * sz1 * sd + b * cz1 * cd;

        
}

void pathLemniscateData (double *py, double *LFpy, double *LF2py, double *LGLFpy, double zeta1, double zeta2, double R) {
    
	double cz1 = cos(zeta1);
    double sz1 = sin(zeta1);
        
    // lemniscate
    py[0] = R * cz1;
	py[1] = R * cz1 * sz1;

	LFpy[0] = -R * sz1 * zeta2;
	LFpy[1] = (R * cz1 * cz1 - R * sz1 * sz1) * zeta2;

	LF2py[0] = -R * cz1 * zeta2 * zeta2;
	LF2py[1] = -0.4e1 * R * cz1 * sz1 * zeta2 * zeta2;

	LGLFpy[0] = 0;
	LGLFpy[1] = 0;
	LGLFpy[2] = -R * sz1;
	LGLFpy[3] = R * (0.2e1 * cz1 * cz1 - 0.1e1);
        
}
