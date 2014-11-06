#ifndef CP_INCL
#define CP_INCL

#include "flyparams.h"

typedef struct {
    double zeta0[2]; // initial value for path parameter and derivative
    double Ts; // sampling time
    flyparams_t cntr_flyparams;
	double Ktransv[3]; // feedback gain for transverse controller
    double a; // length of major axis one of ellipse
    double b; // length of major axis one of ellipse
    double delta; // rotation angle of ellipse
    double xme; // x-coordinate of center of ellipse
    double yme; // y-coordinate of center of ellipse
	double R;   // "radius" of lemniscate
	int path_type; // index of path to be followed: 
				   // 1...ellipse
				   // 2...lemniscate
	double *desiredPath; // holds coordinates of the desired path, length NdesPath*2
	int NdesPath; // number of points for the discretization of the desired path
}contrp_t;

typedef struct {
    double zeta[2];    // internal state zeta of the controller (path parameter and derivative)
	double input[1];   // holds value of input omegae, implemented here in a more general form as currently
					   // needed, for possible future extensions
	double intState[2]; // integrator state for integrating controller (error integrator)
	double status[10]; /* status: array containing status information of the controller: 
                        *  0: is reset -> reset has been performed if value 1
                        *  1: in normal operation if value 1
                        */
}cntrState_t;

#include "calculateInput.h"
#include "ekf_fct_model4_switch.h"

contrp_t * contr_new_params();
cntrState_t * contr_new_state();

void contr_TNF_fly_model4 (double *wout, double *zetaout, double *xi_out, int enable, contrp_t *cp, cntrState_t *cntrState, ekfState_t *ekfState, int enableEKF, cInpState_t *cInpState, double *targetPoint, double *xreal_debug, double *zetadebug);

void initController (cntrState_t *cntrState, contrp_t *cp);

void calculate_control_inputs (double *utnext, double *xi, contrp_t *cp, cntrState_t *cntrState, double *xt);

void systemData (double *h,double *LFh,double *LF2h,double *LGLFh, double v0, double *xt);

const double *get_int_state(cntrState_t *cs, int *len);

const double *get_path(contrp_t *cp, int *num);

#endif



