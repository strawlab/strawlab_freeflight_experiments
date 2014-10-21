#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "simstruc.h"

#include "contr_fct_TNF_model4.h"

#include "calculateInput.h"

#include "helpfunctions.h"

#include "calc_pathsData.h"

contrp_t * contr_new_params() {
    return calloc(1, sizeof(contrp_t));
}
    
cntrState_t * contr_new_state() {
    return calloc(1, sizeof(cntrState_t));
}

void contr_TNF_fly_model4 (double *wout, double *zetaout, double *xi_out, int enable, contrp_t *cp, cntrState_t *cntrState, ekfState_t *ekfState, int enableEKF, cInpState_t *cInpState, double *targetPoint, double *xreal_debug, double *zetadebug) { 
    /* arguments: 
     *  wout: current input to the auxiliary system describing the evolution
     *        of the path parameter, length 1
     *  zetaout: current value of path parameter and derivative, has to be given separately as 
     *            cntrState->zeta is simulated forwards at the end of the function
     *            and therefore, after this function is called, does not correspond
     *            any more to the value for which omegae and w were calculated for, length 2
	 *  xi_out: current value of transverse state xi
     *  enable: 0: do nothing, 1: normal operation
     *  cp: struct containing the parameters for the controller
     *  cntrState: struct containing all the internal variables and status of the controller
     *  ekfState: EFK info including xest and ekf status
     *  enableEKF: enable flag of EKF, needed to determine whether EKF is in 
     *            idle state or provides regular output (estimated state is valid)
     *            statusEKF[1] cannot be used for that as the first normal run
     *            of the EKF occurs possibly after this function is called the first time
	 *  cInpState: struct holding several status information of the function calculating the input value
	 *  targetPoint: array holding the x- and y-coordinates of the point on the desired path to which the 
	 * 			  controller wants to steer the system, length 2
     */
	 
	 double utnext[2];
	 double xt[5]; // overall state (fly and path parameter with derivative)
	 int i;
	 double *zeta = cntrState->zeta; // path parameter and derivative
	 //double *zeta = zetadebug;
	 
	 double Ts = cp->Ts;
	 double zeta_new[2];
	      
     double *xest = ekfState->xest;
	 //double *xest = xreal_debug;
	 
     double *statusEKF = ekfState->status;
	 
	 double LFpyDummy[2];
	 double LF2pyDummy[2], LGLFpyDummy[4];
	 	 	 
	 wout[0] = -1.0; // default-value
     zetaout[0] = 0.0; // default-value
	 zetaout[1] = 0.0; // default-value
	
	
	// Calculate current point on the path according to the value of theta: 
	if (cp->path_type == 1) {
		// ellipse
		pathEllipseData (targetPoint, LFpyDummy, LF2pyDummy, LGLFpyDummy, zeta[0], 0, cp->a, cp->b, cp->delta, cp->xme, cp->yme);
	} else if (cp->path_type == 2) {
		// lemniscate
		pathLemniscateData (targetPoint, LFpyDummy, LF2pyDummy, LGLFpyDummy, zeta[0], 0, cp->R);
	} else {
		printf("unknown path type\n");
	}
		
	
	if ((enable >= 1) && (cntrState->status[0] > 0.5) && (statusEKF[0] > 0.5) && (enableEKF >= 1)) {
		// controller has already been resetted at least once and currently it is enabled
        // furthermore, EKF has been resetted and is currently not in idle-state
        // -> EKF is in normal operation -> estimated state is valid and can be used here, 
        // controller can calculate and provide inputs for the system
        // -> normal operation of controller

        cntrState->status[1] = 1.0; // normal operation
        cInpState->enable = 1; // activate function for calculating the input value

        for (i=0;i<3;i++) xt[i] = xest[i];
        xt[3] = zeta[0];
		xt[4] = zeta[1];
		
		// Decide whether or not to use estimate of v0 or nominal value
		if (EKF_V0EST) {
			cp->cntr_flyparams.v0 = xest[3];
		}

		calculate_control_inputs (utnext, xi_out, cp, cntrState, xt);
		        
        // Tell function which calculates the values of the input that a 
        // new input for the system has been calculated
        cInpState->newInputCalculated = 1;

		cntrState->input[0] = utnext[0];
        wout[0] = utnext[1];
				
        zetaout[0] = zeta[0];
		zetaout[1] = zeta[1];

        // Calculate state zeta of the auxiliary system at the next 
        // sampling instant: 

        // Forward simulation of auxiliary system with input in zero 
        // order hold fashion. 
        zeta_new[0] = zeta[0] + Ts*zeta[1] + 0.5*Ts*Ts*wout[0];
		zeta_new[1] = zeta[1] + Ts*wout[0];
		zeta[0] = zeta_new[0];
		zeta[1] = zeta_new[1];

    } else {
        // controller inactive
        cntrState->status[1] = 0.0;

        cInpState->enable = 0; // disable function for calculating the input value
    }

}


void initController (cntrState_t *cntrState, contrp_t *cp) {
    
    // initialize the arrays and other data of the controller
        
	int i;
    double *status = cntrState->status;
    
    for (i=0;i<10;i++) status[i] = 0.0;
			
    (cntrState->zeta)[0] = cp->zeta0[0];
	(cntrState->zeta)[1] = cp->zeta0[1];
	
	cntrState->input[0] = 0.0;
	
	// Reset integrator state
	cntrState->intState[0] = 0.0;
	cntrState->intState[1] = 0.0;
    
    status[0] = 1.0; // reset has been done
    status[1] = 0.0; // controller currently inactive
            
}


void calculate_control_inputs (double *utnext, double *xi, contrp_t *cp, cntrState_t *cntrState, double *xt) {

	int i;

	int path_type = cp->path_type;
	
	// Read parameters of paths
	double  a = cp->a;
    double  b = cp->b;
    double  delta = cp->delta;
    double  xme = cp->xme;
    double  yme = cp->yme;
	
	double  R = cp->R;
	
	double Ts = cp->Ts;
	
	double v0 = cp->cntr_flyparams.v0;
	
	double py[2],LFpy[2],LF2py[2],LGLFpy[4];
	double h[2],LFh[2],LF2h[2],LGLFh[4];
	
	double zeta1 = xt[3];
	double zeta2 = xt[4];
	
	double D[4],LF2ht[2];
	double detD, invD[4];
	double Ypsilon[2],Ypsilond[2];
	
	double vtransv[2];
	double *Ktransv = cp->Ktransv;
	
	double *intState = cntrState->intState;
	
	
	// Calculate data of path
	if (path_type == 1) {
		// ellipse
		pathEllipseData (py, LFpy, LF2py, LGLFpy, zeta1, zeta2, a, b, delta, xme, yme);
	} else if (path_type == 2) {
		// lemniscate
		pathLemniscateData (py, LFpy, LF2py, LGLFpy, zeta1, zeta2, R);
	} else {
		printf("unknown path type\n");
	}
		
	// Calculate data of system
	systemData (h,LFh,LF2h,LGLFh,v0,xt);
	
	for (i=0;i<4;i++) D[i] = LGLFh[i] - LGLFpy[i];

	LF2ht[0] = LF2h[0] - LF2py[0];
	LF2ht[1] = LF2h[1] - LF2py[1];
	
	Ypsilon[0] = h[0]-py[0];
	Ypsilon[1] = h[1]-py[1];
	
	Ypsilond[0] = LFh[0]-LFpy[0];
	Ypsilond[1] = LFh[1]-LFpy[1];

	xi[0] = Ypsilon[0];
	xi[1] = Ypsilond[0];
	xi[2] = Ypsilon[1];
	xi[3] = Ypsilond[1];

	// Linear controller for regulating system in coordinates of transverse normal form
	vtransv[0] = Ktransv[0]*intState[0] + Ktransv[1]*xi[0] + Ktransv[2]*xi[1];
	vtransv[1] = Ktransv[0]*intState[1] + Ktransv[1]*xi[2] + Ktransv[2]*xi[3];
	
	// Calculate inverse of D
	detD = D[0]*D[3] - D[1]*D[2];
	if (fabs(detD) < 1e-3) printf("D almost singular!\n");
	
	invD[0] = 1/detD*D[3];
	invD[3] = 1/detD*D[0];
	invD[1] = -1/detD*D[1];
	invD[2] = -1/detD*D[2];
	
	utnext[0] = invD[0]*(vtransv[0]-LF2ht[0]) + invD[2]*(vtransv[1]-LF2ht[1]);
	utnext[1] = invD[1]*(vtransv[0]-LF2ht[0]) + invD[3]*(vtransv[1]-LF2ht[1]);

	// Integrate error
	intState[0] = intState[0] + Ts*xi[0];
	intState[1] = intState[1] + Ts*xi[2];

}

void systemData (double *h,double *LFh,double *LF2h,double *LGLFh, double v0, double *xt) {

	double x = xt[0];
	double y = xt[1];
	double ggamma = xt[2];
	
	double cg = cos(ggamma);
	double sg = sin(ggamma);

	h[0] = x;
	h[1] = y;
	LFh[0] = cg * v0;
	LFh[1] = sg * v0;
	LF2h[0] = 0;
	LF2h[1] = 0;
	LGLFh[0] = -sg * v0;
	LGLFh[1] = cg * v0;
	LGLFh[2] = 0;
	LGLFh[3] = 0;

	

}