// on linux M_PI exists if using <C99, otherwise one must define
// #define _GNU_SOURCE and then use -std=c99 with GCC
#define _USE_MATH_DEFINES

#include <stdlib.h>  // for calloc
#include <math.h> // for pi and max-function

#include "simstruc.h"

#include "ekf_fct_model4_switch.h"
#include "contr_fct_TNF_model4.h"
#include "dec_fct.h"
#include "calculateInput.h"
#include "calc_pathsData.h"

#define MAX(a,b) (((a)>(b))?(a):(b))

// Nominal parameters of fly
#define V0 0.3


void init_par_cInpF_decF_ekf_cntr (contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, cInputp_t *cInputp) {

	/* the operator -> references to an element of a pointer to a 
     * struct, instead of, e.g., cp->Nx also (*cp).Nx could be used, but 
     * it is not common
	 */

	int i;
	double deltaTheta;
	double LFpyDummy[2];
	double LF2pyDummy[2], LGLFpyDummy[4];
	
    
    // parameters directly related to decision function
    decfp->Ts = 0.01; // sample time
    
    // parameters directly related to the function for calculating the values of the input
    cInputp->Ts = 0.0125; // sample time, 80Hz

	// system parameters for controller
	cp->cntr_flyparams.v0 = V0;
	
	// system parameters for EKF
	ekfp->ekf_flyparams.v0 = V0;
	
	// parameters directly related to controller
	cp->Ts = 0.025;
	cp->zeta0[0] = 0.5;
	cp->zeta0[1] = 0;
	/*cp->Ktransv[0] = 0.0;
	cp->Ktransv[1] = -1.0;
	cp->Ktransv[2] = -2.0;*/
	cp->Ktransv[0] = -0.1;
	cp->Ktransv[1] = -1.2;
	cp->Ktransv[2] = -2.1;
	
		
	// geometric parameters for path ellipse
	cp->a = 0.2;
	cp->b = 0.15;
	cp->delta = 30*M_PI/180;
	cp->xme = 0;
	cp->yme = 0;
	
	// geometric parameters for path lemniscate
	cp->R = 0.2;
	
	// index of path to be followed
	cp->path_type = 1;
	
    // TEST
    /*cp->a = 0.55;
	cp->b = 0.25;
	cp->delta = 0.15;
	cp->xme = 1.05;
	cp->yme = 0;*/
	
    
    
	// parameters directly related to the EKF
	
	if (EKF_V0EST) {
		ekfp->Rn[0] = 100;
		ekfp->Rn[1] = 0;
		ekfp->Rn[2] = 0;
		ekfp->Rn[3] = 100;
		
		for (i=0;i<16;i++) ekfp->Qn[i] = 0.0;
		for (i=0;i<4;i++) ekfp->Qn[i+4*i] = 1.0; // main diagonal elements 
		
		for (i=0;i<16;i++) ekfp->P0[i] = 0.0;
		for (i=0;i<4;i++) ekfp->P0[i+4*i] = 10.0; // main diagonal elements
		
		/*ekfp->x0[0] = 0.2;
		ekfp->x0[1] = 0.4;
		ekfp->x0[2] = M_PI/8;
		ekfp->x0[3] = V0;*/
		
		ekfp->x0[0] = 0.5;
		ekfp->x0[1] = 0.0;
		ekfp->x0[2] = M_PI/4;
		ekfp->x0[3] = V0/2;
		
	} else {
		ekfp->Rn[0] = 100;
		ekfp->Rn[1] = 0;
		ekfp->Rn[2] = 0;
		ekfp->Rn[3] = 100;
		
		for (i=0;i<9;i++) ekfp->Qn[i] = 0.0;
		for (i=0;i<3;i++) ekfp->Qn[i+3*i] = 1.0; // main diagonal elements 
		
		for (i=0;i<9;i++) ekfp->P0[i] = 0.0;
		for (i=0;i<3;i++) ekfp->P0[i+3*i] = 10.0; // main diagonal elements
		
		/*ekfp->x0[0] = 0.5;
		ekfp->x0[1] = 0;
		ekfp->x0[2] = M_PI;*/
		
		ekfp->x0[0] = 0.5;
		ekfp->x0[1] = 0.0;
		ekfp->x0[2] = M_PI/4;
	}
	
	ekfp->Ts = 0.005;
	
	
	// calculate points of desiredPath
	cp->NdesPath = 100; // number of points calculated for desired path
	cp->desiredPath = (double *)calloc(cp->NdesPath*2, sizeof(double));
	deltaTheta = 2*M_PI/((double)(cp->NdesPath));
	for (i=0;i<cp->NdesPath;i++) {
	
		if (cp->path_type == 1) {
			pathEllipseData (cp->desiredPath+i*2, LFpyDummy, LF2pyDummy, LGLFpyDummy, i*deltaTheta, 0, cp->a, cp->b, cp->delta, cp->xme, cp->yme);
		} else if (cp->path_type == 2) {
			// lemniscate
			pathLemniscateData (cp->desiredPath+i*2, LFpyDummy, LF2pyDummy, LGLFpyDummy, i*deltaTheta, 0, cp->R);
		} else {
			printf("unknown path type\n");
		}
		
	}
	//for (i=0;i<cp->NdesPath;i++) printf("%g %g\n",cp->desiredPath[i*2],cp->desiredPath[i*2+1]);
    
}

void allocate_memory_controller (cntrState_t *cntrState, contrp_t *cp) {

	// allocate memory for the controller
	// executed as separate function because it only needs to be called once
	
}

