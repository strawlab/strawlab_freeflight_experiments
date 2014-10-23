/* test the functions for controlling the fly */
 
#include "simstruc.h"
#include "mex.h"
#include <math.h>

#include "initFunctions.h"

#include "contr_fct_TNF_model4.h"

#include "ekf_fct_model4_switch.h"

#include "dec_fct.h"

#include "calculateInput.h"

 
void mexFunction (
                  int nlhs,       mxArray *plhs[],
                  int nrhs, const mxArray *prhs[]
                  )
{

	// State and parameter structs for all functions
	ekfp_t ekfp;
	ekfState_t ekfState;
	contrp_t cp;
	cntrState_t cntrState;
	decfp_t decfp;
	decfState_t decfState;
	cInputp_t cInputp;
	cInpState_t cInpState;


	  
	double storageVars[3]; // holds enable for EKF and controller as well as current omegae (input to system)
	double *enableEKF = storageVars;
	double *enableContr = enableEKF+1;
	double *omegae = enableContr+1;
	
	double ySystem[2] = {0.25,0}; // fictitious values of the output y of the system
	
	double xi_out[1], w_out[1], zeta_out[2], targetPoint_out[2];

    double xrealdebug[1], zetadebug[1];
	
	double *wreturn, *zetareturn, *xireturn, *xestreturn, *omegaereturn, *intstatereturn, *targetPointreturn;
	
	int i;
	
	// Check for proper number of arguments 
	if (nrhs != 0 ) {
		mexErrMsgTxt ("testFunctions requires 0 input arguments.");
	} 
	  
	// initialize parameters of decision function, controller, 
    // calcInput function, and EKF; 
    init_par_cInpF_decF_ekf_cntr (&cp, &ekfp, &decfp, &cInputp); 
	
	// allocate memory for internal controller variables:
	allocate_memory_controller (&cntrState, &cp);
    
    // initialize EKF: 
    initEKF (&ekfState, &ekfp);
    
    // initialize controller
    initController (&cntrState, &cp);
    
    // initialize function calculating the input to the system
    initCalcInput (&cInputp, &cInpState);
	
	// initialize decision function
	initDecFct (&decfp, &decfState);
	
	// enable EKF and controller
	enableEKF[0] = 1.0;
	enableContr[0] = 1.0;
	
	// call controller function
    contr_TNF_fly_model4 (w_out, zeta_out, xi_out, (int)enableContr[0], &cp, &cntrState, &ekfState, (int)enableEKF[0], &cInpState, targetPoint_out, xrealdebug, zetadebug);
	
	// call function calculating the input value
    calcInput (&cInputp, &cInpState, &cntrState, &cp, omegae);
	
	// call EKF-function
	ekf_fly_model4 ((int)enableEKF[0], &ekfState, omegae[0], ySystem, &ekfp);
	
	// Return interesting results of the functions
	plhs[0] = (mxCreateDoubleMatrix (1,1, mxREAL));
	wreturn = mxGetPr(plhs[0]);
	
	plhs[1] = (mxCreateDoubleMatrix (2,1, mxREAL));
	zetareturn = mxGetPr(plhs[1]);
	
	plhs[2] = (mxCreateDoubleMatrix (4,1, mxREAL));
	xireturn = mxGetPr(plhs[2]);

#if EKF_V0EST
	plhs[3] = (mxCreateDoubleMatrix (4,1, mxREAL));
#else
	plhs[3] = (mxCreateDoubleMatrix (3,1, mxREAL));
#endif
	xestreturn = mxGetPr(plhs[3]);

	plhs[4] = (mxCreateDoubleMatrix (1,1, mxREAL));
	omegaereturn = mxGetPr(plhs[4]);

	plhs[5] = (mxCreateDoubleMatrix (2,1, mxREAL));
	intstatereturn = mxGetPr(plhs[5]);

	plhs[6] = (mxCreateDoubleMatrix (2,1, mxREAL));
	targetPointreturn = mxGetPr(plhs[6]);

    wreturn[0] = w_out[0];
    for(i=0;i<2;i++) zetareturn[i] = zeta_out[i];
    for(i=0;i<4;i++) xireturn[i] = xi_out[i];
#if EKF_V0EST
    for (i=0;i<4;i++) xestreturn[i] = ekfState.xest[i];
#else
    for (i=0;i<3;i++) xestreturn[i] = ekfState.xest[i];
#endif
    omegaereturn[0] = omegae[0];
    for (i=0;i<2;i++) intstatereturn[i] = cntrState.intState[i];
    for (i=0;i<2;i++) targetPointreturn[i] = targetPoint_out[i];
  
}
