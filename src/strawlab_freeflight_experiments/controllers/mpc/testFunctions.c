/* test the functions for controlling the fly */
 
#include "simstruc.h"
#include "mex.h"
#include <math.h>

#include "initFunctions.h"

#include "contr_fct_subopt_MPC_model2.h"

#include "ekf_fct_model2.h"

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
	projGrState_t projGrState;
	decfp_t decfp;
	decfState_t decfState;
	cInputp_t cInputp;
	cInpState_t cInpState;
	  
	double storageVars[3]; // holds enable for EKF and controller as well as current omegae (input to system)
	double *enableEKF = storageVars;
	double *enableContr = enableEKF+1;
	double *omegae = enableContr+1;
	
	double ySystem[2] = {0.25,0}; // fictitious values of the output y of the system
	
	double Jout[1], w_out[1], theta_out[1], targetPoint_out[2];
	
	double *ureturn, *xreturn, *xcoreturn, *Jreturn, *wreturn, *thetareturn, *xestreturn, *omegaereturn, *Pminusreturn, *targetPointreturn;
	
	int i;
	
	// Check for proper number of arguments 
	if (nrhs != 0 ) {
		mexErrMsgTxt ("testFunctions requires 0 input arguments.");
	} 
	  
    double Ts_d     = 0.01;     //100Hz
    double Ts_ci    = 0.0125;   //80Hz
    double Ts_c     = 0.05;     //20Hz
    double Ts_ekf   = 0.005;    //200Hz
    init_par_cInpF_decF_ekf_subopt_MPC_model2 (&cp, &ekfp, &decfp, &cInputp, Ts_ekf, Ts_c, Ts_d, Ts_ci);
	
	// allocate memory for internal controller variables:
	allocate_memory_controller (&projGrState, &cp);
    
    // initialize EKF: 
    initEKF (&ekfState, &ekfp);
    
    // initialize controller
    initProjGradMethod (&projGrState, &cp);
    
    // initialize function calculating the input to the system
    initCalcInput (&cInputp, &cInpState);
	
	// initialize decision function
	initDecFct (&decfp, &decfState);
	
	
	// enable EKF and controller
	enableEKF[0] = 1.0;
	enableContr[0] = 1.0;
	
	// call controller function
    contr_subopt_MPC_fly_model2 (Jout, w_out, theta_out, (int)enableContr[0], &cp, &projGrState, &ekfState, (int)enableEKF[0], &cInpState, targetPoint_out);
	
	// call function calculating the input value
	calcInput (&cInputp, &cInpState, &projGrState, &cp, omegae);
	
	// call EKF-function
	ekf_fly_model2 ((int)enableEKF[0], &ekfState, omegae[0], ySystem, &ekfp);
	
	
	// Return interesting results of the functions
	plhs[0] = mxCreateDoubleMatrix (cp.Nu,cp.Nhor, mxREAL);
	ureturn = mxGetPr(plhs[0]);
	
	plhs[1] = mxCreateDoubleMatrix (cp.Nx,cp.Nhor, mxREAL);
	xreturn = mxGetPr(plhs[1]);
	
	plhs[2] = (mxCreateDoubleMatrix (cp.Nx,cp.Nhor, mxREAL));
	xcoreturn = mxGetPr(plhs[2]);
	
	plhs[3] = (mxCreateDoubleMatrix (1,1, mxREAL));
	Jreturn = mxGetPr(plhs[3]);
	
	plhs[4] = (mxCreateDoubleMatrix (1,1, mxREAL));
	wreturn = mxGetPr(plhs[4]);
	
	plhs[5] = (mxCreateDoubleMatrix (1,1, mxREAL));
	thetareturn = mxGetPr(plhs[5]);
	
	plhs[6] = (mxCreateDoubleMatrix (5,1, mxREAL));
	xestreturn = mxGetPr(plhs[6]);
	
	plhs[7] = (mxCreateDoubleMatrix (1,1, mxREAL));
	omegaereturn = mxGetPr(plhs[7]);
	
	plhs[8] = (mxCreateDoubleMatrix (5,5, mxREAL));
	Pminusreturn = mxGetPr(plhs[8]);	

	plhs[9] = (mxCreateDoubleMatrix (2,1, mxREAL));
	targetPointreturn = mxGetPr(plhs[9]);

	
	for (i=0;i<(cp.Nu*cp.Nhor);i++) ureturn[i] = projGrState.u[i];
	for (i=0;i<(cp.Nx*cp.Nhor);i++) xreturn[i] = projGrState.x[i];
	for (i=0;i<(cp.Nx*cp.Nhor);i++) xcoreturn[i] = projGrState.xco[i];
	Jreturn[0] = Jout[0];
	wreturn[0] = w_out[0];
	thetareturn[0] = theta_out[0];
	for (i=0;i<5;i++) xestreturn[i] = ekfState.xest[i];
	omegaereturn[0] = omegae[0];
	for (i=0;i<25;i++) Pminusreturn[i] = ekfState.Pminus[i];
	for (i=0;i<2;i++) targetPointreturn[i] = targetPoint_out[i];
	
  
}
