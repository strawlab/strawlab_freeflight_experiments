/* test the functions for controlling the fly */
 
#include <stdio.h>
#include <math.h>

#include "initFunctions.h"
#include "contr_fct_subopt_MPC_model2.h"
#include "ekf_fct_model2.h"
#include "dec_fct.h"
#include "calculateInput.h"

int main(void) {

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
	
	double Jout[1], w_out[1], theta_out[1];
	
	// initialize parameters of decision function, controller, 
    // calcInput function, and EKF; 
    init_par_cInpF_decF_ekf_subopt_MPC_model2 (&cp, &ekfp, &decfp, &cInputp);
	
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

    int i;
    for (i=0; i < 100; i++) {
	
	    // call controller function
        contr_subopt_MPC_fly_model2 (Jout, w_out, theta_out, (int)enableContr[0], &cp, &projGrState, &ekfState, (int)enableEKF[0], &cInpState);
	
	    // call function calculating the input value
	    calcInput (&cInputp, &cInpState, &projGrState, &cp, omegae);
	
	    // call EKF-function
	    ekf_fly_model2 ((int)enableEKF[0], &ekfState, omegae[0], ySystem, &ekfp);

    }
	
    return 0;
}
