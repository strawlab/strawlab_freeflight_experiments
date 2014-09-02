/* 
 * provide environment for controller and EKF for path-following control of the fly
 *
 * this environment aims at simulating the Python-environment where the functions are finally
 * executed in, i.e. it only provides pointers to pre-allocated memory locations
 * and manages interactions with the rest of the simulation
 * 
 */

#define S_FUNCTION_NAME environment_SFct
#define S_FUNCTION_LEVEL 2

#include "simstruc.h"
#include "mex.h"
#include <math.h>
#include <inttypes.h>

#include "initFunctions.h"

#include "contr_fct_subopt_MPC_model2.h"

#include "ekf_fct_model2.h"

#include "dec_fct.h"

#include "calculateInput.h"

#if defined(_WIN32)

    #include <Windows.h>

#else

    #include <time.h>

    #define LARGE_INTEGER uint64_t

    void QueryPerformanceFrequency(uint64_t *res) {
        *res = 1;
    }

    void QueryPerformanceCounter(uint64_t *res) {
        uint64_t nsec_count, nsec_per_tick;
        uint64_t nsec_per_sec = 1000000000;
        /*
         * clock_gettime() returns the number of secs. We translate that to number of nanosecs.
         * clock_getres() returns number of seconds per tick. We translate that to number of nanosecs per tick.
         * Number of nanosecs divided by number of nanosecs per tick - will give the number of ticks.
         */
         struct timespec ts1, ts2;

         if (clock_gettime(CLOCK_MONOTONIC, &ts1) != 0) {
             printf("clock_gettime() failed\n");
            *res = 0;
         }

         nsec_count = ts1.tv_nsec + ts1.tv_sec * nsec_per_sec;

         if (clock_getres(CLOCK_MONOTONIC, &ts2) != 0) {
             printf("clock_getres() failed\n");
             *res = 0;
         }

         nsec_per_tick = ts2.tv_nsec + ts2.tv_sec * nsec_per_sec;

         *res = (nsec_count / nsec_per_tick);
    }

#endif


// Global variables (for convenience, otherwise pointer type work vector would 
// be necessary to store the data from one call of the S function to the next): 
ekfp_t ekfp;
ekfState_t ekfState;
contrp_t cp;
projGrState_t projGrState;
decfp_t decfp;
decfState_t decfState;
cInputp_t cInputp;
cInpState_t cInpState;

/*====================*
 * S-function methods *
 *====================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumSFcnParams(S, 0);  /* Number of expected parameters */
        // all parameters set internally with C-function
    
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        return; /* Parameter mismatch will be reported by Simulink */
    }

    ssSetNumContStates(S, 0);
    ssSetNumDiscStates(S, 0);

    if (!ssSetNumInputPorts(S, 4)) return;
        // output of the system (position of fly)
		// for debug purposes: input to the fly externally given
        // for debug purposes: estimated state of external EKF
        // for debug purposes: state theta of aux. system
    
    ssSetInputPortWidth(S, 0, 2); 
    ssSetInputPortDirectFeedThrough(S, 0, 1);
    ssSetInputPortRequiredContiguous(S, 0, 1);
	
	ssSetInputPortWidth(S, 1, 1); 
    ssSetInputPortDirectFeedThrough(S, 1, 1);
    ssSetInputPortRequiredContiguous(S, 1, 1);

    ssSetInputPortWidth(S, 2, 5); 
    ssSetInputPortDirectFeedThrough(S, 2, 1);
    ssSetInputPortRequiredContiguous(S, 2, 1);
    
    ssSetInputPortWidth(S, 3, 1); 
    ssSetInputPortDirectFeedThrough(S, 3, 1);
    ssSetInputPortRequiredContiguous(S, 3, 1);

    
    if (!ssSetNumOutputPorts(S, 8)) return;
    ssSetOutputPortWidth(S, 0, 5); // estimated state of EKF
    ssSetOutputPortWidth(S, 1, 1); // omega_e from controller, input to the system
	ssSetOutputPortWidth(S, 2, 1); // cost from controller
	ssSetOutputPortWidth(S, 3, 1); // path parameter evolution
	ssSetOutputPortWidth(S, 4, 1); // input to path parameter system
    ssSetOutputPortWidth(S, 5, 1); // enable flag for controller
    ssSetOutputPortWidth(S, 6, 1); // enable flag for EKF
	ssSetOutputPortWidth(S, 7, 2); // current target point on the desired path
		

    ssSetNumSampleTimes(S, 4); 
        // first sampling time for EKF, second for the controller (suboptimal MPC)
        // third for the decision function whether or not to activate the controller, 
        // fourth for the function calculating the actual input to the system
    
    ssSetNumRWork(S, 3);
        /*
         * enable for EKF
         * enable for controller
         * current value of the input omegae for the system, needed to exchange the current value between 
		 *   controller and EKF and to hold this value over multiple calls of this S-function
         *
         */    
    
    ssSetNumIWork(S, 0); 
    ssSetNumPWork(S, 0);
    ssSetNumModes(S, 0);
    ssSetNumNonsampledZCs(S, 0);
    ssSetSimStateCompliance(S, USE_DEFAULT_SIM_STATE);

    /* Take care when specifying exception free code - see sfuntmpl_doc.c */
    //ssSetOptions(S, SS_OPTION_EXCEPTION_FREE_CODE);
}



/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *    
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    double Ts_ekf, Ts_c, Ts_d, Ts_ci;
    
    // initialize parameters of decision function, controller, 
    // calcInput function, and EKF; 
    // this is already done here because sampling times are needed and 
    // this function is called by Simulink before mdlSTART:
    init_par_cInpF_decF_ekf_subopt_MPC_model2 (&cp, &ekfp, &decfp, &cInputp);
    
    Ts_ekf = ekfp.Ts;
    Ts_c = cp.Ts;
    Ts_d = decfp.Ts;
    Ts_ci = cInputp.Ts;
    
    ssSetSampleTime(S, 0, Ts_ekf);
    ssSetOffsetTime(S, 0, 0.0);
    
    ssSetSampleTime(S, 1, Ts_c);
    ssSetOffsetTime(S, 1, 0.0);
    
    ssSetSampleTime(S, 2, Ts_d);
    ssSetOffsetTime(S, 2, 0.0);
    
    ssSetSampleTime(S, 3, Ts_ci);
    ssSetOffsetTime(S, 3, 0.0);
        
    ssSetModelReferenceSampleTimeDefaultInheritance(S);   
}

#define MDL_START                      /* Change to #undef to remove function */
#if defined(MDL_START)
/* Function: mdlStart ==========================================================
 * Abstract:
 *
 */
static void mdlStart(SimStruct *S)
{   
    double *rworkVector = ssGetRWork(S);
    double *enableEKF = rworkVector;
    double *enableContr = enableEKF+1;
    double *omegae = enableContr+1;
                    
    enableEKF[0] = 0.0;
    enableContr[0] = 0.0;
    
    omegae[0] = 0.0;
    	
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
    
    
	// CHECKS: 
	//for (i=0;i<(cp.Nu*cp.Nhor);i++) printf("u0: %g\n",cp.u0[i]);
	/*for (i=0;i<5;i++) printf("ekfp x0: %g\n",ekfp.x0[i]);
	printf("cp v0: %g\n",cp.cntr_flyparams.v0);
	printf("ekfp Vz: %g\n",ekfp.ekf_flyparams.Vz);
	mexErrMsgTxt("here");*/
    
}
#endif /*  MDL_START */


#define MDL_INITIALIZE_CONDITIONS
/* Function: mdlInitializeConditions ========================================
 * Abstract:
 *    
 */
static void mdlInitializeConditions(SimStruct *S)
{
    
}



/* Function: mdlOutputs =======================================================
 * 
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
    real_T         *xhat_out   = ssGetOutputPortRealSignal(S,0);
    real_T         *omegae_out = ssGetOutputPortRealSignal(S,1);
	real_T         *Jout       = ssGetOutputPortRealSignal(S,2);
	real_T         *theta_out  = ssGetOutputPortRealSignal(S,3);
	real_T         *w_out       = ssGetOutputPortRealSignal(S,4);
    real_T         *en_cntr_out = ssGetOutputPortRealSignal(S,5);
    real_T         *en_ekf_out  = ssGetOutputPortRealSignal(S,6);
	real_T    *targetPoint_out  = ssGetOutputPortRealSignal(S,7);
    
	real_T            *y     = (real_T*)ssGetInputPortSignal(S,0);
	real_T            *udebug     = (real_T*)ssGetInputPortSignal(S,1);
    real_T            *xestdebug     = (real_T*)ssGetInputPortSignal(S,2);
    real_T            *thetadebug     = (real_T*)ssGetInputPortSignal(S,3);
    
    double *rworkVector = ssGetRWork(S);
    double *enableEKF = rworkVector;
    double *enableContr = enableEKF+1;
    double *omegae = enableContr+1;
    	
	int i;
	
	double est_gamma_dummy[1]; // dummy variable, could contain an estimate for gamma
    
    int indexFlyToBeCntr;
    
    int doTimeMeasurement = 0;
            
    // test arguments for decision function: 
    double xFlyPos[3] = {124,95,1.6};//0.301};
    double yFlyPos[3] = {-100.0,-0.35,0.05};//0.15};
    int flyIDs[3] = {3,4,10};
    int arrayLen = 3;
    int resetDecFct = 0;
        
    // variables for time-measurement
	uint64_t pf;
    double freq_;
	uint64_t baseTime_;
	uint64_t val_bef, val_after;
	double time_bef, time_after;
    
    if (doTimeMeasurement) {
        QueryPerformanceFrequency( (LARGE_INTEGER *)&pf );
        freq_ = 1.0 / (double)pf;
		
        QueryPerformanceCounter( (LARGE_INTEGER *)&baseTime_ );
    }
    
    	   
    if (ssIsSampleHit(S, 1, tid)) {
		// sampling instant of controller is hit
                
        if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_bef );
        }
	    
        // call controller function
        contr_subopt_MPC_fly_model2 (Jout, w_out, theta_out, (int)enableContr[0], &cp, &projGrState, &ekfState, (int)enableEKF[0], &cInpState, targetPoint_out);
        
        if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_after );

            time_bef = (val_bef - baseTime_) * freq_;
            time_after = (val_after - baseTime_) * freq_;

            printf( "time controller in ms: %g \n", (time_after-time_bef)*1000);
        }
        
    }	
    
    if (ssIsSampleHit(S, 3, tid)) { // sampling instant of function calculating the
                                    // input for the system is hit
        
		if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_bef );
        }
		
        calcInput (&cInputp, &cInpState, &projGrState, &cp, omegae);
        
		if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_after );

            time_bef = (val_bef - baseTime_) * freq_;
            time_after = (val_after - baseTime_) * freq_;

            printf( "time calculate input in ms: %g \n", (time_after-time_bef)*1000);
        }
		
        // assign outputs
        omegae_out[0] = omegae[0];
        
    }
    
    if (ssIsSampleHit(S, 0, tid)) { // sampling instant of EKF is hit
        
        if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_bef );
        }
        
        // call EKF-function
		ekf_fly_model2 ((int)enableEKF[0], &ekfState, omegae[0], y, &ekfp);
        
        if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_after );

            time_bef = (val_bef - baseTime_) * freq_;
            time_after = (val_after - baseTime_) * freq_;

            printf( "time EKF in ms: %g \n", (time_after-time_bef)*1000);
        }
		
        // assign output
		for (i=0;i<5;i++) xhat_out[i] = ekfState.xest[i];
        
        en_cntr_out[0] = enableContr[0];
        en_ekf_out[0] = enableEKF[0];
        
    }
	
    
    if (ssIsSampleHit(S, 2, tid)) { // sampling instant of decision function is hit
        
		// here only one fly is considered, its position is assigned to the corresponding variables
        xFlyPos[2] = y[0];
        yFlyPos[2] = y[1];
            
        if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_bef );
        }
                
        indexFlyToBeCntr = decFct (xFlyPos, yFlyPos, flyIDs, arrayLen, resetDecFct, enableContr, enableEKF, &cp, &ekfp, &decfp, &decfState, &projGrState, &ekfState, est_gamma_dummy);
        
        if (doTimeMeasurement) {
            QueryPerformanceCounter( (LARGE_INTEGER *)&val_after );

            time_bef = (val_bef - baseTime_) * freq_;
            time_after = (val_after - baseTime_) * freq_;

            printf( "time decision function in ms: %g \n", (time_after-time_bef)*1000);
        }
        
        //printf("index fly: %d\n",indexFlyToBeCntr);
        
    }
        
    UNUSED_ARG(tid); /* not used in single tasking mode */
    
}



#undef MDL_DERIVATIVES
/* Function: mdlDerivatives =================================================
 * 
 */
static void mdlDerivatives(SimStruct *S)
{
     
}



/* Function: mdlTerminate =====================================================
 * Abstract:
 *    No termination needed, but we are required to have this routine.
 */
static void mdlTerminate(SimStruct *S)
{
    mxArray *parOut, *pm1, *pm2, *pm3, *pm4, *pm5;
    const char *field_names[] = {"a", "b", "xme", "yme", "delta"}; // Variable needed for generating the output-struct
    mwSize dims[2] = {1, 1}; // Variable needed for generating the output-struct
    double *pm1_val,*pm2_val,*pm3_val,*pm4_val,*pm5_val;
    
    parOut = mxCreateStructArray(2, dims, 5, field_names);
    
    pm1 = mxCreateDoubleMatrix (1,1, mxREAL); 
    pm1_val = mxGetPr(pm1); 
    pm2 = mxCreateDoubleMatrix (1,1, mxREAL); 
    pm2_val = mxGetPr(pm2); 
    pm3 = mxCreateDoubleMatrix (1,1, mxREAL); 
    pm3_val = mxGetPr(pm3); 
    pm4 = mxCreateDoubleMatrix (1,1, mxREAL); 
    pm4_val = mxGetPr(pm4); 
    pm5 = mxCreateDoubleMatrix (1,1, mxREAL); 
    pm5_val = mxGetPr(pm5); 
  
    pm1_val[0] = cp.a;
    pm2_val[0] = cp.b;
    pm3_val[0] = cp.xme;
    pm4_val[0] = cp.yme;
    pm5_val[0] = cp.delta;
    
    mxSetFieldByNumber(parOut, 0, 0, pm1);
    mxSetFieldByNumber(parOut, 0, 1, pm2);
    mxSetFieldByNumber(parOut, 0, 2, pm3);
    mxSetFieldByNumber(parOut, 0, 3, pm4);
    mxSetFieldByNumber(parOut, 0, 4, pm5);
    
    mexPutVariable("base", "cntrParSim", parOut);
    
    //FILE *fp;
    
    // store parameters
    /*fp = fopen("pars.dat","wb");
    fwrite(&cp,sizeof(cp),1,fp);
    fclose(fp);*/

    
    UNUSED_ARG(S); /* unused input argument */
}

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
