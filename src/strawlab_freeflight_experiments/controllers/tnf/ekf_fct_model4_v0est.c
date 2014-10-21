#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "simstruc.h"

#include "ekf_fct_model4_switch.h"

#include "helpfunctions.h"

ekfp_t * ekf_new_params(void) {
    return calloc(1, sizeof(ekfp_t));
}

ekfState_t * ekf_new_state(void) {
    return calloc(1, sizeof(ekfState_t));
}


void ekf_fly_model4 (int enable, ekfState_t *ekfState, double omegae, double *sysOut, ekfp_t *ekfp) {
   
    /* arguments: 
     *  enable: 0: do nothing, 1: EKF is running; this flag is needed as in 
     *              the normal operation this function is called with ekfp->Ts. 
     *              However, if, e.g., a new fly to be controlled is determined, 
     *              this function must not do anything. 
     *  ekfState: struct holding the state of the EKF with xminus, Pminus, xest, and status
     *  omegae: current input to the system (rotation rate of environment)
     *  sysOut: current measured output of the system (planar position of the fly), length 2
     *  ekfp: struct containing the parameters for the EKF
     *
     */
    
	// *******************************************
	// VARIABLES
	// *******************************************
	
    int i,j;
	
	// output matrix 
    double Csyslind[8] = {1,0, 0,1, 0,0,0,0};
	
	double eye4[16];
    
    double CsyslindT[8];
    
	double temp1[8];
    double temp2[4];
    double temp3[4];
	double temp3_inv[4];
	double detTemp3;
	double L[8];
	
    double e_q[2];
	double xplus[4];
    
    double temp4[16];
    double temp5[16];
	
	double temp6[16];
    double temp7[16];
    	
	double AsysTs[16];
	double Pplus[16];
	double Asysd[16];
	double AsysdT[16];   
    
    double f_Ts[4];
    
    double *xminus = ekfState->xminus;
    double *Pminus = ekfState->Pminus;
	
	// identity matrix of dimension 4: 
    for (i=0;i<16;i++) {
        eye4[i] = 0.0;
    }
    for (i=0;i<4;i++) {
        eye4[i*4+i] = 1.0;
    }
	
	    
	// *******************************************
	// EXECUTION
	// *******************************************

    // BE AWARE: 
    // an initial reset of the EKF (by calling initEKF) is required (xest set to x0_guess) 
    // before the controller is called, as with this initial state the controller 
    // has to calculate the value of the input to the system and with that the EKF
    // can estimate the state of the system
    
    if ((enable >= 1) && (ekfState->status[0] > 0.5)) {
        // EKF enabled and has been reset at least once -> normal operation
        
		ekfState->status[1] = 1.0; // normal operation of the EKF has been done at least once

        // CORRECTION
		//---------------------------------------
        		 
		MatTransp(CsyslindT, Csyslind, 2, 4);
     
		// Pminus * Csyslind' = temp1
		MatMult(temp1,Pminus,CsyslindT,4,4,2);
						 
		// Csyslind*Pminus * Csyslind' = temp2
		MatMult(temp2,Csyslind,temp1,2,4,2);
						 
		// Csyslind*Pminus * Csyslind' + Rn = temp3
		for (i=0;i<4;i++) temp3[i] = temp2[i] + ekfp->Rn[i];
		 
		// Inversion of temp3
		detTemp3 = temp3[0]*temp3[3] - temp3[1]*temp3[2];
		if (fabs(detTemp3) < 1e-3) printf("temp3 almost singular!\n");
	
		temp3_inv[0] = 1/detTemp3*temp3[3];
		temp3_inv[3] = 1/detTemp3*temp3[0];
		temp3_inv[1] = -1/detTemp3*temp3[1];
		temp3_inv[2] = -1/detTemp3*temp3[2];
		
        // Calculation of L
		MatMult(L,temp1,temp3_inv,4,2,2);
		 
		// Error
		for (i=0;i<2;i++) {
            e_q[i] = sysOut[i] - xminus[i];
		}
		 
		// Calculation of xplus
		for (i=0;i<4;i++) {
            xplus[i] = xminus[i];
			for (j=0;j<2;j++) {
                xplus[i] += L[i+j*4]*e_q[j];
			}
		}
				
		// Calculation of Pplus
		 
		// L*Csyslind = temp4
		MatMult(temp4,L,Csyslind,4,2,4);
		 
		// eye(4)-L*Csyslind = temp5
		MatSub(temp5, eye4, temp4, 4, 4);
		 
		MatMult(Pplus,temp5,Pminus,4,4,4);
	 
		// Update of P
		fcnAsys(AsysTs,xplus,omegae,ekfp->Ts);
		 
		// linearized discrete system matrix
		MatAdd(Asysd,eye4,AsysTs,4,4);

		// Asysd*Pplus = temp6
		MatMult(temp6,Asysd,Pplus,4,4,4);
		
		// Asysd*Pplus*Asysd' = temp7
		MatTransp(AsysdT, Asysd, 4, 4);
		MatMult(temp7,temp6,AsysdT,4,4,4);
		MatAdd(Pminus,temp7,ekfp->Qn,4,4);
			
		
		// PREDICTION (Euler-forward)
		//---------------------------------------
		syseq(f_Ts,xplus,omegae,ekfp->Ts);
					  
		MatAdd(xminus,xplus,f_Ts,4,1);  
		         
         
        // OUTPUT ESTIMATED STATE
		//---------------------------------------
        // output xminus which is already (due to the prediction) the state 
        // xminus for the next sampling instant -> controller is not called
        // after EKF but in principle in the next sampling instant (aside
        // from the different sampling times of controller and EKF), therefore
        // xest = xminus of the next sampling instant is correct to be used 
        // by the controller
        for (i=0;i<4;i++) { 
           (ekfState->xest)[i] = xminus[i];
        }
        
    } else {
        ekfState->status[1] = 0.0; // idle
    }

}

void initEKF (ekfState_t *ekfState, ekfp_t *ekfp) {
        
    // reset internal variables of EKF
        
    int i;
    double *xminus = ekfState->xminus;
    double *Pminus = ekfState->Pminus;
    double *xest = ekfState->xest;
    double *status = ekfState->status;
    
    for (i=0;i<10;i++) status[i] = 0.0;
        
    for (i=0;i<4;i++) {
        xminus[i] = ekfp->x0[i];
    }
    for (i=0;i<16;i++) Pminus[i] = ekfp->P0[i];
    
    for (i=0;i<4;i++) {
        xest[i] = xminus[i]; // estimate of state
    }
    
    status[0] = 1.0; // reset has been done
    status[1] = 0.0; // not in normal operation loop

}


void fcnAsys(double *Asys,double *xplus,double u, double Ts) {
    
    // calculation of dfdx*Ts
    
double x = xplus[0];
double y = xplus[1];
double ggamma = xplus[2];
double v0 = xplus[3];

        
Asys[0] = 0;
Asys[1] = 0;
Asys[2] = 0;
Asys[3] = 0;

Asys[4] = 0;
Asys[5] = 0;
Asys[6] = 0;
Asys[7] = 0;

Asys[8] = Ts * (-sin(ggamma) * v0);
Asys[9] = Ts * (cos(ggamma) * v0);
Asys[10] = 0;
Asys[11] = 0;

Asys[12] = Ts * cos(ggamma);
Asys[13] = Ts * sin(ggamma);
Asys[14] = 0.0;
Asys[15] = 0.0;

}

void syseq(double *fTs,double *xstate,double u,double Ts) {

double x = xstate[0];
double y = xstate[1];
double ggamma = xstate[2];
double v0 = xstate[3];

double omegae = u;

fTs[0] = Ts * (cos(ggamma) * v0);
fTs[1] = Ts * (sin(ggamma) * v0);
fTs[2] = Ts * omegae;
fTs[3] = 0.0;


}

const double *ekf_get_state_estimate(ekfState_t *ekfState, int *n) {
    *n = 4;
    return ekfState->xest;
}

const double *ekf_get_state_covariance(ekfState_t *ekfState, int *m, int *n) {
    *m = *n = 4;
    return ekfState->Pminus;
}




