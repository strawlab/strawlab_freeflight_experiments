#include "simstruc.h" // needed for printf to Matlab-console
#include <math.h>

#include "ekf_fct_model2.h"

#include "helpfunctions.h"

void ekf_fly_model2 (int enable, ekfState_t *ekfState, double omegae, double *sysOut, ekfp_t *ekfp) {
    
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
    double Csyslind[10] = {1,0, 0,1, 0,0, 0,0, 0,0};
	
	double eye5[25];
    
    double CsyslindT[10];
    
	double temp1[10];
    double temp2[4];
    double temp3[4];
	double temp3_inv[4];
	double L[10];
	
    double e_q[2];
	double xplus[5];
    
    double temp4[25];
    double temp5[25];
	
	double temp6[25];
    double temp7[25];
    	
	double AsysTs[25];
	double Pplus[25];
	double Asysd[25];
	double AsysdT[25];   
    
    double f_Ts[5];
    
    double *xminus = ekfState->xminus;
    double *Pminus = ekfState->Pminus;
	
	// identity matrix of dimension 5: 
    for (i=0;i<25;i++) {    
        eye5[i] = 0.0;
    }
    for (i=0;i<5;i++) {
        eye5[i*5+i] = 1.0;
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
        		 
		MatTransp(CsyslindT, Csyslind, 2, 5);
     
		// Pminus * Csyslind' = temp1
		MatMult(temp1,Pminus,CsyslindT,5,5,2);
		 
		// Csyslind*Pminus * Csyslind' = temp2
		MatMult(temp2,Csyslind,temp1,2,5,2);
		 
		// Csyslind*Pminus * Csyslind' + Rn = temp3
		for (i=0;i<4;i++) temp3[i] = temp2[i] + ekfp->Rn[i];
		 
		// Inversion of temp3
		// Simplification: Inverse of temp3 is simply 1/main diagonal elements
		for (i=0;i<4;i++) temp3_inv[i] = 0.0;
		for (i=0;i<2;i++) {
            temp3_inv[i*2+i] = 1/temp3[i*2+i];
		}
        
        // Calculation of L
		MatMult(L,temp1,temp3_inv,5,2,2);
		 
		// Error
		for (i=0;i<2;i++) {
            e_q[i] = sysOut[i] - xminus[i];
		}
		 
		// Calculation of xplus
		for (i=0;i<5;i++) {
            xplus[i] = xminus[i];
			for (j=0;j<2;j++) {
                xplus[i] += L[i+j*5]*e_q[j];
			}
		}
		
		// Calculation of Pplus
		 
		// L*Csyslind = temp4
		MatMult(temp4,L,Csyslind,5,2,5);
		 
		// eye(5)-L*Csyslind = temp5
		MatSub(temp5, eye5, temp4, 5, 5);
		 
		MatMult(Pplus,temp5,Pminus,5,5,5);
	 
		 
		// Update of P
		fcnAsys(AsysTs,xplus,omegae,ekfp->ekf_flyparams.v0, ekfp->ekf_flyparams.mf, ekfp->ekf_flyparams.Vy, ekfp->ekf_flyparams.Ifzz, ekfp->ekf_flyparams.Vz, ekfp->Ts);
		 
		// linearized discrete system matrix
		MatAdd(Asysd,eye5,AsysTs,5,5);

		// Asysd*Pplus = temp6
		MatMult(temp6,Asysd,Pplus,5,5,5);
		 
		// Asysd*Pplus*Asysd' = temp7
		MatTransp(AsysdT, Asysd, 5, 5);
		MatMult(temp7,temp6,AsysdT,5,5,5);
		MatAdd(Pminus,temp7,ekfp->Qn,5,5);
			  
		// PREDICTION (Euler-forward)
		//---------------------------------------
		syseq(f_Ts,xplus,omegae,ekfp->ekf_flyparams.v0,ekfp->ekf_flyparams.mf, ekfp->ekf_flyparams.Vy, ekfp->ekf_flyparams.Ifzz, ekfp->ekf_flyparams.Vz, ekfp->Ts);
			  
		MatAdd(xminus,xplus,f_Ts,5,1);     
         
         
        // OUTPUT ESTIMATED STATE
		//---------------------------------------
        // output xminus which is already (due to the prediction) the state 
        // xminus for the next sampling instant -> controller is not called
        // after EKF but in principle in the next sampling instant (aside
        // from the different sampling times of controller and EKF), therefore
        // xest = xminus of the next sampling instant is correct to be used 
        // by the controller
        for (i=0;i<5;i++) {
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
        
    for (i=0;i<5;i++) {
        xminus[i] = ekfp->x0[i];
    }
    for (i=0;i<25;i++) Pminus[i] = ekfp->P0[i];
    
    for (i=0;i<5;i++) {
        xest[i] = xminus[i]; // estimate of state
    }
    
    status[0] = 1.0; // reset has been done
    status[1] = 0.0; // not in normal operation loop

}


void fcnAsys(double *Asys,double *xplus,double u,double v0, double mf, double Vy, double Ifzz, double Vz, double Ts) {
    
    // calculation of dfdx*Ts
    
double x = xplus[0];
double y = xplus[1];
double ggamma = xplus[2];
double vy = xplus[3];
double ggammad = xplus[4];

        
Asys[0] = 0;
Asys[1] = 0;
Asys[2] = 0;
Asys[3] = 0;
Asys[4] = 0;
Asys[5] = 0;
Asys[6] = 0;
Asys[7] = 0;
Asys[8] = 0;
Asys[9] = 0;
Asys[10] = Ts * (-sin(ggamma) * v0 - cos(ggamma) * vy);
Asys[11] = Ts * (cos(ggamma) * v0 - sin(ggamma) * vy);
Asys[12] = 0;
Asys[13] = 0;
Asys[14] = 0;
Asys[15] = -Ts * sin(ggamma);
Asys[16] = Ts * cos(ggamma);
Asys[17] = 0;
Asys[18] = -Ts / mf * Vy;
Asys[19] = 0;
Asys[20] = 0;
Asys[21] = 0;
Asys[22] = Ts;
Asys[23] = -Ts * v0;
Asys[24] = -Ts * Vz / Ifzz;

}

void syseq(double *fTs,double *xstate,double u,double v0, double mf, double Vy, double Ifzz, double Vz, double Ts) {

double x = xstate[0];
double y = xstate[1];
double ggamma = xstate[2];
double vy = xstate[3];
double ggammad = xstate[4];

double omegae = u;

fTs[0] = Ts * (cos(ggamma) * v0 - sin(ggamma) * vy);
fTs[1] = Ts * (sin(ggamma) * v0 + cos(ggamma) * vy);
fTs[2] = Ts * ggammad;
fTs[3] = Ts / mf * (-ggammad * mf * v0 - Vy * vy);
fTs[4] = Ts * Vz * (omegae - ggammad) / Ifzz;


}




