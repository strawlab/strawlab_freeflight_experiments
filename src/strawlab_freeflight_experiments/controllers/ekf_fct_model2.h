#ifndef EKFP_INCL
#define EKFP_INCL

#include <math.h>

#include "flyparams.h"

typedef struct {
    double Rn[4]; // covariance matrix measurement noise
    double Qn[25]; // covariance matrix process noise
    double P0[25]; // start value for covariance matrix of estimation error
    double x0[5]; // estimated initial state of system
    double Ts; // sampling time
    flyparams_t ekf_flyparams;
}ekfp_t;

typedef struct {
    double xminus[5];  // estimated state in the EKF
    double Pminus[25]; // covariance matrix of estimation error
    double xest[5];    // provided estimated state, is equivalent to xminus, 
                       // however, xminus is considered as internal to the calculations
    double status[10]; /* array containing status information of EKF: 
                        *  0: is reset -> reset has been performed if value 1
                        *  1: in normal operation if value 1
                        */
}ekfState_t;


void ekf_fly_model2 (int enable, ekfState_t *ekfState, double omegae, double *sysOut, ekfp_t *ekfp);

void initEKF (ekfState_t *ekfState, ekfp_t *ekfp);

void fcnAsys(double *Asys,double *xplus,double u,double v0, double mf, double Vy, double Ifzz, double Vz, double Ts);
void syseq(double *fTs,double *xstate,double u,double v0, double mf, double Vy, double Ifzz, double Vz, double Ts);

#endif