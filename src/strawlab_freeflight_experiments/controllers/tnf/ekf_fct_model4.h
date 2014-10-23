#ifndef EKFP_INCL
#define EKFP_INCL

#include <math.h>

#include "flyparams.h"

typedef struct {
    double Rn[4]; // covariance matrix measurement noise
    double Qn[9]; // covariance matrix process noise
    double P0[9]; // start value for covariance matrix of estimation error
    double x0[3]; // estimated initial state of system
    double Ts; // sampling time
    flyparams_t ekf_flyparams;
}ekfp_t;

typedef struct {
    double xminus[3];  // estimated state in the EKF
    double Pminus[9]; // covariance matrix of estimation error
    double xest[3];    // provided estimated state, is equivalent to xminus, 
                       // however, xminus is considered as internal to the calculations
    double status[10]; /* array containing status information of EKF: 
                        *  0: is reset -> reset has been performed if value 1
                        *  1: in normal operation if value 1
                        */
}ekfState_t;


void ekf_fly_model4 (int enable, ekfState_t *ekfState, double omegae, double *sysOut, ekfp_t *ekfp);

void initEKF (ekfState_t *ekfState, ekfp_t *ekfp);
ekfp_t * ekf_new_params(void);
ekfState_t * ekf_new_state(void);

void fcnAsys(double *Asys,double *xplus,double u,double v0, double Ts);
void syseq(double *fTs,double *xstate,double u,double v0,double Ts);

const double *ekf_get_state_estimate(ekfState_t *ekfState, int *n);
const double *ekf_get_state_covariance(ekfState_t *ekfState, int *m, int *n);

#endif
