#ifndef DECFP_INCL
#define DECFP_INCL

#include <math.h>

#include "flyparams.h"

#include "contr_fct_TNF_model4.h"

#include "ekf_fct_model4_switch.h"


typedef struct {
    double Ts; // sampling time
}decfp_t;

typedef struct {
    double status[10]; /* status: array containing status information of the decision function:
                        *  0: has successfully determined a fly which shall be controlled if value 1
                        *  1: id of the fly to be controlled from the last successful determination  
                        */
}decfState_t;

int decFct (double *xpos, double *ypos, int *id, int arrayLen, int reset,
                double *enableCntr, double *enableEKF,  
            contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, decfState_t *decfState,  
            cntrState_t *contrState, ekfState_t *ekfState, double *gammaEstimate);

decfp_t * decfct_new_params(void);
decfState_t * decfct_new_state(void);

			
void initDecFct (decfp_t *decfp, decfState_t *decfState);

double distancePointToLemniscate (double x,double y, double *thetaMin, double R);

void pathLemniscateData_easy (double *py, double theta, double R);

double calcDistanceToEllipse (double theta, double a, double b, double x, double y);

double distancePointToEllipseSpecial (double *e,double *y,double *x);
double distancePointToEllipse (double *e, double *y, double *x);

#endif
