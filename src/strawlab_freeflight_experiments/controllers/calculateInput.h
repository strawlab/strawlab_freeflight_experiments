#ifndef CINP_INCL
#define CINP_INCL

#include <math.h>


typedef struct {
    double Ts; // sampling time
}cInputp_t;

typedef struct {
    int noCalls;            // holds number of calls of function for calculating input
    int newInputCalculated; // flag for indication that a new input trajectory has been calculated by the controller
    int enable;             // enable calculation of input to the system, if this flag is zero, the calculated input is zero too
                            // this flag is being set by the controller function, if the controller is in idle state, it sets this
                            // flag to zero, if the controller is in normal operation, this flag is being set to one
}cInpState_t;

#include "contr_fct_subopt_MPC_model2.h"

cInputp_t * calcinput_new_params();
cInpState_t * calcinput_new_state();

void calcInput (cInputp_t *cInputp, cInpState_t *cInpState, projGrState_t *projGrState, contrp_t *cp, double *omegae);

void initCalcInput (cInputp_t *cInputp, cInpState_t *cInpState);

#endif