#include "simstruc.h" // needed for printf to Matlab-console

#include "calculateInput.h"

#include "helpfunctions.h"

void calcInput (cInputp_t *cInputp, cInpState_t *cInpState, projGrState_t *projGrState, contrp_t *cp, double *omegae) {
    /* arguments: 
     *  cInputp: struct holding parameters of this function
     *  cInpState: struct holding several status information of this function
     *  projGradState: struct containing all the internal variables and status of the controller
     *  cp: struct containing the parameters for the controller
     *  omegae: input to the system, length 1
     *
     */
    
    double curt; // current time in the prediction horizon of the controller
    double interpInput[2];
        
    if (cInpState->enable >= 1) {
        // Calculate input
    
        if (cInpState->newInputCalculated >= 1) {
            // new input trajectory has been calculated by the controller
            cInpState->noCalls = 0;
            cInpState->newInputCalculated = 0;
        }

        curt = cInpState->noCalls*cInputp->Ts;
        cInpState->noCalls = cInpState->noCalls + 1;

        // interpolate to get current value of input at curt: 
        interplin(interpInput, projGrState->t, projGrState->u, curt, cp->Nu, cp->Nhor, 0);    

        //omegae[0] = interpInput[0];  // use interpolation of input in the prediction horizon
        omegae[0] = projGrState->u[0]; // use first value of u in the prediction horizon in ZOH-fashion
                
    } else {
        // idle
        omegae[0] = 0.0;
    }

}

void initCalcInput (cInputp_t *cInputp, cInpState_t *cInpState) {
    
    cInpState->newInputCalculated = 0;
    cInpState->noCalls = 0;
    cInpState->enable = 0;
    
}