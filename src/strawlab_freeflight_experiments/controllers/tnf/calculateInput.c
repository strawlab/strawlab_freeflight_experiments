#include <stdlib.h>

#include "calculateInput.h"

#include "helpfunctions.h"

cInputp_t * calcinput_new_params() {
    return calloc(1, sizeof(cInputp_t));
}

cInpState_t * calcinput_new_state() {
    return calloc(1, sizeof(cInpState_t));
}

void calcInput (cInputp_t *cInputp, cInpState_t *cInpState, cntrState_t *cntrState, contrp_t *cp, double *omegae) {
    /* arguments: 
     *  cInputp: struct holding parameters of this function
     *  cInpState: struct holding several status information of this function
     *  cntrState: struct containing all the internal variables and status of the controller
     *  cp: struct containing the parameters for the controller
     *  omegae: input to the system, length 1
     *
     */
    
            
    if (cInpState->enable >= 1) {
        // Calculate input
    
        if (cInpState->newInputCalculated >= 1) {
            // new input trajectory has been calculated by the controller
        }
        
        omegae[0] = cntrState->input[0]; // use value of u in ZOH-fashion
                
    } else {
        // idle
        omegae[0] = 0.0;
    }

}

void initCalcInput (cInputp_t *cInputp, cInpState_t *cInpState) {
    
	cInpState->newInputCalculated = 0;
    cInpState->enable = 0;
    
}
