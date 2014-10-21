
#include "contr_fct_TNF_model4.h"

#include "ekf_fct_model4_switch.h"

#include "dec_fct.h"

#include "calculateInput.h"

void init_par_cInpF_decF_ekf_cntr (contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, cInputp_t *cInputp);

void allocate_memory_controller (cntrState_t *cntrState, contrp_t *cp);