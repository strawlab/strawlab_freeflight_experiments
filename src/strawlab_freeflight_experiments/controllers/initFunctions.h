
#include "contr_fct_subopt_MPC_model2.h"

#include "ekf_fct_model2.h"

#include "dec_fct.h"

#include "calculateInput.h"

void init_par_cInpF_decF_ekf_subopt_MPC_model2 (contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, cInputp_t *cInputp);

void allocate_memory_controller (projGrState_t *projGrState, contrp_t *cp);