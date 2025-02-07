
#include "contr_fct_TNF_model4.h"

#include "ekf_fct_model4_switch.h"

#include "dec_fct.h"

#include "calculateInput.h"

void init_par_cInpF_decF_ekf_cntr (contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, cInputp_t *cInputp, double k0, double k1, double k2, double ts_ekf, double ts_c, double ts_d, double ts_ci);

void allocate_memory_controller (cntrState_t *cntrState, contrp_t *cp);
