#define EKF_V0EST 1 // decide whether or not to use EKF with estimation of v0

#if EKF_V0EST
	#include "ekf_fct_model4_v0est.h"
#else
	#include "ekf_fct_model4.h"
#endif
