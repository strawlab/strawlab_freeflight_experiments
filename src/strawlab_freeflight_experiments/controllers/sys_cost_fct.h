void sysfct(double *f, double *t, double *x, double *u, contrp_t *cp);
void sysjacx(double *dfdx, double *t, double *x, double *u, contrp_t *cp);
void sysjacu(double *dfdu, double *t, double *x, double *u, contrp_t *cp);
void icostfct(double *rhs, double *t, double *x, double *u, contrp_t *cp);
void icostjacx(double *dldx, double *t, double *x, double *u, contrp_t *cp);
void icostjacu(double *dldu, double *t, double *x, double *u, contrp_t *cp);
void fcostfct(double *V, double *x, contrp_t *cp);
void fcostjacx(double *dVdx, double *x, contrp_t *cp);
