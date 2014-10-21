void MatAdd(double *C, double *A, double *B, int n1, int n2);
void MatSub(double *C, double *A, double *B, int n1, int n2);
void MatTransp(double *C, double *A, int n1, int n2);
void MatMult(double *C, double *A, double *B, int n1, int n2, int n3);

void interplin(double *varint, double *tvec, double *varvec, double tint, int Nvar, int Nvec, int searchdir);
double innerProd(double *x, double *y, double *t, int Nh, int Ncol);
double quadraticForm (double *x, double *Q, int n);
void minfct(double *amin, int *amini, double *a, int Na);
void maxfct(double *amax, int *amaxi, double *a, int Na);
void lsearch_fit2(double *kfit, double *Jfit, double *k, double *J);
void lsearch_fit3(double *kfit, double *Jfit, double *k, double *J);