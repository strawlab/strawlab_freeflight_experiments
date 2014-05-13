#ifndef CP_INCL
#define CP_INCL

#include "flyparams.h"

typedef struct {
    double theta0; // initial value for path parameter
    double Ts; // sampling time
    flyparams_t cntr_flyparams;
    double Tp; // length of prediction horizon
    double a; // length of major axis one of ellipse
    double b; // length of major axis one of ellipse
    double delta; // rotation angle of ellipse
    double xme; // x-coordinate of center of ellipse
    double yme; // y-coordinate of center of ellipse
    double Qy[4]; // weighting matrix for position of fly
    double Ru; // weighting of input to fly
    double Rw; // weighting of input to auxiliary system of path parameter
    double wdes; // desired value for input to auxiliary system of path parameter
    int Nx, Nu; // number of states and inputs
    int shiftu; // shift trajectory of the input from one sampling instant to the next
    int Nhor; // number of discretization points in horizon vector
    int Nls; // no. of interpolation points in line search (3 or 4) (3 corresponds to a quadratic polynomial)
    double ls_min; // minimum line search parameter (adaptation)
    double ls_max; // maximum line search parameter (adaptation)
    double *u0; // initial trajectory for u0, length depends on number of points in horizon vector, 
                // therefore the allocation of memory is left to the initialization function
    int Ngrad; // number of gradient steps per sampling instant
    double lsmin0, lsmax0; // initial lower and upper bound of interval for approximate line search
    double ulim[4]; // lower and upper bound for inputs (omegae to the fly and w to the auxiliary system)
}contrp_t;

typedef struct {
    double *t; // pointer to time grid over prediction horizon
    double *x; // pointer to state over prediction horizon
    double *xco; // pointer to co-state over prediction horizon
    double *u; // pointer to input over prediction horizon
    double *dHdu; // pointer to dH/du over prediction horizon
	double *s; // pointer to search direction over prediction horizon
	double *ip; // pointer to help variables
    double *ls; // array for performing line search (holds step lengths)
    double *uls; // stores input over the prediction horizon during line search
    double *J; // used for calculating the cost
	double *rws; // used for storing temporary values
	double *theta; // state of the auxiliary system describing the evolution of the path parameter
    double status[10]; /* status: array containing status information of the controller: 
                        *  0: is reset -> reset has been performed if value 1
                        *  1: in normal operation if value 1
                        */
}projGrState_t;

#include "calculateInput.h"

void contr_subopt_MPC_fly_model2 (double *Jout, double *wout, double *thetaout, int enable, double *xest, contrp_t *cp, projGrState_t *projGrState, double *statusEKF, int enableEKF, cInpState_t *cInpState);

void initProjGradMethod (projGrState_t *projGrState, contrp_t *cp);

void perform_gradStep (double *unext, double *Jout, contrp_t *cp, projGrState_t *projGrState, double *xt);

void intsys(double *x, double *J, double *t, double *u, contrp_t *cp, int Nint, double *rws);

void intadj(double *xco, double *t, double *x, double *u, contrp_t *cp, int Nint, double *rws);

void adjsys(double *rhs, double *t, double *x, double *xco, double *u,  contrp_t *cp, double *dLdx, double *dfdx);

void dHdufct(double *rhs, double *t, double *x, double *xco, double *u, contrp_t *cp, double *rws);

void inputproj(double *u, double *t, contrp_t *cp);

#endif



