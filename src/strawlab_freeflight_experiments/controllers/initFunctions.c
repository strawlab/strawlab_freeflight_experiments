// on linux M_PI exists if using <C99, otherwise one must define
// #define _GNU_SOURCE and then use -std=c99 with GCC
#define _USE_MATH_DEFINES

#include <stdlib.h>  // for calloc
#include <math.h> // for pi and max-function

#include "ekf_fct_model2.h"
#include "contr_fct_subopt_MPC_model2.h"
#include "dec_fct.h"
#include "calculateInput.h"
#include "calc_pathAndDer.h"

#define MAX(a,b) (((a)>(b))?(a):(b))

// Nominal parameters of fly
#define IFZZ 0.306e-12
#define MF 0.9315e-6
#define V0 0.5
#define VY 1e-4
#define VZ 1e-10


void init_par_cInpF_decF_ekf_subopt_MPC_model2 (contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, cInputp_t *cInputp) {

	/* the operator -> references to an element of a pointer to a 
     * struct, instead of, e.g., cp->Nx also (*cp).Nx could be used, but 
     * it is not common
	 */

	int i;
	double pathDerDummy[2], pathDerDerDummy[2];
	double deltaTheta;
    
    // parameters directly related to decision function
    decfp->Ts = 0.01; // sample time
    
    // parameters directly related to the function for calculating the values of the input
    cInputp->Ts = 0.0125; // sample time, 80Hz

	// system parameters for controller
	cp->cntr_flyparams.Ifzz = IFZZ;
	cp->cntr_flyparams.mf = MF;
	cp->cntr_flyparams.v0 = V0;
	cp->cntr_flyparams.Vy = VY;
	cp->cntr_flyparams.Vz = VZ;
	
	// system parameters for EKF
	ekfp->ekf_flyparams.Ifzz = IFZZ;
	ekfp->ekf_flyparams.mf = MF;
	ekfp->ekf_flyparams.v0 = V0;
	ekfp->ekf_flyparams.Vy = VY;
	ekfp->ekf_flyparams.Vz = VZ;
	
	// parameters directly related to controller
	cp->Ts = 0.15;
	cp->theta0 = 0;
	cp->Tp = 2;
	cp->Qy[0] = 10000;
	cp->Qy[1] = 0;
	cp->Qy[2] = 0;
	cp->Qy[3] = 10000;
	cp->Ru = 1;
	cp->Rw = 1;
	cp->ulim[0] = -1000;
	cp->ulim[1] = 1000;
	cp->ulim[2] = -1000;
	cp->ulim[3] = 1000;
	cp->wdes = V0;
	cp->Nx = 6;
    cp->Nu = 2;
	cp->shiftu = 0;
	cp->Ngrad = 50;
	cp->lsmin0 = 1e-6;
	cp->lsmax0 = 1e-3;
	cp->ls_min = 1e-9;
	cp->ls_max = 100;
	cp->Nhor = 500;
	cp->Nls = 3;
	// allocate memory for u0 and initialize as zero
	cp->u0 = (double *)calloc(cp->Nu*cp->Nhor, sizeof(double));
	
		
	// geometric parameters of path
	cp->a = 0.2;
	cp->b = 0.15;
	cp->delta = 0;
	cp->xme = 0;
	cp->yme = 0;
	
    // TEST
    /*cp->a = 0.55;
	cp->b = 0.25;
	cp->delta = 0.15;
	cp->xme = 1.05;
	cp->yme = 0;*/
	
    
    
	// parameters directly related to the EKF
	ekfp->Rn[0] = 100;
	ekfp->Rn[1] = 0;
	ekfp->Rn[2] = 0;
	ekfp->Rn[3] = 100;
	
	for (i=0;i<25;i++) ekfp->Qn[i] = 0.0;
	for (i=0;i<5;i++) ekfp->Qn[i+5*i] = 1.0; // main diagonal elements 
	
	for (i=0;i<25;i++) ekfp->P0[i] = 0.0;
	for (i=0;i<5;i++) ekfp->P0[i+5*i] = 10.0; // main diagonal elements
	
	ekfp->x0[0] = 0.5;
	ekfp->x0[1] = 0;
	ekfp->x0[2] = M_PI;
	ekfp->x0[3] = 0.0;
	ekfp->x0[4] = 0.0;
	
	ekfp->Ts = 0.005;
	
	// calculate points of desiredPath
	cp->NdesPath = 100; // number of points calculated for desired path
	cp->desiredPath = (double *)calloc(cp->NdesPath*2, sizeof(double));
	deltaTheta = 2*M_PI/((double)(cp->NdesPath));
	for (i=0;i<cp->NdesPath;i++) {
		pathAndDer (cp->desiredPath+i*2, pathDerDummy, pathDerDerDummy, i*deltaTheta, cp->a, cp->b, cp->delta, cp->xme, cp->yme);
	}
	//for (i=0;i<cp->NdesPath;i++) printf("%g %g\n",cp->desiredPath[i*2],cp->desiredPath[i*2+1]);
    
}

void allocate_memory_controller (projGrState_t *projGrState, contrp_t *cp) {

	// allocate memory for the arrays of the projected gradient method within the suboptimal MPC
	// executed as separate function because it only needs to be called once
	
	int Nxh = (cp->Nx)*(cp->Nhor);
	int Nuh = (cp->Nu)*(cp->Nhor);
	int Nrws_ls = (2*(cp->Nls+1));
	
	int Nrws_dHdufct = (2*(cp->Nu)+(cp->Nx)*(cp->Nu));
	int Nrws_intsys = 3*((cp->Nx)+1);
	int Nrws_intadj = 4*(cp->Nx) + (cp->Nx)*(cp->Nx);
	int Nrws_int = MAX(Nrws_intsys,Nrws_intadj);

	projGrState->t = (double *)calloc(cp->Nhor, sizeof(double));
	projGrState->x = (double *)calloc(Nxh, sizeof(double));
	projGrState->xco = (double *)calloc(Nxh, sizeof(double));
	projGrState->u = (double *)calloc(Nuh, sizeof(double));
	projGrState->dHdu = (double *)calloc(Nuh, sizeof(double));
	projGrState->s = (double *)calloc(Nuh, sizeof(double));
	projGrState->ip = (double *)calloc(3, sizeof(double));
	projGrState->ls = (double *)calloc(Nrws_ls, sizeof(double));
	projGrState->uls = (double *)calloc(Nuh, sizeof(double));
	projGrState->J = (double *)calloc(cp->Nhor, sizeof(double));
	projGrState->rws = (double *)calloc(MAX(Nrws_dHdufct,Nrws_int) + 10, sizeof(double));
	projGrState->theta = (double *)calloc(1, sizeof(double));
	projGrState->finalInput = (double *)calloc(Nuh, sizeof(double));

}

