#include "calc_pathAndDer.h"
#include <math.h>
#include "helpfunctions.h"

#include "contr_fct_subopt_MPC_model2.h"

void sysfct(double *f,
		   double *t, double *x, double *u, contrp_t *cp)
{
    double v0 = cp->cntr_flyparams.v0;
	double mf = cp->cntr_flyparams.mf;
	double Vy = cp->cntr_flyparams.Vy;
	double Vz = cp->cntr_flyparams.Vz;
	double Ifzz = cp->cntr_flyparams.Ifzz;

	double xpos    = x[0];
    double ypos    = x[1];
    double gamma   = x[2];
    double vy      = x[3];    
    double gammad  = x[4]; 

    double theta   = x[5];
        
    double omegae  = u[0];
    double w       = u[1];

    double sg = sin(gamma);
    double cg = cos(gamma);
    
    f[0] = v0*cg - vy*sg; 
    f[1] = v0*sg + vy*cg;
    f[2] = gammad;
    f[3] = -v0*gammad - 1/mf*Vy*vy;
    f[4] = 1/Ifzz*Vz*(omegae - gammad);
    
	f[5] = w;
     
}


void sysjacx(double *dfdx,
		    double *t, double *x, double *u, contrp_t *cp)
{
	// matrix must be entered columnwise, i.e. A=[a11,a12;a21,a22] becomes
    // dfdx[0]=a11; dfdx[1]=a21; dfdx[2]=a12; dfdx[3]=a22;
        
	double v0 = cp->cntr_flyparams.v0;
	double mf = cp->cntr_flyparams.mf;
	double Vy = cp->cntr_flyparams.Vy;
	double Vz = cp->cntr_flyparams.Vz;
	double Ifzz = cp->cntr_flyparams.Ifzz;

    double gamma   = x[2];
    double vy      = x[3];

    double sg = sin(gamma);
    double cg = cos(gamma);

	int i; 

	// also elements containing zero must be filled explicitly because
	// the array is also used for other aux. variables
	// (e.g. in intadj and in dHdufct)
	
	for (i=0;i<36;i++) dfdx[i] = 0.0;

    // first column: zero
    
    // second column: zero
    
    // 3. column:
    dfdx[12] = -v0*sg - vy*cg;
    dfdx[13] = v0*cg - vy*sg;

    // 4. column:
    dfdx[18] = -sg;
    dfdx[19] = cg;
	dfdx[21] = -Vy/mf;

    // 5. column:
    dfdx[26] = 1.0;
    dfdx[27] = -v0;
    dfdx[28] = -Vz/Ifzz;
    
    // 6. column: zero

    
}

void sysjacu(double *dfdu,
		    double *t, double *x, double *u, contrp_t *cp)
{	
	double Vz = cp->cntr_flyparams.Vz;
	double Ifzz = cp->cntr_flyparams.Ifzz;

	int i;

	// also elements containing zero must be filled explicitly because
	// the array is also used for other aux. variables
	// (e.g. in intadj and in dHdufct)
	
	for (i=0;i<12;i++) dfdu[i] = 0.0;

    // 1.column:
    dfdu[4] = Vz/Ifzz;
    
    // 2.column:
    dfdu[11] = 1.0;
    
}


void icostfct(double *rhs,
		     double *t, double *x, double *u, contrp_t *cp)
{
    
    double *Qy = cp->Qy;
    double Ru = cp->Ru;
	double Rw = cp->Rw;

	double  a = cp->a;
    double  b = cp->b;
    double  delta = cp->delta;
    double  xme = cp->xme;
    double  yme = cp->yme;
	double  v0 = cp->cntr_flyparams.v0;
	double  wdes = cp->wdes;
	
	double xpos    = x[0];
    double ypos    = x[1];
    double gamma   = x[2];
    double vy      = x[3];    
    double gammad  = x[4]; 
    double theta   = x[5];

	double omegae  = u[0];
	double w       = u[1];

	double py[2],dpydtheta[2],ddpyddtheta[2];
    double qtheta;
    double temp1[2];	    
        
	
	// Calculate path-value and derivative
    pathAndDer (py, dpydtheta, ddpyddtheta, theta, a,b,delta,xme,yme);
	//printf("py, theta: %g %g, %g \n",py[0], py[1], theta);
    
	temp1[0] = xpos-py[0];
	temp1[1] = ypos-py[1];
    
    qtheta = sqrt(pow(dpydtheta[0],2) + pow(dpydtheta[1],2));

    rhs[0] = 0.5*( quadraticForm (temp1, Qy, 2) + Rw*pow(wdes-qtheta*w,2) + Ru*omegae*omegae );
    	    
}

void icostjacx(double *dldx,
		      double *t, double *x, double *u, contrp_t *cp)
{
    double *Qy = cp->Qy;
    double Rw = cp->Rw;
    	    
	double xpos    = x[0];
    double ypos    = x[1];
	double theta   = x[5];

	double  a = cp->a;
    double  b = cp->b;
    double  delta = cp->delta;
    double  xme = cp->xme;
    double  yme = cp->yme;
	double  wdes = cp->wdes;
    
    double w       = u[1];

    double py[2],dpydtheta[2],ddpyddtheta[2];
    double qtheta,dqtheta;
    double temp1[2];	
    
    // Calculate path-value and derivative
    pathAndDer (py, dpydtheta, ddpyddtheta, theta, a,b,delta,xme,yme);

	temp1[0] = xpos - py[0];
    temp1[1] = ypos - py[1];
    
    qtheta = sqrt(pow(dpydtheta[0],2) + pow(dpydtheta[1],2));
    dqtheta = 0.5/qtheta*(2*dpydtheta[0]*ddpyddtheta[0] + 2*dpydtheta[1]*ddpyddtheta[1]);

    dldx[0] = temp1[0]*Qy[0] + temp1[1]*Qy[1];
    dldx[1] = temp1[0]*Qy[2] + temp1[1]*Qy[3];

	dldx[2] = 0.0;
	dldx[3] = 0.0;
	dldx[4] = 0.0;
        
    dldx[5] = -dldx[0]*dpydtheta[0] - dldx[1]*dpydtheta[1] - (wdes - qtheta*w)*Rw*dqtheta*w;
    

    
}

void icostjacu(double *dldu,
		      double *t, double *x, double *u, contrp_t *cp)
{
	double Ru = cp->Ru;
	double Rw = cp->Rw;

	double  a = cp->a;
    double  b = cp->b;
    double  delta = cp->delta;
    double  xme = cp->xme;
    double  yme = cp->yme;
	double  v0 = cp->cntr_flyparams.v0;
	double  wdes = cp->wdes;
    double qtheta;

    double omegae  = u[0];
    double w       = u[1];
    
    double theta   = x[5];
    
    double py[2],dpydtheta[2],ddpyddtheta[2];
    
    // Calculate path-value and derivative
    pathAndDer (py, dpydtheta, ddpyddtheta, theta, a,b,delta,xme,yme);
    
    qtheta = sqrt(pow(dpydtheta[0],2) + pow(dpydtheta[1],2));
        
    dldu[0] = Ru*omegae;
    dldu[1] = -qtheta*Rw*(wdes - qtheta*w);
}

void fcostfct(double *V, double *x, contrp_t *cp)
{
       
    V[0] = 0;
       
} 
    
void fcostjacx(double *dVdx, double *x, contrp_t *cp)
{
    int i;

	for (i=0;i<6;i++) dVdx[i] = 0.0;

}

