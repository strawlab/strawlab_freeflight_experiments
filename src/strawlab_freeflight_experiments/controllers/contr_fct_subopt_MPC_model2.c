#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "contr_fct_subopt_MPC_model2.h"

#include "calculateInput.h"

#include "helpfunctions.h"
#include "sys_cost_fct.h"


void contr_subopt_MPC_fly_model2 (double *Jout, double *wout, double *thetaout, int enable, double *xest, contrp_t *cp, projGrState_t *projGrState, double *statusEKF, int enableEKF, cInpState_t *cInpState) {
   
    /* arguments: 
     *  Jout: current value of the cost functional, length 1
     *  wout: current input to the auxiliary system describing the evolution
     *        of the path parameter, length 1
     *  thetaout: current value of path parameter, has to be given separately as 
     *            projGrState->theta is simulated forwards at the end of the function
     *            and therefore, after this function is called, does not correspond
     *            any more to the value for which omegae and w were calculated for, length 1
     *  enable: 0: do nothing, 1: normal operation
     *  xest: estimated states from EKF, length 5
     *  cp: struct containing the parameters for the controller
     *  projGradState: struct containing all the internal variables and status of the controller
     *  statusEKF: array containing status information of EKF: 
     *          0: is reset -> reset has been performed if value 1
     *          1: in normal operation -> normal operation loop has been passed at least once if value 1
     *  enableEKF: enable flag of EKF, needed to determine whether EKF is in 
     *            idle state or provides regular output (estimated state is valid)
     *            statusEKF[1] cannot be used for that as the first normal run
     *            of the EKF occurs possibly after this function is called the first time
	 *  cInpState: struct holding several status information of the function calculating the input value
     */
	 
	 double vnext[2];
	 double xt[6]; // overall state (fly and path parameter)
	 int i;
	 double *theta = projGrState->theta; // path parameter
     int timeIndexTheta;
	 
	 Jout[0] = -1.0; // default-value
	 wout[0] = -1.0; // default-value
     thetaout[0] = 0.0; // default-value
    
   	
	if ((enable >= 1) && (projGrState->status[0] > 0.5) && (statusEKF[0] > 0.5) && (enableEKF >= 1)) {
		// controller has already been resetted at least once and currently it is enabled
        // furthermore, EKF has been resetted and is currently not in idle-state
        // -> EKF is in normal operation -> estimated state is valid and can be used here, 
        // controller can calculate and provide inputs for the system
        // -> normal operation of controller

        projGrState->status[1] = 1.0; // normal operation
        cInpState->enable = 1; // activate function for calculating the input value

        for (i=0;i<5;i++) xt[i] = xest[i];
        xt[5] = theta[0];

        perform_gradStep (vnext, Jout, cp, projGrState, xt);

        // Tell function which calculates the values of the input that a 
        // new input trajectory has been calculated
        cInpState->newInputCalculated = 1;

        wout[0] = vnext[1];
        thetaout[0] = theta[0];

        // Calculate state theta of the auxiliary system at the next 
        // sampling instant: 

        // Forward simulation of auxiliary system with input in zero 
        // order hold fashion which is a huge simplification for large
        // Ts, especially if the internal integration is done with a 
        // smaller step length than Ts (cause then more input values
        // are available for integrating theta!). 
        //theta[0] = theta[0] + cp->Ts*vnext[1];

        // Forward simulation is already done by integration in the controller: 
        // index for theta at approximately Ts (for speed reasons instead of interpolation): 
        timeIndexTheta = (int)ceil(cp->Ts/(cp->Tp/(cp->Nhor-1)));
                /* ceil used because for the case Ts < Tp/(Nhor-1) the value
                 * of theta at the next sampling instant has to be used, 
                 * with floor timeIndexTheta = 0 and theta always stays 
                 * at the same value. 
                 * This method works good if Ts >> Tp/(Nhor-1). 
                 * In all other cases including Ts < Tp/(Nhor-1) either 
                 * interpolation or the above method  
                 * theta[0] = theta[0] + cp->Ts*vnext[1]; 
                 * is better suited to get the value of theta at Ts!!
                 */           

        theta[0] = projGrState->x[timeIndexTheta*cp->Nx+(cp->Nx-1)];

    } else {
        // controller inactive
        projGrState->status[1] = 0.0;

        cInpState->enable = 0; // disable function for calculating the input value
    }

}


void initProjGradMethod (projGrState_t *projGrState, contrp_t *cp) {
    
    // initialize the arrays and other data of the projected gradient method within the suboptimal MPC
        
	int i,j;
    double *status = projGrState->status;
    
    for (i=0;i<10;i++) status[i] = 0.0;
			
    // time vector
    for (i=0;i<=(cp->Nhor-1);i++) projGrState->t[i]=(cp->Tp)/((cp->Nhor)-1)*i; 
    
    // line search
    for (i=0;i<=(cp->Nls-1);i++) projGrState->ls[i] = cp->lsmin0+(cp->lsmax0-cp->lsmin0)/(cp->Nls-1)*i;
    projGrState->ls[cp->Nls] = 0.5*(cp->lsmin0+cp->lsmax0);
    
    // input over horizon
    for (i=0;i<=cp->Nhor-1;i++)
	for (j=0;j<=(cp->Nu-1);j++)
	    projGrState->u[j+i*cp->Nu] = cp->u0[j+i*(cp->Nu)];
		
	(projGrState->theta)[0] = cp->theta0;
    
    status[0] = 1.0; // reset has been done
    status[1] = 0.0; // controller currently inactive
            
}


void perform_gradStep (double *unext, double *Jout, contrp_t *cp, projGrState_t *projGrState, double *xt) {

	int i,j,k,igrad;

	double *t = projGrState->t;
    double *x = projGrState->x;
    double *xco = projGrState->xco;
    double *u = projGrState->u;
    double *dHdu = projGrState->dHdu;
	double *s = projGrState->s;
	double *ip = projGrState->ip;
    double *ls = projGrState->ls;
    double *uls = projGrState->uls;
    double *J = projGrState->J;
	double *rws = projGrState->rws;
	
	double *ui;
	
	int Nx = cp->Nx; 
	int Nu = cp->Nu; 
	int Nhor = cp->Nhor; 
	double Ts = cp->Ts;
	int Nls = cp->Nls;
	double ls_max = cp->ls_max;
	double ls_min = cp->ls_min;
    
    int deltaShift,baseIndex;
	
	// initial conditions
    for (i=0;i<=Nx-1;i++) x[i]=xt[i]; // xt is overall state
    J[0] = 0.0;
	
	if ((cp->shiftu)==1){
		       
        if (Ts <= (t[1]-t[0])) { // assuming equidistant time grid
			// STRATEGY FOR SHIFTING THE INPUT TRAJECTORY WHICH ONLY WORKS IF
			// THE SAMPLING TIME Ts (SHIFT-TIME) IS SMALLER THAN OR EQUAL TO THE
			// STEP SIZE USED FOR NUMERICAL INTEGRATION WITHIN THE GRADIENT 
			// PROJECTION METHOD, i.e. THE GRID WIDTH OF t
            ui = u;
            for (i=0; i<=Nhor-2; i++){
                for (j=0; j<Nu; j++) // linear interpolation to achieve shift of Ts
                    ui[j] = ui[j] + (ui[j+Nu]-ui[j])/(t[i+1]-t[i])*Ts;
                ui += Nu;
            }
            // last element: extrapolation
            for (j=0; j<=Nu-1; j++) {
                ui[j] = ui[j] + (ui[j]-ui[j-Nu])/(t[Nhor-1]-t[Nhor-2])*Ts;
            }
            inputproj(u,t,cp);
        } else {
            // shift time is greater than step size of time grid -> simply shift
            // the input trajectory and do not perform interpolation
            deltaShift = (int)floor((Ts/(t[1]-t[0]))+0.5); // floor + 0.5 replaces round which is not available here
            
            for (i=0; i<(Nhor-deltaShift); i++) {
                for (j=0;j<Nu;j++) {
                    u[i*Nu+j] = u[i*Nu+j+deltaShift*Nu];
                }
            }
            
            baseIndex = (Nhor-deltaShift-1)*Nu;
            for (i=1;i<=deltaShift;i++) {
                // last deltaShift elements are copies of the value 
                // of u at the end of the time-horizon
                // (otherwise extrapolation would be necessary)
                for (j=0;j<Nu;j++) {
                    u[baseIndex+i*Nu+j] = u[baseIndex+j];
                }
            }
        }        
        
    }
	
	// loop over no. of gradient steps
    // -------------------------------
    for (igrad=1; igrad<=(cp->Ngrad); igrad++) {
	//printf("igrad: %d\n",igrad);
		
		// forward integration of system
		intsys(x,J,  t,u,cp, Nhor, rws);

		// final conditions for costates
        fcostjacx(xco+Nx*(Nhor-1), x+Nx*(Nhor-1), cp);
       
		// integration of adjoint system in reverse time
		intadj(xco,  t,x,u,cp, Nhor, rws);
        	
		// residual in dHdu
		dHdufct(dHdu, t,x,xco,u,cp, rws);
		
        // Calculate search direction as negative gradient direction
        for (i=0; i<Nhor; i++) {
            for (j=0; j<Nu; j++) { 
                s[i*Nu+j] = -dHdu[i*Nu+j];
            }
        }
				
		// line search: adaption of interval, ls[Nls] holds optimal step length
		if (ls[Nls]>=ls[0]+0.9*(ls[Nls-1]-ls[0]) && ls[Nls-1]<=ls_max) {
			// Optimal value for step length is nearer than 10% to the biggest interpolation point of the step length => enlarge all interpolation points
			for(i=0;i<=Nls-1;i++) ls[i]=1.5*ls[i];
			//printf(" Interval enlargement \n");
		} else if (ls[Nls]<=ls[0]+0.1*(ls[Nls-1]-ls[0]) && ls[0]>=ls_min) {
			// Optimal value for step length is nearer than 10% to the smallest interpolation point of the step length => decrease all interpolation points
			for(i=0;i<=Nls-1;i++) ls[i]=2.0/3.0*ls[i];
			//printf(" Interval reduction \n");
		}

		// Calculate u for the different values of ls (alpha_i), integrate system forwards and determine costs
		for (i=0;i<=Nls-1;i++){
			for (j=0;j<=Nhor-1;j++)
				for (k=0;k<=Nu-1;k++)
					uls[j*Nu+k] = u[j*Nu+k]+ls[i]*s[j*Nu+k];
			inputproj(uls,t,cp);
			
			intsys(x,J, t,uls,cp, Nhor, rws);
			// Calculate final cost
            fcostfct(ls+i+Nls+1, x+Nx*(Nhor-1), cp);
			// Add running cost to final cost, overall cost for ls[i] is stored in ls[i+Nls+1]
			ls[i+Nls+1] += J[Nhor-1];
			
			// Check whether intermediate iterates make sense
			if ((ls[i+Nls+1] > 1e12)) {// || mxIsInf(ls[i+Nls+1]) || mxIsNaN(ls[i+Nls+1])) {
				printf(" cost: %f \n",ls[i+Nls+1]);
				//mexErrMsgTxt("Fehler, Kosten undefiniert oder sehr groß, Zwischeniteration nicht sinnvoll ");
				//mexWarnMsgTxt("Fehler, Kosten undefiniert oder sehr groß, Zwischeniteration nicht sinnvoll \n");
			}
		}
		// curve fitting, ls[Nls] holds optimal step length according to the interpolation
		if (Nls==3)
			lsearch_fit2(ls+Nls,ls+2*Nls+1, ls,ls+Nls+1);
		if (Nls==4)
			lsearch_fit3(ls+Nls,ls+2*Nls+1, ls,ls+Nls+1);
	
		// next trajectory with optimal step length ls[Nls]
		for (j=0;j<=Nhor-1;j++)
			for (k=0;k<=Nu-1;k++)
				u[j*Nu+k] = u[j*Nu+k]+ls[Nls]*s[j*Nu+k];
		inputproj(u,t,cp);

    }
    // end of loop over no. of gradient steps
    // --------------------------------------
	
    // final forward integration of system to get the trajectory of x belonging to the returned u
    intsys(x,J,t,u,cp,Nhor,rws);
    
    // output: next control + cost value
    for (i=0;i<=Nu-1;i++)  unext[i]=u[i];
    Jout[0] = ls[2*Nls+1]; // cost from polynomial approximation
    
    /*// calculation of xnext at dt with linear interpolation
    if (outQ==1){
		i=0;
		while (t[i]<dt) i++;
        intsys(x,J, t,u,p, i+1, rws);
		interplin(xnext,t,x,dt,Nx,i+1,0); // todo: if this is needed it has to be checked
    }
    // calculation of x over whole horizon (workspace array)
    else if (outQ==2){ 
		// todo: if this is needed it has to be checked
        intsys(x,J, t,u,p, Nhor, rws);
		interplin(xnext,t,x,dt,Nx,Nhor,0);
        fcostfct(Jout, x+Nx*(Nhor-1), p);
		Jout[0] += J[Nhor-1]; // update of J if traj. are calculated anyway
    }*/

}


//
// Heun forward integrator
//
void intsys(double *x, double *J,
	    double *t, double *u, contrp_t *cp, int Nint,
	    double *rws)
// required size of rws: 3*(Nx+1)
// Nint = Nhor
{
	int Nx = cp->Nx;
	int Nu = cp->Nu;
	
    int i,j;
    double htime, h2;
    double *s1  = rws;
    double *xs1 = s1  + Nx+1;
    double *s2  = xs1 + Nx+1;
    
    double *tnow = t;
    double *xnow = x;
    double *unow = u;
    double *Jnow = J;
	
	
    	
    for (j=0;j<=Nint-2;j++)
	{
	    htime = tnow[1]-tnow[0];
		
	    h2 = 0.5*htime;
	    
	    // s1 holds dx/dt at xnow, unow
	    sysfct(s1, tnow,xnow,unow,cp); 
		// s1+Nx holds value of l at xnow, unow
	    icostfct(s1+Nx, tnow,xnow,unow,cp);
	    // explicit Euler1
	    for (i=0;i<=Nx-1;i++) xs1[i] = xnow[i]+htime*s1[i];
		
		// s2 holds dx/dt at t_{k+1}, xs1 and u_{k+1}
	    sysfct(s2, tnow+1,xs1,unow+Nu,cp); 
		
	    // xnew, trapecoidal rule of Heun-method
	    for (i=0;i<=Nx-1;i++) xnow[Nx+i] = xnow[i]+h2*(s1[i]+s2[i]);
                        
		// s2+Nx holds l at t_{k+1}, x_{k+1} and u_{k+1}
	    icostfct(s2+Nx, tnow+1,xnow+Nx,unow+Nu,cp);
	    
		// Cost is approximated with trapezoidal rule
		Jnow[1] = Jnow[0]+h2*(s1[Nx]+s2[Nx]);
		
	    // next pointers
	    tnow += 1;
	    xnow += Nx;
	    unow += Nu;
	    Jnow += 1;
	}
    
    	
} 


//
// Heun backward integrator for adjoint system
//
void intadj(double *xco,
	    double *t, double *x, double *u, contrp_t *cp, int Nint,
	    double *rws)
// required size of rws: 4*Nx+Nx^2 (s1, xcos1, s2 and dLdx need Nx, dfdx needs Nx^2)
// Nint = Nhor
{

	int Nx = cp->Nx;
	int Nu = cp->Nu;
	
    int i,j;
    double htime, h2;  
    double *s1     = rws;
    double *xcos1  = s1    + Nx;
    double *s2     = xcos1 + Nx;
    double *dLdx   = s2    + Nx;
    double *dfdx   = dLdx  + Nx;
    
	// Set pointers of xco, x, u and t to end of horizon
	double *xconow = xco + Nx*(Nint-1);
    double *xnow   = x   + Nx*(Nint-1);
    double *unow   = u   + Nu*(Nint-1);
    double *tnow   = t   + Nint-1;
	
	
    
        
    for (j=0;j<=Nint-2;j++)
	{
	    htime = tnow[-1]-tnow[0];
	    h2 = 0.5*htime;
        
        // s1 = lambdap(tnow, ...)
        adjsys(s1, tnow,xnow,xconow,unow,cp, dLdx,dfdx);     
                        
	    // explicit Euler1
	    for (i=0;i<=Nx-1;i++) xcos1[i] = xconow[i]+htime*s1[i];
		// s2 = lambdap(tnow-1, xcos1, ...)
	    adjsys(s2, tnow-1,xnow-Nx, xcos1, unow-Nu,cp, dLdx,dfdx); 
                               
	    // xconew, trapecoidal rule of Heun-method
	    for (i=0;i<=Nx-1;i++) xconow[i-Nx] = xconow[i]+h2*(s1[i]+s2[i]);
        
                
        // next pointers
	    tnow   -= 1;
	    xconow -= Nx;
	    xnow   -= Nx;
	    unow   -= Nu;
	}

}

void adjsys(double *rhs,
	    double *t, double *x, double *xco, double *u,  contrp_t *cp,
	    double *dLdx, double *dfdx)
{
    int i;

    icostjacx(dLdx, t,x,u,cp);
	    
	sysjacx(dfdx, t,x,u,cp);
	
	MatMult(rhs, xco,dfdx,1,cp->Nx,cp->Nx);
        
    for (i=0;i<=(cp->Nx-1);i++){
		rhs[i] = -dLdx[i]-rhs[i];
    }
}


void dHdufct(double *rhs,
	     double *t, double *x, double *xco, double *u,
	     contrp_t *cp, 
	     double *rws)
// required size of rws: 2*Nu+Nx*Nu 
{
	int Nx = cp->Nx;
	int Nu = cp->Nu;
	
    int i,j;
    double *dLdu = rws;
    double *dfdu = rws+Nu;
    double *tmp  = rws+Nu+Nx*Nu; // holds lambda'*df/du => Nu elements

	
	
    for (i=0;i<=(cp->Nhor-1);i++){

		icostjacu(dLdu, t+i,x+i*Nx,u+i*Nu,cp);
	
		sysjacu(dfdu, t+i,x+i*Nx,u+i*Nu,cp);
		MatMult(tmp, xco+i*Nx,dfdu,1,Nx,Nu);

		for(j=0;j<=Nu-1;j++)
			rhs[i*Nu+j] = dLdu[j]+tmp[j];	// dHdu
    }
    
}

void inputproj(double *u, double *t, contrp_t *cp)
{
    int i,j;
    int k=0;
    double umin, umax;
	int Nhor = cp->Nhor;
	int Nu = cp->Nu;
	double *ulimABS = cp->ulim;

    for (i=0;i<Nhor;i++){
		k = 0;
		for (j=0; j<Nu; j++){
			umin  = ulimABS[k];
			umax  = ulimABS[k+1];
            
			if (u[j]<umin)  {
                u[j] = umin;
            } else if (u[j]>umax) {
                u[j] = umax;    
            }
			k += 2;
		}
		// next time step
		u += Nu;
    }
}



