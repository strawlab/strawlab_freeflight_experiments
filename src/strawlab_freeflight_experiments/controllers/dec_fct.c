#include <stdlib.h>  // for calloc
#include "simstruc.h" // needed for printf to Matlab-console

#define _USE_MATH_DEFINES // for pi

#include <math.h>

#include "dec_fct.h"
#include "contr_fct_subopt_MPC_model2.h"
#include "ekf_fct_model2.h"

#define ZEROTOL 1e-5 // Numerical tolerance for a positive number to be considered as zero
#define THRES_DIST 0.3 // threshold distance in m, if a fly is closer than this distance
                       // to the path, then controlling the fly is started

int decFct (double *xpos, double *ypos, int *id, int arrayLen, int reset,
                double *enableCntr, double *enableEKF,  
            contrp_t *cp, ekfp_t *ekfp, decfp_t *decfp, decfState_t *decfState,  
            projGrState_t *projGrState, ekfState_t *ekfState, double *gammaEstimate) {
        
    
    /* arguments: 
     * xpos, ypos: coordinates of the positions of all flies currently tracked by visual system, 
     *             lengths: arrayLen
     * id: ids of the flies currently being tracked by visual system, length: arrayLen
     * arrayLen: overall number of flies currently being tracked by visual system
     * reset: reset this function, i.e. determine a new fly to be controlled, 
     *         as long as this argument is 1 this function is in reset-state
     *         when returning to zero, a new fly to be controlled is determined
     * enableCntr: 0: idle, 1: controller enabled, length 1
     * enableEKF: 0: idle, 1: EKF enabled, length 1
     * cp: struct containing the parameters for the controller
     * ekfp: struct containing the parameters for the EKF
     * decfp: struct containing the parameters for the decision function
     * decfState: struct containing the status of the decision function
     * projGrState: struct containing all the internal variables and status of the controller
     * ekfState: struct holding the state of the EKF with xminus, Pminus, xest, and status
	 * gammaEstimate: currently unused, may contain an estimate of gamma (orientation of the fly)
	 *           which can be used to initialize the third state of the EKF, length 1
     *
     * returns: id of fly to be controlled, if there isn't any suitable fly this function returns -1. 
     */
    
    int i;
    
    double *sd; // minimum distance to the path for each fly
    double *thetaMin; // values of the path parameter yielding the minimum distance corresponding to sd for each fly
    
    double xlocal, ylocal; // positions in coordinate system aligned with major axes of ellipse
    double delta = cp->delta; // rotation of ellipse
    double xme = cp->xme; // center coordinates of ellipse
    double yme = cp->yme;
    double a = cp->a; // length of major axes of ellipse
    double b = cp->b;
    double majAxesLengths[2] = {a,b};
    double queryPoint[2];   // point whose distance to ellipse shall be determined
    double closestPoint[2]; // closest point on ellipse to querypoint
    double x,y; // aux. variables
        
    double distClosestFly = 1e9; // distance of the closest fly determined so far
    int indexClosestFly = -1;    // index of the closest fly determined so far
    
    double *status = decfState->status;
       
    if (reset >= 1) {
        // determine a new fly to be controlled
        status[0] = 0.0;  // no fly to be controlled has been determined 
        status[1] = -1.0; // default-index
        
        enableCntr[0] = 0.0; // switch off controller 
        // reset controller
        initProjGradMethod (projGrState, cp);
                
        enableEKF[0] = 0.0;  // switch off EKF
        // reset EKF
        initEKF (ekfState, ekfp);
        
        return -1; 
    }    
    
    if (status[0] > 0.5) {
        // one of the previous calls of this function has already yielded a fly to be controlled
        // remain in this lock-status until this function is reset
        return (int)status[1];
    }
    // Currently no fly is being controlled
            
    if ((arrayLen <= 0)) {
        // no flies being tracked by visual system
        
        // just to be sure
        enableCntr[0] = 0.0; // switch off controller 
        enableEKF[0] = 0.0;  // switch off EKF
        // if necessary reset of controller and EKF could also be done here
        
        return -1;
    }
        
    sd = (double *)calloc(arrayLen, sizeof(double));
    thetaMin = (double *)calloc(arrayLen, sizeof(double));
    
    // Calculate minimum distance of each fly to the path with corresponding 
    // path parameter value, if the fly is inside the ellipse the distance is negative, 
    // otherwise positive: 
    for (i=0;i<arrayLen;i++) {
        
        // Position of fly under investigation
        x = xpos[i];
        y = ypos[i];
        
        // Transform fly position into local coordinate system aligned with major axes of ellipse
        xlocal = cos(delta)*(x-xme) + sin(delta)*(y-yme);
        ylocal = -sin(delta)*(x-xme) + cos(delta)*(y-yme);
        
        //printf("xlocal, ylocal: %g, %g\n",xlocal,ylocal);
        
        // Determine shortest distance of the current fly to the path
        queryPoint[0] = xlocal;
        queryPoint[1] = ylocal;
        sd[i] = distancePointToEllipse (majAxesLengths, queryPoint, closestPoint);
                        
        // Calculate path parameter value corresponding to closest point: 
        thetaMin[i] = acos(closestPoint[0]/a);
        if (closestPoint[1] < 0) {
            thetaMin[i] = -thetaMin[i];
        }
        
        //printf("thetaMin: %g\n",thetaMin[i]*180/M_PI);
        //printf("sd: %g\n",sd[i]);
        
        // Check if current fly is the one closest to the path: 
        if (fabs(sd[i]) < distClosestFly) {
            // Current fly is closer to the path than all previous ones
            distClosestFly = sd[i];
            indexClosestFly = i;
        }
        
    }
    
    // Check if the closest fly is close enough to the path such that controlling it
    // makes sense
    if (distClosestFly <= THRES_DIST) {
        
        // reset controller and EKF, take advantage of the facts that 
        // 1) the position of the fly is already known -> good for initializing the EKF
        // 2) the value of theta yielding the closest distance to the fly is known
        //       => good for initializing theta in the controller
        
        // set initial condition for theta of controller to theta closest to 
        // the current fly position
        cp->theta0 = thetaMin[indexClosestFly]; 
        // reset controller
        initProjGradMethod (projGrState, cp);
		        
                
        // set initial condition for the state xminus of the EKF to the current 
        // known position of the fly which is to be controlled
        (ekfp->x0)[0] = xpos[indexClosestFly]; 
        (ekfp->x0)[1] = ypos[indexClosestFly]; 
        // reset EKF
        initEKF (ekfState, ekfp);
        
        
        // activate controller and EKF
        enableCntr[0] = 1.0;
        enableEKF[0] = 1.0;
                
        status[0] = 1.0; // fly to be controlled is found -> no further fly is 
                         // determined until this function is reset
        status[1] = id[indexClosestFly]; // id of fly to be controlled
        
        // return ID of fly to be tracked and controlled
        return id[indexClosestFly];
        
    } else {
        // no suitable fly which could be controlled detected
        
        // just to be sure
        enableCntr[0] = 0.0; // switch off controller 
        enableEKF[0] = 0.0;  // switch off EKF
        // if necessary reset of controller and EKF could also be done here
        
        return -1;
    }

}

void initDecFct (decfp_t *decfp, decfState_t *decfState) {
    
    double *status = decfState->status;
	int i;
	
	for (i=0;i<10;i++) status[i] = 0;
    
}

double distancePointToEllipse (double *e, double *y, double *x) {
    /* Calculate the minimum distance of a query point with coordinates
     * (y[0],y[1]) to an ellipse with major axes lengths e[0] and e[1]. 
     * The point on the ellipse yielding the minimum distance has the coordinates
     * (x[0],x[1]). 
     * There aren't any restrictions on neither e nor y. 
     * 
     * Based on results from Mr. David Eberly. 
     */
    
    int reflect[2];
    int i,j;
    int permute[2],invpermute[2];
    double localE[2],localY[2];
    double localX[2];
    double distance;
    
    // Determine reflections for y to the first quadrant: 
    for (i=0;i<2;i++) {
        reflect[i] = (y[i] < 0.0);
    }
            
    // Determine the axis order for decreasing major axes lengths, i.e.
    // length of major axes I should be larger than or equal to length of 
    // major axes II -> switch axes if this is not the case. 
    if (e[0] < e[1]) {
        permute[0] = 1;
        permute[1] = 0;
    } else {
        permute[0] = 0;
        permute[1] = 1;
    }
    
    for (i=0;i<2;i++) {
        invpermute[permute[i]] = i;
    }
    
    // Do permutations and reflections if necessary: 
    for (i=0;i<2;i++) {
        j = permute[i];
        localE[i] = e[j];
        localY[i] = y[j];
        if (reflect[j]) {
            localY[i] = -localY[i];
        }
    }
    
    distance = distancePointToEllipseSpecial (localE,localY,localX);
    
    // Restore the axis order and reflections: 
    for (i=0;i<2;i++) {
        j = invpermute[i];
        if (reflect[i]) {
            localX[j] = -localX[j];
        }
        x[i] = localX[j];
    }   
    
    return distance;
}

double distancePointToEllipseSpecial (double *e,double *y,double *x) {
    /* Calculate the minimum distance of a query point with coordinates
     * (y[0],y[1]) to an ellipse with major axes lengths e[0] and e[1]. 
     * The point on the ellipse yielding the minimum distance has the coordinates
     * (x[0],x[1]). 
     * It must hold that e[0]>=e[1] and y[0]>=0 as well as y[1]>=0
     * (query point has to lie in the first quadrant which implies that also
     * the closest point x on the ellipse lies in the first quadrant). 
     * 
     * Based on results from Mr. David Eberly. 
     */
    
    double distance;
    double esqr[2], ey[2];
    double t0, t1, t;
    double r[2];
    double f;
    double d[2];
    double denom0, e0y0;
    double x0de0, x0de0sqr;
    double d0;
    
    int imax,i;
            
    if (y[1] > ZEROTOL) {
        if (y[0] > ZEROTOL) {
            // Bisect to compute the root of F(t) for t >= -e1*e1.
            esqr[0] = e[0]*e[0];
            esqr[1] = e[1]*e[1];
            ey[0] = e[0]*y[0];
            ey[1] = e[1]*y[1];
            t0 = -esqr[1] + ey[1];
            t1 = -esqr[1] + sqrt(ey[0]*ey[0] + ey[1]*ey[1]);
            t = t0;

            imax = 1000;
            for (i = 0; i < imax; i++) {
                t = (0.5)*(t0 + t1);
                //if (t == t0 || t == t1) {
                if (fabs(t0-t1) < ZEROTOL) {
                    // Bounds sufficiently close together
                    // F must be zero at t
                    break;
                }
                r[0] = ey[0]/(t + esqr[0]);
                r[1] = ey[1]/(t + esqr[1]);
                f = r[0]*r[0] + r[1]*r[1] - 1.0; // Current value of F (function whose zero-crossing is searched for)
                if (f > ZEROTOL) {
                    // at t, F is larger than zero
                    // -> set left interval boundary to t
                    t0 = t;
                }
                else if (f < -ZEROTOL) {
                    // at t, F is smaller than zero
                    // -> set right interval boundary to t
                    t1 = t;
                }
                else {
                    // F is zero at t -> finished
                    break;
                }
            }
            
            if (i>=imax-1) {
                printf("WARNING: bisection stopped because of maximum number of iterations reached!\n");
            }
                                    
            x[0] = esqr[0]*y[0]/(t + esqr[0]);
            x[1] = esqr[1]*y[1]/(t + esqr[1]);
            d[0] = x[0] - y[0];
            d[1] = x[1] - y[1];
            distance = sqrt(d[0]*d[0] + d[1]*d[1]);
        } else {
            // y0 == 0
            x[0] = 0.0;
            x[1] = e[1];
            distance = fabs(y[1] - e[1]);
        }
    } else {
        // y1 == 0
        denom0 = e[0]*e[0] - e[1]*e[1];
        e0y0 = e[0]*y[0];
        if (e0y0 < denom0) {
            // y0 is inside the subinterval.
            x0de0 = e0y0/denom0;
            x0de0sqr = x0de0*x0de0;
            x[0] = e[0]*x0de0;
            x[1] = e[1]*sqrt(fabs(1.0 - x0de0sqr)); // fabs not really necessary, just to avoid numerical problems
            d0 = x[0] - y[0];
            distance = sqrt(d0*d0 + x[1]*x[1]);
        } else {
            // y0 is outside the subinterval. The closest ellipse point has
            // x1 == 0 and is on the domain-boundary interval (x0/e0)^2 = 1.
            x[0] = e[0];
            x[1] = 0.0;
            distance = fabs(y[0] - e[0]);
        }
    }
    
    // Correct sign of the distance, if query point is outside of the 
    // ellipse, the distance shall be positive, otherwise negative: 
    if ( ( pow(y[0]/e[0],2) + pow(y[1]/e[1],2) - 1.0 ) < 0) {
        // query point is inside of the ellipse
        distance = -distance;
    }

    return distance;
}


    


double calcDistanceToEllipse (double theta, double a, double b, double x, double y) {
    // Calculate the Euclidean distance of a point (x,y) to a point on the 
    // ellipse with path parameter value theta, the lengths of the major axes
    // are given by a and b
    return sqrt(pow(x-a*cos(theta),2) + pow(y-b*sin(theta),2));
}
    
