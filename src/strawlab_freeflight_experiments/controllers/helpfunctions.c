#include <math.h>

void MatAdd(double *C, double *A, double *B, int n1, int n2)
{
    // matrix summation C = A+B
    // A,B,C: (n1 x n2) 
    int i;

    for ( i = 0; i <= n1*n2-1; i++ )
	  C[i] = A[i] + B[i];
}


void MatSub(double *C, double *A, double *B, int n1, int n2)
{
    // matrix subtraction C = A-B
    // A,B,C: (n1 x n2) 
    int i;

    for ( i = 0; i <= n1*n2-1; i++ )
	  C[i] = A[i] - B[i];
}

void MatTransp(double *C, double *A, int n1, int n2)
{
    // matrix transpose
    // A: (n1 x n2) 
    // C: (n2 x n1)
    int i;
    int j;
    
    for ( i = 0; i < n2; i++ ) {
        // over all columns of given matrix
        for (j = 0; j < n1; j++) {
            // over all rows of given matrix
            C[i + j*n2] = A[j+i*n1];
        }
    }
	  
}

void MatMult(double *C, double *A, double *B, int n1, int n2, int n3)
{
    // matrix multiplication C = A*B
    // A: (n1 x n2)    B: (n2 x n3)   C: (n1 x n3)
    
    // the matrices must be given and are retourned columnwise, i.e.
    // A=[a11,a12;a21,a22] becomes
    // A[0]=a11; A[1]=a21; A[2]=a12; A[3]=a22;
    
    int i,j,k;
    double sigma;

    for ( i = 0; i < n1; i++ ) {
        // over all rows of result
        for ( j = 0; j < n3; j++ ){
            // over all columns of result
            sigma = 0;

            for ( k = 0; k < n2; k++ ){
                sigma += A[i + k*n1] * B[k + j*n2];
            }
            C[i + j*n1] = sigma;
        }
    }
}


void interplin(double *varint, double *tvec, double *varvec, double tint, int Nvar, int Nvec, int searchdir)
{
    /*
     * varint ... interpolated value
     * tvec   ... time vector of data
     * varvec ... data to be interpolated (can be vector-valued)
     * tint   ... time point where data has to be interpolated
     * Nvar   ... dimension of data-vector
     * Nvec   ... Number of data points in time
     * searchdir: 
     *
        // option to determine position ioff in time vector such that tvec[ioff] <= tint <= tvec[ioff+1]
        // searchdir= 0: search in forward direction starting at ioff=0
        // searchdir=-1: search in backward direction starting at ioff=Nvec-1
     */

    
    int i;
    int ioff;
    double dtratio;
    double *var0, *var1;

    if (tint<tvec[0])
    {   // point to be interpolated is below intervall of given data -> use first point of data
        for (i=0; i<=Nvar-1; i++)
            varint[i] = varvec[i];
    }
    else if (tint>tvec[Nvec-1])
    {   // point to be interpolated is above intervall of given data -> use last point of data
        var0 = varvec + (Nvec-1)*Nvar;
        for (i=0; i<=Nvar-1; i++)
            varint[i] = var0[i];
    }
    else
    {   // interpolation can be done in a meaningful manner
        if (searchdir==0){
            // forward direction
            ioff = 0;
            while (tvec[ioff]<tint) ioff+=1;
            ioff-= 1;
        } else {
            // backward direction
            ioff = Nvec-2;
            while (tvec[ioff]>tint) ioff-=1;
        }
        //printf("ioff = %i, t = [%f, %f, %f]\n",ioff,tvec[ioff],tint,tvec[ioff+1]);
        dtratio = (tint-tvec[ioff])/(tvec[ioff+1]-tvec[ioff]);
        var0 = varvec + ioff*Nvar;
        var1 = var0 + Nvar;    
        // Calculate interpolating value varint
        for (i=0; i<=Nvar-1; i++)
            varint[i] = var0[i] + dtratio*(var1[i]-var0[i]);		
    }
}




double innerProd(double *x, double *y, double *t, int Nh, int Ncol)
{
	//  Inner product int(x'(t)*y(t),t=0..T) whereby t defines the time grid. 
	//  The integration is done by means of the trapezoidal rule. 
	//  x(t) and y(t) may be vector-valued time functions which dimension is
	//  given by Ncol. 
	//  The elements 0...Ncol-1 of x and y correspond to the first time-step, 
	//  the elements Ncol...2*Ncol-1 to the second time-step and so forth. 
	
	int i,j;
	double innerProduct;
	double h;

	double xTyi;
	double xTyip1;

	innerProduct = 0.0;
	
	for (i=0; i<Nh-1; i++) {
		// over all time instants
		
		xTyi = 0.0;
		xTyip1 = 0.0;

		// calc x'*y
		for (j=0;j<Ncol;j++) {
			xTyi += x[i*Ncol + j]*y[i*Ncol + j];
			xTyip1 += x[(i+1)*Ncol + j]*y[(i+1)*Ncol + j];
		}

		h = t[i+1]-t[i];

		innerProduct += h/2*(xTyi + xTyip1);
	}
	
	return innerProduct;

}

double quadraticForm (double *x, double *Q, int n) {
    /* Evaluates quadratic forms x'*Q*x with x \in R^n
     * and appropriate dimension of Q. Q must be symmetric. 
     */
    double res = 0.0;
    int i,j;
    
    for (i=0; i<n; i++) {
        res = res + Q[i+n*i]*x[i]*x[i];
        for (j=i+1; j<n; j++) {
            res = res + 2*x[i]*Q[i*n+j]*x[j];
        }
    }
    return res;
}

void minfct(double *amin, int *amini, double *a, int Na)
{
    int i;
    amin[0] = a[0];
    amini[0] = 0;
    for (i=1;i<=Na-1;i++)
	if (a[i]<amin[0]){
	    amin[0]  = a[i];
	    amini[0] = i;
	}
}

void maxfct(double *amax, int *amaxi, double *a, int Na)
{
    int i;
    amax[0] = a[0];
    amaxi[0] = 1;
    for (i=1;i<=Na-1;i++)
	if (a[i]>amax[0]){
	    amax[0]  = a[i];
	    amaxi[0] = i;
	}
}



void lsearch_fit2(double *kfit, double *Jfit, 
	          double *k, double *J)
{   // kfit...optimal step length
	// Jfit...optimal cost
	// k...parameter values of points to be interpolated
	// J...cost values of points to be interpolated
    double k02 = k[0]*k[0];
    double k12 = k[1]*k[1];
    double k22 = k[2]*k[2];
	// Todo: mit eigener Berechnung vergleichen
    double a0 = (J[2]*k[0]*(k[0]-k[1])*k[1]+k[2]*(J[0]*k[1]*(k[1]-k[2])+\
		 J[1]*k[0]*(-k[0]+k[2])))/((k[0]-k[1])*(k[0]-k[2])*(k[1]-k[2]));
    double a1 = (J[2]*(-k02+k12)+J[1]*(k02-k22)+J[0]*\
		 (-k12+k22))/((k[0]-k[1])*(k[0]-k[2])*(k[1]-k[2]));
    double a2 = ((J[0]-J[2])/(k[0]-k[2])+(-J[1]+J[2])/(k[1]-k[2]))/(k[0]-k[1]);
    double a2eps = 1e-6;
    double Jeps  = 1e-9;
    
    // minimum -> polynom convex?
    if (a2>=a2eps){
		kfit[0] = -a1/(2*a2);
		Jfit[0] = a0 + a1*kfit[0] + a2*kfit[0]*kfit[0];
    }
    // smallest J
    if (a2<a2eps || kfit[0]<k[0] || kfit[0]>k[2]){
		if (J[0]<=J[1]-Jeps && J[0]<=J[2]-Jeps){
			kfit[0]=k[0];
			Jfit[0]=J[0];
		} else if (J[2]<=J[0]-Jeps && J[2]<=J[1]-Jeps) {
			kfit[0]=k[2];
			Jfit[0]=J[2];
		} else {
			kfit[0]=k[1];
			Jfit[0]=J[1];	    
		}
    }
}



void lsearch_fit3(double *kfit, double *Jfit, 
		  double *k, double *J)
{
int i;
double av6264=k[0];
double av6265=k[1];
double av6269=k[2];
double av6270=-av6269;
double av6275=k[3];
double av6276=-av6275;
double av6266=-av6265;
double av6267=av6264+av6266;
double av6271=av6264+av6270;
double av6273=av6265+av6270;
double av6277=av6264+av6276;
double av6279=av6265+av6276;
double av6281=av6269+av6276;
double av6268=1/av6267;
double av6272=1/av6271;
double av6274=1/av6273;
double av6278=1/av6277;
double av6280=1/av6279;
double av6282=1/av6281;
double av6289=J[0];
double av6287=J[1];
double av6283=J[3];
double av6300=av6264*av6264;
double av6298=av6265*av6265;
double av6297=av6269*av6269;
double av6285=J[2];
double av6311=pow(av6264,3.);
double av6312=av6287*av6311;
double av6313=pow(av6265,3.);
double av6314=-(av6289*av6313);
double av6315=-av6287;
double av6316=av6289+av6315;
double av6317=pow(av6269,3.);
double av6318=av6316*av6317;
double av6319=av6312+av6314+av6318;
double av6328=pow(av6275,3.);
double av6357=-av6283;
double a0=av6268*av6272*av6274*av6278*av6280*av6282*(av6264*av6265*av6267*av6269\
*av6271*av6273*av6283+av6275*(-(av6264*av6265*av6267*av6277*av6279*av6285)+\
av6269*av6281*(av6264*av6271*av6277*av6287-av6265*av6273*av6279*av6289)));
double a1=av6268*av6272*av6274*av6278*av6280*av6282*(-(av6267*(av6265*av6269+av6\
264*(av6265+av6269))*av6271*av6273*av6283)+av6267*(av6265*av6275+av6264*(av\
6265+av6275))*av6277*av6279*av6285+av6297*(av6273*av6289*av6298+(-av6264+av\
6269)*av6287*av6300)+((av6287-av6289)*av6297+av6289*av6298-av6287*av6300)*a\
v6328+av6319*(av6275*av6275));
double a2=av6268*av6272*av6274*av6278*av6280*av6282*(av6267*(av6264+av6265+av626\
9)*av6271*av6273*av6283-av6267*(av6264+av6265+av6275)*av6277*av6279*av6285+\
av6269*(av6265*av6289*(av6297-av6298)+av6287*(-(av6264*av6297)+av6311))-av6\
275*av6319+(av6271*av6287+(av6266+av6269)*av6289)*av6328);
double a3=av6268*av6274*av6280*(av6283+av6315)+av6272*av6274*av6282*(av6285+av63\
57)+av6268*av6272*av6278*(av6289+av6357);
// minimum
double av6695=a2;
double av6693=a3;
double av6694=1/av6693;
double av6696=-av6695;
double av6697=av6695*av6695;
double av6698=a1;
double av6699=-3.*av6693*av6698;
double av6700=av6697+av6699;
double av6701=sqrt(av6700);
// kmin1 und kmin2 sind die beiden Extremalstellen des kubischen Polynoms, d.h. wo f'=0 gilt
double kmin1=0.3333333333333333*av6694*(av6696-av6701);
double kmin2=0.3333333333333333*av6694*(av6696+av6701);
double Jmin1=a0 + a1*kmin1 + a2*kmin1*kmin1 + a3*kmin1*kmin1*kmin1;
double Jmin2=a0 + a1*kmin2 + a2*kmin2*kmin2 + a3*kmin2*kmin2*kmin2;
    
// second derivative at minimum
double Jpp1=2*a2 + 6*kmin1*a3;
double Jpp2=2*a2 + 6*kmin2*a3;

//
 double Jmink; 
 // Bestimmen des Minimums der 4 Stützstellen
 minfct(&Jmink,&i,J,4);
// printf("kmin = %f\n",kmin);
// printf("Jmin = %f\n",Jmin);
// printf("Jpp = %f\n",Jpp);
 
 
 // if minimum
 if (Jpp1>=0 && Jmin1<Jmink && kmin1>=k[0] && kmin1<=k[3]){
     kfit[0] = kmin1;
     Jfit[0] = Jmin1;
 }
 else if (Jpp2>=0 && Jmin2<Jmink && kmin2>=k[0] && kmin2<=k[3]){
     kfit[0] = kmin2;
     Jfit[0] = Jmin2;     
 }
 // else smallest J
 else {
     Jfit[0] = Jmink; 
     kfit[0] = k[i];
 }


}


