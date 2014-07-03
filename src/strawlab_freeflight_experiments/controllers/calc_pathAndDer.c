#include <math.h>

void pathAndDer (double *path, double *pathDer, double *pathDerDer, double theta, double a, double b, double delta, double xme, double yme) {
    
    // circle
    /*path[0] = R*cos(theta/R);
    path[1] = R*sin(theta/R);
    
    pathDer[0] = -sin(theta/R);
    pathDer[1] = cos(theta/R);*/
    
    //=====================================
    
    // lemniscate
    /*path[0] = R*cos(theta);
    path[1] = R*cos(theta)*sin(theta);
    
    pathDer[0] = -R*sin(theta);
    pathDer[1] = R*(-pow(sin(theta),2) + pow(cos(theta),2));*/
    double ct = cos(theta);
    double cd = cos(delta);
    double st = sin(theta);
    double sd = sin(delta);
    
    // ellipse
    path[0] = xme + a*ct*cd - b*st*sd;
    path[1] = yme + a*ct*sd + b*st*cd;
    
    pathDer[0] = -a*st*cd - b*ct*sd;
    pathDer[1] = -a*st*sd + b*ct*cd;
    
    pathDerDer[0] = -a*ct*cd + b*st*sd;
    pathDerDer[1] = -a*ct*sd - b*st*cd;
        
}
