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
    
    // ellipse
    path[0] = xme + a*cos(theta)*cos(delta) - b*sin(theta)*sin(delta);
    path[1] = yme + a*cos(theta)*sin(delta) + b*sin(theta)*cos(delta);
    
    pathDer[0] = -a*sin(theta)*cos(delta) - b*cos(theta)*sin(delta);
    pathDer[1] = -a*sin(theta)*sin(delta) + b*cos(theta)*cos(delta);
    
    pathDerDer[0] = -a*cos(theta)*cos(delta) + b*sin(theta)*sin(delta);
    pathDerDer[1] = -a*cos(theta)*sin(delta) - b*sin(theta)*cos(delta);
        
}