clear all
close all
clc

[w_out,zeta_out,xi_out,xest,omegae,intstate,targetPoint] = testFunctions();

save ../test/tnf.mat w_out zeta_out xi_out xest omegae intstate targetPoint
