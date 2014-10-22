clear all
close all
clc

[u,x,xco,Jout,w_out,theta_out,xest,omegae,Pminus,targetPoint] = testFunctions();

save ../test/mpc.mat u x xco Jout w_out theta_out xest omegae Pminus targetPoint
