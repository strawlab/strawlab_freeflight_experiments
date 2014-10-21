clear all
close all
clc

mex -v environment_SFct.c contr_fct_TNF_model4.c initFunctions.c...
    helpfunctions.c calc_pathsData.c calculateInput.c dec_fct.c ekf_fct_model4_v0est.c

copyfile('environment_SFct.mexw64','../')

cd ..