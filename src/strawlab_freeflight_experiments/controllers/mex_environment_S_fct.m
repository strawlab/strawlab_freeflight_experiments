clear all
clc

if isunix
    mex -v -O CFLAGS="\$CFLAGS -std=c99" -lrt ...
        environment_SFct.c ekf_fct_model2.c contr_fct_subopt_MPC_model2.c initFunctions.c...
        helpfunctions.c sys_cost_fct.c calc_pathAndDer.c dec_fct.c calculateInput.c
else
    mex -v -O CFLAGS="\$CFLAGS -std=c99" ...
        environment_SFct.c ekf_fct_model2.c contr_fct_subopt_MPC_model2.c initFunctions.c...
        helpfunctions.c sys_cost_fct.c calc_pathAndDer.c dec_fct.c calculateInput.c
end

