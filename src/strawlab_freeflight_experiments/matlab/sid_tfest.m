function [z, p, k, fitpct, fitmse, sid_model] = sid_tfest(trial_data,np,nz,ioDelay,Ts)

    Options = tfestOptions;                                    
    %Options.Display = 'on';                                    

    num = arrayfun(@(x)NaN(1,x), nz+1,'UniformOutput',false);  
    den = arrayfun(@(x)[1, NaN(1,x)],np,'UniformOutput',false);

    % Prepare input/output delay                               
    iodValue = ioDelay;
    iodFree = false;
    iodMin = 0;
    iodMax = 30;
    sysinit = idtf(num, den, Ts);
    sysinit.Structure(1,1).num.Value(1) = 0;
    sysinit.Structure(1,1).num.Free(1) = false;
    iod = sysinit.Structure.ioDelay;
    iod.Value = iodValue;
    iod.Free = iodFree;
    iod.Maximum = iodMax;
    iod.Minimum = iodMin;
    sysinit.Structure.ioDelay = iod;

    % Perform estimation using "sysinit" as template           
    tf1 = tfest(trial_data, sysinit, Options);

    fitmse = tf1.Report.Fit.MSE;
    fitpct = tf1.Report.Fit.FitPercent;

    sid_data = trial_data;
    sid_model = tf1;

    [z p k] = zpkdata(tf1);
end
