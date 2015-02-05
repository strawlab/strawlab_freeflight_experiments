function [h1,h2,h3] = do_freq_plots(iddata, f0, f1, fs, subtitle, u_name, y_name)
id = iddata.InputData(:);

if iscell(id)
    i = cell2mat(id);
    o = cell2mat(iddata.OutputData(:));
else
    i = id;
    o = iddata.OutputData(:);
end

ni = length(i);

%choose a window such that we have 20 cycles of the lowest frequency
%for our given Fs
ws = max(f0,0.1)*20*fs;

ts=1/fs;

window = hamming(ws);
noverlap = round(0.75*ws);          %75% overlap
nfft = max(256,2^nextpow2(ni));

h1 = figure(1);

mscohere(i,o,window,noverlap,nfft,fs);
xlim([0 fs/4]);
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title({['Coherence Estimate ' u_name ':' y_name],subtitle},'interpreter', 'none');

h2 = figure(2);

tfestimate(i,o,window,noverlap,nfft,fs);
xlim([0 fs/4]);
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title({['Transfer Function Estimate ' u_name ':' y_name],subtitle},'interpreter', 'none');

h3 = figure(3);

subplot(211);

%periodogram(i,[],nfft,fs);
Hs1 = spectrum.mtm(3,'adapt');
psd(Hs1,i,'Fs',fs,'NFFT',nfft)

xlim([0 fs/4]);
ylim([-50 25])
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title({'Periodogram Power Spectral Density Estimate',subtitle},'interpreter', 'none');
legend(u_name)
subplot(212);

periodogram(o,[],nfft,fs);
Hs1 = spectrum.mtm(3,'adapt');
psd(Hs1,o,'Fs',fs,'NFFT',nfft)

xlim([0 fs/4]);
ylim([-50 25])
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title('');
legend(y_name)
end
