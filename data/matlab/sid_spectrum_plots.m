function [h1,h2,h3,indata,outdata] = do_freq_plots(iddata, f0, f1, fs, subtitle, u_name, y_name)

id = iddata.InputData(:);

if iscell(id)
    i = cell2mat(id);
    o = cell2mat(iddata.OutputData(:));
else
    i = id;
    o = iddata.OutputData(:);
end

indata = i;
outdata = o;

ni = length(i);

dF0 = 1;  %min resolvable frequency

%http://support.ircam.fr/docs/AudioSculpt/3.0/co/Window%20Size.html
nfft = max(256,2^nextpow2(ni));
%choose a window such that we have 5 cycles of the lowest frequency
%for our given Fs
ws = floor(5*fs/dF0);

overlappct = 75;        %75% overlap

ts=1/fs;

window = hamming(ws);
noverlap = round((overlappct/100.0)*ws);

h1 = figure;

mscohere(i,o,window,noverlap,nfft,fs);
xlim([0 fs/4]);
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title({['Coherence Estimate ' u_name ':' y_name],subtitle},'interpreter', 'none');

h2 = figure;

tfestimate(i,o,window,noverlap,nfft,fs);
xlim([0 fs/4]);
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title({['Transfer Function Estimate ' u_name ':' y_name],subtitle},'interpreter', 'none');

h3 = figure;

subplot(211);

%share the psd method for input and output
Hs = spectrum.welch('hamming',ws);

Hpsdi=psd(Hs,i,'Fs',fs,'NFFT',nfft);

plot(Hpsdi)
xlim([0 fs/4]);
ylim([-50 25])
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title({'Power Spectral Density Estimate',subtitle},'interpreter', 'none');

text(0.01,0.99,...
    sprintf('fs: %.1f ws: %d nfft: %d',fs,ws,nfft),...
    'Units','normalized','VerticalAlignment','top');

legend(u_name)
subplot(212);

Hpsdo=psd(Hs,o,'Fs',fs,'NFFT',nfft);

plot(Hpsdo);
xlim([0 fs/4]);
ylim([-50 25])
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title('');
legend(y_name)
end
