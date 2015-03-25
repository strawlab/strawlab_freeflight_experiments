function h1 = multi_data_psd_plot(datastruct, f0, f1, fs, subtitle)

ni = max(cell2mat(datastruct.n));
dF0 = 1;  %min resolvable frequency

%http://support.ircam.fr/docs/AudioSculpt/3.0/co/Window%20Size.html
nfft = max(256,2^nextpow2(ni));
%choose a window such that we have 5 cycles of the lowest frequency
%for our given Fs
ws = floor(5*fs/dF0);

Hs = spectrum.welch('hamming', ws)
for i=1:numel(datastruct.values)
    Hpsd=psd(Hs,datastruct.values{i},'Fs',fs,'NFFT',nfft);
    datastruct.psd{i} = Hpsd.Data;
end

%frequency is shared between all psd (because nfft and Hs are the same)
W = Hpsd.Frequencies;
%put the psd results into a PSD data object
Hallpsd = dspdata.psd(cell2mat(datastruct.psd),W,'Fs',fs);

h1 = figure;

plot(Hallpsd)
xlim([0 min(f1*2,fs/4)]);
%ylim([-20 20])
line([f1 f1],get(gca,'YLim'),'Color',[1 0 0]);
title({'Power Spectral Density Estimate' ,subtitle},'interpreter', 'none');

text(0.01,0.99,...
    sprintf('fs: %.1f ws: %d nfft: %d',fs,ws,nfft),...
    'Units','normalized','VerticalAlignment','top');

legend(datastruct.names)

end
