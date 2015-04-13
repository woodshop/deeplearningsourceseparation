function [DATA, mixture_spectrum, eI] = compute_features_temporal(dmix, eI)
% compute features
% Spectra, log power spectra, MFCC, logmel
    winsize = eI.winsize;
    hop = eI.hop;
    scf = eI.scf;
%     windows = scf*hanning(winsize, 'periodic');
    windows = scf * ones(winsize, 1);
    nframes = floor(length(dmix)/hop);
    pad = winsize - nframes * hop;
    dmix = [dmix; zeros(pad, 1)];
    ix = repmat((1:winsize)', 1, nframes) + repmat(0:nframes-1, winsize, 1);
    DATA = bsxfun(@times, dmix(ix), windows);
    if eI.RealorComplex
        DATA = hilbert(cast(DATA, 'single'));
    end
    mixture_spectrum = DATA;
return

%% Test
eI.winsize = 512; %#ok<*UNRCH>
eI.hop = 256;
eI.scf = 1.0;
eI.RealorComplex = 1;
[x, fs] = audioread('mir1k/Wavfile/dev/abjones_5_08.wav');
x = x(:,1) + x(:,2);
[DATA, mixture_spectrum, eI] = compute_features_temporal(x, eI);
