function sig_vec = compute_inv_features_td(sig_mat, eI)
    winsize = eI.winsize;
    hop = eI.hop;
    [~, nframes] = size(sig_mat); 
    sig_vec = zeros(nframes*hop+winsize-hop+1,1);
    win = hanning(winsize, 'periodic');
    for i=1:nframes
        ix = (i-1)*hop+1:(i-1)*hop+winsize;
        if eI.RealorComplex
            sig_vec(ix, 1) = sig_vec(ix,1) + win.*real(sig_mat(:,i));
        else
            sig_vec(ix, 1) = sig_vec(ix,1) + win.*sig_mat(:,i);
        end
    end
return

%%
%% Test
eI.winsize = 512; %#ok<*UNRCH>
eI.hop = 256;
eI.scf = 1.0;
eI.RealorComplex = 0;
[x, fs] = audioread('mir1k/Wavfile/dev/abjones_5_08.wav');
x = x(:,1) + x(:,2);
[DATA, mixture_spectrum, eI] = compute_features_td(x, eI);
sig_vec = compute_inv_features_td(DATA, eI);
