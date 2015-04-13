function [DATA, mixture_spectrum, eI] = compute_features_temporal(dmix, eI)
% compute features
% Spectra, log power spectra, MFCC, logmel

    winsize = eI.winsize; nFFT = eI.nFFT; hop = eI.hop; scf = eI.scf;
    windows = sin(0:pi/winsize:pi-pi/winsize);

    spectrum_mix = scf * stft(dmix, eI.nFFT ,windows, hop, eI.fs);
    mixture_spectrum=abs(spectrum_mix);

    if eI.MFCCorlogMelorSpectrum==2, %Spectrum
        DATA = scf * stft(dmix, eI.nFFT ,windows, hop, eI.fs);
        DATA=abs(DATA);
    elseif eI.MFCCorlogMelorSpectrum==3, %log power spectrum
        DATA = scf * stft(dmix, eI.nFFT ,windows, hop, eI.fs);
        DATA=abs(DATA);
        DATA=log(DATA.*DATA+eps);
    elseif eI.MFCCorlogMelorSpectrum == 4 % real valued time domain
        s = len(dmix);
        d = zeros((1+f/2),1+fix((s-f)/h));
        DATA = zeros(winsize,1+fix((s-winsize)/hop));
        c = 1;
        for b = 0:hop:(s-winsize)
            DATA(:,c) = dmix((b+1):(b+winsize));
            c = c+1;
        end
    else
        %% training features
        filename=[eI.saveDir,'dmix_temp.wav'];
        wavwrite(dmix, eI.fs, filename);

        if eI.framerate==64,
            if eI.MFCCorlogMelorSpectrum==0, %MFCC
                eI.config='mfcc_64ms_step32ms.cfg';
            elseif eI.MFCCorlogMelorSpectrum==1, %logmel
                eI.config='fbank_64ms_step32ms.cfg';
            else % spectrum
                eI.config='spectrum_64ms_step32ms.cfg';
            end
        else % framerate == 32
            if eI.MFCCorlogMelorSpectrum==0, %MFCC
                eI.config='mfcc_32ms_step16ms.cfg';
            elseif eI.MFCCorlogMelorSpectrum==1, %logmel
                eI.config='fbank_32ms_step16ms.cfg';
            else % spectrum
                eI.config='spectrum_32ms_step16ms.cfg';
            end
        end
        command=sprintf('HCopy -A -C %s%s %s%s %s%s',...
            eI.CFGPath,eI.config,eI.saveDir,'dmix_temp.wav',...
            eI.saveDir, 'train.fea');

        system(command);
        [ DATA, HTKCode ] = htkread( [eI.saveDir,'train.fea'] );
        DATA=DATA';
    end
end
