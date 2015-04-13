function  varargout = formulate_data_time_domain(train_files, eI, mode)
% (TODO) add
% eI.winSize: Size of window
% eI.seqLen: unique lengths (in ascending order)
%             files are chopped by these lengths. (ex: [1, 10, 100])
% eI.targetWhiten: Specify the path to the whitening data.
% data_ag: noisy data. cell array of different training lengths.
% target_ag: clean data. cell array of different training lengths.

% mode:
%     0: Training (noisy data and clean data)
%     1: Testing (just noisy data)
%     2: Error testing (noisy data and clean data, both loaded without chunking)


%% Testing code
% input_fnames = {};
unique_lengths = [];

%% Set up. During testing, dont know the lengths so cant pre-allocate
%#ok<*AGROW>
if mode,
    data_ag = {};
    target_ag = {};  % returns empty targets
    mixture_ag = {};
    item=0;
else % training -- chunk
    seqLenSizes = zeros(1,length(eI.seqLen));
    for ifile = 1:numel(train_files)
        if (strcmp(train_files(ifile).name,'.') || ...
                strcmp(train_files(ifile).name,'..'))
            continue
        end
        [train_wav,fs] = audioread([eI.DataPath, 'train', filesep, ...
            train_files(ifile).name]);
        eI.fs=fs;
        train1 = train_wav(:,2); % singing
        train2 = train_wav(:,1); % music
        
        maxLength=max([length(train1), length(train2)]);
        train1(end+1:maxLength)=eps;
        train2(end+1:maxLength)=eps;
        
        train1=train1./sqrt(sum(train1.^2));
        train2=train2./sqrt(sum(train2.^2));
        
        shift_size = min(eI.circular_step, numel(train2)-1);
        fprintf('Allocating training data: %.2f\n', ...
            ifile/numel(train_files));
        for ioffset = 1:shift_size:numel(train2)-shift_size % circle shift
            fprintf('\tShift: %d\n', ioffset)
            train2_shift = [train2(ioffset: end); train2(1: ioffset-1)];
            %             fprintf('%d %d, %d %d, %d\n', ioffset, numel(train2), 1, ...
            %                 ioffset-1, numel(train2_shift));
            dmix = train1 + train2_shift;
            
            % input feature calculate
            [DATA, mixture_spectrum, eI] = compute_features_temporal(dmix, eI);
            
            [T, ~] = size(DATA');
            nfeat = size(mixture_spectrum,1);
            
            remainder = T;
            for i=length(eI.seqLen):-1:1
                num = floor(remainder/eI.seqLen(i));
                remainder = mod(remainder,eI.seqLen(i));
                seqLenSizes(i) = seqLenSizes(i)+num;
            end
        end
        
    end
    
    % Compute necessary memory before we crash
    n = 0;
    for i = length(eI.seqLen):-1:1
        n = n + eI.inputDim*eI.seqLen(i) * seqLenSizes(i);
        n = n + nfeat * eI.seqLen(i) * seqLenSizes(i);
        if eI.cleanonly == 1,
            n = n + nfeat * eI.seqLen(i) * seqLenSizes(i);
        else
            n = n + 2 * nfeat * eI.seqLen(i) * seqLenSizes(i);
        end
    end    
    fprintf('Memory needed: approx %.2f GB\n', n * 8 / 1024 / 1024 / 1024);

    data_ag = cell(1,length(eI.seqLen));
    target_ag = cell(1,length(eI.seqLen));
    mixture_ag = cell(1,length(eI.seqLen));

    for i=length(eI.seqLen):-1:1
        data_ag{i} = zeros(eI.inputDim*eI.seqLen(i),seqLenSizes(i));
        mixture_ag{i} = zeros(nfeat*eI.seqLen(i),seqLenSizes(i));
        if eI.cleanonly==1,
            target_ag{i} = zeros(nfeat*eI.seqLen(i),seqLenSizes(i));
        else
            target_ag{i} = zeros(2*nfeat*eI.seqLen(i),seqLenSizes(i));
        end    
    end
end

seqLenPositions = ones(1,length(eI.seqLen));
% winsize=eI.winsize;
% nFFT= eI.nFFT;
% hop= eI.hop;
% scf=eI.scf;
% windows=sin(0:pi/winsize:pi-pi/winsize);

for ifile=1:numel(train_files)
    if (strcmp(train_files(ifile).name,'.') || ...
            strcmp(train_files(ifile).name,'..'))
        continue
    end
    [train_wav,fs] = audioread([eI.DataPath, 'train', filesep, ...
        train_files(ifile).name]);
    eI.fs = fs;
    train1 = train_wav(:,2); % singing
    train2 = train_wav(:,1); % music
    
    maxLength = max([length(train1), length(train2)]);
    train1(end+1:maxLength) = eps;
    train2(end+1:maxLength) = eps;
    
    train1=train1./sqrt(sum(train1.^2));
    train2=train2./sqrt(sum(train2.^2));
    
    shift_step = min(eI.circular_step,numel(train2)-1);
    fprintf('Loading training data: %.2f\n', ifile/numel(train_files));
    for ioffset=1:shift_step:numel(train2)-shift_step % circle shift
        fprintf('\tShift: %d\n', ioffset)
        train2_shift = [train2(ioffset: end); train2(1: ioffset-1)];
        dmix = train1 + train2_shift;
        [DATA, mixture_spectrum, eI] = compute_features_temporal(dmix, eI);
        
        spectrum.signal = compute_features_temporal(train1, eI);
        spectrum.noise = compute_features_temporal(train2_shift, eI);
        
        multi_data = DATA;
        [~,T] = size(multi_data);
        
        %% input normalize
        if eI.inputL1 == 1, % DATA (NUMCOFS x nSamp)
            % apply CMVN to targets (normalize such that the freq bean equal to zero mean var 1)
            cur_mean = mean(multi_data, 2);
            cur_std = std(multi_data, 0, 2);
            multi_data = bsxfun(@minus, multi_data, cur_mean);
            multi_data = bsxfun(@rdivide, multi_data, cur_std);
        elseif eI.inputL1==2,  % at each time frame, freq sum to 1
            l1norm = sum(multi_data,1)+eps;
            multi_data = bsxfun(@rdivide, multi_data, l1norm);
        end
        
        %% output normalize
        if eI.outputL1 == 1
            if eI.cleanonly == 1,
                l1norm = sum(abs(spectrum.signal),1)+eps;
                spectrum.signal = bsxfun(@rdivide, spectrum.signal, l1norm);
            else
                l1norm = sum([abs(spectrum.noise); abs(spectrum.signal)],1)+eps;
                spectrum.noise = bsxfun(@rdivide, spectrum.noise, l1norm);
                spectrum.signal = bsxfun(@rdivide, spectrum.signal, l1norm);
            end
        end
        
        if eI.cleanonly==1,
            clean_data = spectrum.signal;
        else
            clean_data=[spectrum.noise; spectrum.signal];
        end
        
        %% zero pad
        if eI.num_contextwin > 1
            % winSize must be odd for padding to work
            if mod(eI.num_contextwin,2) ~= 1
                fprintf(1,'error! winSize must be odd!');
                return
            end;
            % pad with repeated frames on both sides so im2col data
            % aligns with output data
            nP = (eI.num_contextwin-1)/2;
            multi_data = [repmat(multi_data(:,1),1,nP), multi_data, ...
                repmat(multi_data(:,end),1,nP)];
        end;
        
        %% im2col puts winSize frames in each column
        nframes = size(multi_data, 2) - eI.num_contextwin + 1;
        ix = repmat((1:eI.num_contextwin*nfeat)', 1, nframes) + ...
            repmat(0:nframes-1, eI.num_contextwin*nfeat, 1);
        multi_data_slid = multi_data(ix);
        % concatenate noise estimate to each input
        if mode == 1, % Testing
            c = find(unique_lengths == T);
            if isempty(c)
                % add new unique length if necessary
                data_ag = [data_ag, multi_data_slid(:)];
                unique_lengths = [unique_lengths, T];
            else
                data_ag{c} = [data_ag{c}, multi_data_slid(:)];
            end;
        elseif mode == 2, % Error analysis.
            c = find(unique_lengths == T);
            if isempty(c)
                % add new unique length if necessary
                data_ag = [data_ag, multi_data_slid(:)];
                target_ag = [target_ag, clean_data(:)];
                mixture_ag = [mixture_ag, mixture_spectrum(:)];
                unique_lengths = [unique_lengths, T];
            else
                data_ag{c} = [data_ag{c}, multi_data_slid(:)];
                target_ag{c} = [target_ag{c}, clean_data(:) ];
                mixture_ag{c} = [mixture_ag{c}, mixture_spectrum(:) ];
            end;
        elseif mode == 3 % formulate one data per cell
            item =item+1;
            % feadim x nframes
            data_ag{item} = multi_data_slid;
            target_ag{item} = clean_data;
            mixture_ag{item} = mixture_spectrum;
        else % training
            %% put it in the correct cell area.
            last = 0;
            while T > 0
                % assumes length in ascending order.
                % Finds longest length shorter than utterance
                c = find(eI.seqLen <= T, 1,'last');
                
                binLen = eI.seqLen(c);
                ix = last+1:last+binLen;
                last = ix(end);
                assert(~isempty(c),'could not find length bin for %d',T);
                % copy data for this chunk
                data_ag{c}(:,seqLenPositions(c)) = ...
                    reshape(multi_data_slid(:,ix),[],1);
                target_ag{c}(:,seqLenPositions(c)) = ...
                    reshape(clean_data(:,ix),[],1);
                mixture_ag{c}(:,seqLenPositions(c)) = ...
                    reshape(mixture_spectrum(:,ix),[],1);
                
                seqLenPositions(c) = seqLenPositions(c)+1;
                % trim for next iteration
                T = T-binLen;
            end;
        end;
    end;
end

theoutputs = {data_ag, target_ag, mixture_ag};
varargout = theoutputs(1:nargout);

return;

%% Unit test
profile on %#ok<UNRCH>
eI.DataPath=['.',filesep, 'Wavfile', filesep];

eI.RealorComplex = 0; % 0- real, 1- complex
eI.winsize = 512;
eI.hop =eI.winsize/2;
eI.scf=1;
eI.featDim = 512;
eI.num_contextwin = 3;
eI.inputDim = eI.featDim * eI.num_contextwin;
train_files= dir( [eI.DataPath, 'train',filesep,'*wav']);
train_files = train_files(1:3);
eI.seqLen = [1 50 100];
eI.inputL1=0;
eI.outputL1=0;
eI.circular_step=100000;
[data_ag, target_ag, mixture_ag] = ...
    formulate_data_time_domain(train_files, eI, 0)
profile viewer
profile off
