function  varargout = formulate_data_time_domain(train_files, eI)
% (TODO) add
% eI.winSize: Size of window
% eI.seqLen: unique lengths (in ascending order)
%             files are chopped by these lengths. (ex: [1, 10, 100])
% eI.targetWhiten: Specify the path to the whitening data.
% data_ag: noisy data. cell array of different training lengths.
% target_ag: clean data. cell array of different training lengths.

%% Set up. During testing, dont know the lengths so cant pre-allocate
seqLenSizes = zeros(1,length(eI.seqLen));

for ifile = 1:numel(train_files)
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
    
    train1 = train1./sqrt(sum(train1.^2));
    train2 = train2./sqrt(sum(train2.^2));
    
    shift_size = min(eI.circular_step, numel(train2)-1);
    fprintf('Allocating training data: %.2f\n', ifile/numel(train_files));
    for ioffset = 1:shift_size:numel(train2)-shift_size % circle shift
        train2_shift = [train2(ioffset: end); train2(1: ioffset-1)];
        fprintf('\tShift: %d\n', ioffset)
        dmix = train1 + train2_shift;
        
        % input feature calculate
        DATA = dmix';
        
        [T, innfeat] = size(DATA');
        nfeat = innfeat;
        
        remainder = T;
        for i = length(eI.seqLen):-1:1
            num = floor(remainder/eI.seqLen(i));
            remainder = mod(remainder,eI.seqLen(i));
            seqLenSizes(i) = seqLenSizes(i)+num;
        end
    end
end

data_ag = cell(1,length(eI.seqLen));
target_ag = cell(1,length(eI.seqLen));
mixture_ag = cell(1,length(eI.seqLen));
% !!! TODO: make this handle complex
for i = length(eI.seqLen):-1:1
    data_ag{i} = zeros(eI.inputDim*eI.seqLen(i), seqLenSizes(i));
    mixture_ag{i} = zeros(nfeat*eI.seqLen(i), seqLenSizes(i));
    if eI.cleanonly == 1,
        target_ag{i} = zeros(nfeat*eI.seqLen(i), seqLenSizes(i));
    else
        target_ag{i} = zeros(2*nfeat*eI.seqLen(i), seqLenSizes(i));
    end
end

seqLenPositions = ones(1,length(eI.seqLen));
for ifile = 1:numel(train_files)
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
    for ioffset = 1:shift_step:numel(train2)-shift_step % circle shift
        train2_shift = [train2(ioffset:end); train2(1:ioffset-1)];
        fprintf('\tShift: %d\n', ioffset)
        dmix = train1 + train2_shift;
        signal = train1';
        noise = train2_shift';
        
        % Size should be 1 x len(multi_data)
        multi_data = dmix';
        mixture_spectrum = multi_data;
        [~,T] = size(multi_data);
        
        %% input normalize
        if eI.inputL1 == 1 % DATA (NUMCOFS x nSamp)
            % apply CMVN to targets (normalize such that the freq
            % bean equal to zero mean var 1)
            cur_mean = mean(multi_data, 2);
            cur_std = std(multi_data, 0, 2);
            multi_data = multi_data - cur_mean;
            multi_data = multi_data / cur_std;
        elseif eI.inputL1 == 2  % at each time frame, freq sum to 1
            l1norm = sum(multi_data,1) + eps;
            multi_data = multi_data / l1norm;
        end
        
        %% output normalize
        if eI.outputL1 == 1
            fprintf(1,'Warning: L1 reg on output not supported.');
        end
        
        if eI.cleanonly == 1
            clean_data = signal;
        else
            clean_data = [noise; signal];
        end
        
        %% zero pad
        if eI.num_contextwin > 1
            % winSize must be odd for padding to work
            if mod(eI.num_contextwin,2) ~= 1
                fprintf(1,'error! winSize must be odd!');
                return
            end
            % pad with repeated frames on both sides so im2col data
            % aligns with output data
            nP = (eI.num_contextwin-1)/2;
            multi_data = [repmat(multi_data(:,1),1,nP), multi_data, ...
                repmat(multi_data(:,end),1,nP)];
        end
        
        %% im2col puts winSize frames in each column
        len = length(multi_data) - eI.num_contextwin + 1;
        ix = repmat((1:eI.num_contextwin)', 1, len) + ...
            repmat(0:len-1, eI.num_contextwin, 1);
        multi_data_slid = multi_data(ix);
        % concatenate noise estimate to each input
        %% put it in the correct cell area.
        last = 0;
        while T > 0
            % assumes length in ascending order.
            % Finds longest length shorter than utterance
            c = find(eI.seqLen <= T, 1,'last');
            assert(~isempty(c),'could not find length bin for %d',T);
            binLen = eI.seqLen(c);
            ix = last+1:last+binLen;
            last = ix(end);
            % copy data for this chunk
            data_ag{c}(:,seqLenPositions(c))=reshape(...
                multi_data_slid(:,ix),[],1);
            target_ag{c}(:,seqLenPositions(c))=reshape(...
                clean_data(:,ix),[],1);
            mixture_ag{c}(:,seqLenPositions(c))=reshape(...
                mixture_spectrum(:,ix),[],1);
            
            seqLenPositions(c) = seqLenPositions(c)+1;
            % trim for next iteration
            T = T-binLen;
        end
    end
end

theoutputs = {data_ag, target_ag, mixture_ag};
varargout = theoutputs(1:nargout);

return

%% Unit test
% (TODO) add
% before edit: 17.725 s
% 5.140 s
% 0.771 s
eI.DataPath = ['.', filesep, 'Wavfile', filesep]; %#ok<UNRCH>
profile on

eI.scf = 1;
eI.featDim = 1;
eI.num_contextwin = 1;
eI.inputDim = eI.featDim * eI.num_contextwin;
eI.cleanonly = 0;
train_files = dir([eI.DataPath, 'train', filesep, '*wav']);
train_files = train_files(1:3);
eI.seqLen = [1 50 100];
eI.inputL1 = 0;
eI.outputL1 = 0;
eI.circular_step = 100000;
[data_ag, target_ag, mixture_ag] = ...
    formulate_data_time_domain(train_files, eI);
profile viewer

