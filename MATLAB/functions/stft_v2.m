function[out,out_tail,pm] = stft(modestr,in,pm,inc_or_in_tail,n_fft,fs)
%function[~,out_tail,pm] = stft('init',n_chans,win,inc,n_fft,fs)
%function[X,out_tail,pm] = stft('fwd',in,w_or_pm,inc_or_in_tail,n_fft,fs)
%function[x,out_tail,pm] = stft('inv',in,w_or_pm,inc_or_in_tail,n_fft,fs)

%STFT takes 2D multichannel signal matrix and returns 3D short time fourier
%transform matrix at n_freq=fix(1+n_fft/2) frequencies
%
%[X,pm,t,f] = stft(x,w,inc,n_fft,fs)
%
% Inputs
%       x  input signal [n_samples n_channels]
%      fs  sample rate of input signal
%       w  frame length or window of frame length
%     inc  frame increment in samples
%
% Outputs
%       X  STFT matrix [n_freq, n_channels, n_frames]
%       t  centre time of each frame
%       f  frequency frequency of each bin
%
%

% for block based processing must intialise pm structure first either by
% call to stft where x=nChans or my manually specifying the parameters
% structure and populating the tail

%[~,out_tail,pm] = stft(nChans,win,inc,n_fft,fs,in_tail)

%subsequent calls can then just be
%[X,out_tail] = stft(x,in_tail,pm)

%or if the timestamps for each output frame are required
%[X,out_tail,pm] = stft(...)

% w_or_pm:
%  specify the window to be used for the fwd and inverse tranform
%  include both here so that the output struct can be used directly by
%  istft without any futher initialisation


% n_fft:                   []  - fft will be the same length as win (no padding)
%                         [N]  - the fft length to use (post padding if required)
%    [fft_zp_pre fft_zp_post]  - specify directly the padding

% %% dummy data for testing
% x = bsxfun(@times,[repmat([1:20]',1,4)],[1 10 100 1000])
% fs = 1;
% w = ones(8,1);
% inc = 4;


switch modestr
    case 'init'
        n_chans = in;
        if ~isscalar(n_chans), error('init stft requires a scalar specifying number of input channels'),end
        win = pm;
        pm = internal_init(win,inc_or_in_tail,n_fft,fs);
        out_tail = [];%zeros(pm.n_win_fwd-pm.inc,n_chans); %accept that we lose some information from the start for the sake of simplicity
        out = [];
    case 'fwd'
        [n_samples,n_chans,dim3] = size(in);
        if size(dim3)~=1, error('fwd stft requires 2D matrix of column aligned time domain signals'),end
        
        % deal with case that stft is being called as a one-off (not block based)    
        if ~isstruct(pm)
            % assume we don't have an initialised pm struct and so
            % processing is to be done in one go (not in blocks)
            win = pm;
            pm = internal_init(win,inc_or_in_tail,n_fft,fs);
            pm.proc_in_blocks = 0;
            
            % to make sure no samples get lost or attenuated in the windowing
            % need to pre and post pad
            % pre-pad is handled in main code using in_tail
            
            pm.pre_pad_len = pm.n_win_fwd-pm.inc;                                % last inc samples of first frame will be non-zero
            inc_or_in_tail = zeros(pm.len_tail,n_chans);
            
            % post-pad handled here
            % n.b. need to fill the final block and then add enough samples so
            % that all samples are in the same number of windows
            pm.post_pad_len = pm.inc-mod(n_samples,pm.inc) + pm.pre_pad_len;    % fill final block and then pad
            in = [in; zeros(pm.post_pad_len,n_chans)];
        end
        
        % prepend signals left over from previous block
        in = [inc_or_in_tail; in]; %TODO: investigate how bad is this for efficiency?
        
        % determine the indices required to enframe the data
        len_x = size(in,1);
        fr_st = 1:pm.inc:len_x-(pm.n_win_fwd-1);
        n_frames = length(fr_st);
        if n_frames==0, error('Input signal is too short'),end
        fr_idc = bsxfun(@plus,fr_st,[0:pm.n_win_fwd-1]');
        
        % deal with the tail before actually enframing and computing stft
        out_tail = in(fr_st(end)+pm.inc:end,:); %tail starts with first incomplete frame
        
        % error check
        if ~pm.proc_in_blocks && ~all(out_tail(:)==0)
            error('Something went with the padding - tail contains non-zero samples')
        end
        
        % only populate output struct if required
        if nargout==3
            if pm.proc_in_blocks && isfield(pm,'fr_st')
                %find the start index of this block's in_tail relative to full
                %signal - relies on the previous call having populated these values
                idc_offset = pm.fr_st(end)+pm.inc-1;
            else
                idc_offset = 0;
            end
            pm.fr_st = idc_offset + fr_st;
            pm.t = (pm.fr_st - 1 - pm.pre_pad_len + (pm.n_win_fwd+1)/2) ./ pm.fs;
        end
        
        
        % --- real processing happens here ---
        
        % enframe signal
        y = permute(reshape(in(fr_idc,:),[pm.n_win_fwd,n_frames,n_chans]),[1,3,2]);
        
        % apply window
        y = y .* repmat(pm.win_fwd(:),[1, n_chans, n_frames]);
        
        if pm.fft_pre_pad || pm.fft_post_pad
            y = [zeros(pm.fft_pre_pad,n_chans,n_frames); y; zeros(pm.fft_post_pad,n_chans,n_frames)];
        end
        
        % do fft
        out = rfft(y,pm.n_fft,1);
        
        % --- done ---
        
        
    case 'inv'
        if ~isstruct(pm)
            error('stft(''inv'',...) requires an initialised parameter struct')
        end
        [n_freqs,n_chans,n_frames] = size(in);
        
        % do ifft
        %x = irfft(X);
        out_frames = ifft(in, pm.n_fft, 'symmetric'); %this is ~5 times faster that irfft
        
        
        % apply window
        out_frames = out_frames .* repmat(pm.win_inv(:),[1, n_chans, n_frames]);
        
        %% overlap add frames
        % using sparse and clever indexing
        %ii = repmat(pm.fr_idc,1,n_chans);
        %ij = repmat(1:n_chans,numel(pm.fr_idc),1);
        %x = permute(x, [1 3 2]);
        %out = full(sparse(ii(:),ij(:),x(:),pm.len_x_pad,n_chans));
        
        % using old school for loop is ~10 times faster
        fr_st = 1 + [0:n_frames-1]*pm.inc;          %frame start indices
        fr_idc = bsxfun(@plus,fr_st,[0:pm.n_win_inv-1]');    %frame indices fully expanded
        n_samples = fr_idc(end,end);
        out = zeros(n_samples,n_chans);
        for n = 1:n_frames
            out(fr_idc(:,n),:) = out(fr_idc(:,n),:) + out_frames(:,:,n);
        end
        
        %deal with tail / padding
        if pm.proc_in_blocks
            %add the previous tail to the leading samples
            if (nargin > 3) && ~isempty(inc_or_in_tail)
                len_in_tail = size(inc_or_in_tail,1);
                out(1:len_in_tail,:) = out(1:len_in_tail,:) + inc_or_in_tail;
            end
            %retain the unfilled samples for the next block - starting from ninc
            %samples after last frame start
            out_tail = out(fr_st(end)+pm.inc:end,:);
            out(fr_st(end)+pm.inc:end,:) = [];
        else
            out(end-pm.post_pad_len+(1:pm.post_pad_len),:) = [];
            out(1:pm.pre_pad_len,:) = [];
        end
        
    otherwise
        error('Unknown mode %s specified for modestr',modestr)
end









function[pm] = internal_init(win,inc,n_fft,fs)
% deals with windows, fft sizes, etc
% TODO: Add assumptions for convenience - for now everything must be fully
% specified

% inc must be a scalar integer
if ~isscalar(inc) || rem(inc,1)~=0
    error('inc must be a scalar integer')
end
pm.inc = inc;

% number of samples in fwd and inv windows
pm.n_win_fwd = length(win{1});
pm.n_win_inv = length(win{2});

% windows themselves copied directly to struct
pm.win_fwd = win{1};
pm.win_inv = win{2};

% interpret n_fft specification
if isempty(n_fft)
    pm.n_fft = pm.n_win_fwd;
    pm.fft_pre_pad = 0;
    pm.fft_post_pad = 0;
elseif isscalar(n_fft)
    pm.n_fft = n_fft;
    pm.fft_pre_pad = 0;
    pm.fft_post_pad = n_fft-pm.n_win_fwd;
else
    n_fft = n_fft(:);
    if size(n_fft)~=[2 1]
        error('dimensions of n_fft cannot be interpreted')
    end
    pm.fft_pre_pad = n_fft(1);
    pm.fft_post_pad = n_fft(2);
    pm.n_fft = pm.fft_pre_pad + pm.n_win_fwd + pm.fft_post_pad;
end

% check that the window for inverse transformed frames is the right size
if pm.n_win_inv~=pm.n_fft
    error('window for inverse transform must match the selected FFT size')
end

pm.len_tail = pm.n_win_fwd - pm.inc; %number of sample which will be carried over from each frame


if isempty(fs)
    pm.fs = 1;
else
    pm.fs = fs;
end
pm.f = (0:fix(pm.n_fft/2)).' * pm.fs/pm.n_fft; % frequency scale


%defaults for block-based processing, must override in main function if required
pm.proc_in_blocks = 1;
pm.pre_pad_len = 0;
pm.post_pad_len = 0;

