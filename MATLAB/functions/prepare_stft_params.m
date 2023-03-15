function stft_params = prepare_stft_params(fs,window,step)
%UNTITLED Summary of this function goes here
%   fs:         sampling rate [Hz]
%   window:     time window duratio [s]
%   step:       overlap coefficeint (0< <=1)
stft_params.fs=fs;
sig_frame_len=round(stft_params.fs * window);
sig_frame_inc=round(sig_frame_len * step);
stft_params.sig_frame_inc = sig_frame_inc;
stft_params.nfft = sig_frame_len;
w = sqrt(hamming(sig_frame_len,'periodic'));
stft_params.win_anal{1} = w ./ sqrt(sum(w(1:sig_frame_inc:sig_frame_len).^2 * sig_frame_len * sig_frame_inc));
stft_params.win_anal{2} = w .* sqrt(sig_frame_len * sig_frame_inc / sum(w(1:sig_frame_inc:sig_frame_len).^2)); %
end

