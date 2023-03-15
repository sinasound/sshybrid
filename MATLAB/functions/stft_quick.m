function [Y,t,f,pm] = stft_quick(y,fs)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

windowL=16e-3; %16 ms
overlapstep=.5; % 50 percent overlap
stft_params = prepare_stft_params(fs,windowL,overlapstep);
[Y,x_tail_anal,pm] = stft_v2('fwd', y, ...
stft_params.win_anal, ...
stft_params.sig_frame_inc,...
stft_params.nfft,...
stft_params.fs);
%[nFreq,nChan,nFrames] = size(Y);
t=pm.t;
f=pm.f;
end

