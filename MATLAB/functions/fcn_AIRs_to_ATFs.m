function [H,f] = fcn_AIRs_to_ATFs(ir,fs_ir,fs_out)
% Author: Sina Hafezi
% DESCRIPTION: converts array's Acoustic Impulse Rsponses (AIRs) to Acoustic Transfer Functions (ATFs)
%   --- INPUTS ---
%   ir              impulse response [nSample x nChan x nDir] 
%   fs_ir           original sample rate (Hz)
%   fs_out          sample rate (Hz) for the output ATFs
%   --- OUTPUTS ---
%   H               ATFs    [nFreq x nChan x nDir]
%   f               frequencies [nFreq x 1]

[nSamples,nChan,nDir] = size(ir);
ir = resample(reshape(ir,nSamples,nChan*nDir),fs_out,fs_ir);
nSamples = size(ir,1);
ir = reshape(ir,nSamples,nChan,nDir); 
nfft = size(ir,1); % nFFT
H = rfft(ir,[],1); % ATFs [nFreq x nChan x nDir]
f = (0:((nfft+2)/2 - 1)).' * fs_out/nfft; % freq vector [nFreq x 1] 
end