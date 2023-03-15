function [ir,fs,dirs] = fcn_extract_IRs_EasyCom(file_path)
% Author: Sina Hafezi
% returns the raw impulse responses and their positions
%       ir: impulse responses in [nSamples,nMic,nDir] matrix
%  src_pos: source positions as [nDir x 3] matrix in cartesian
%           co-ordinates
%       fs: sample rate of the data (in Hz)

ir=h5read(file_path,['/' 'IR']); % [nChan x nDir x nSamples]
ir = permute(ir,[3 1 2]); % impulse responses [nSample x nChan x nDir] 
fs=h5read(file_path,['/' 'SamplingFreq_Hz']);
azi=h5read(file_path,['/' 'Phi']); % [nDir x 1] azimuth (radian)
inc=h5read(file_path,['/' 'Theta']); % [nDir x 1] inclination (radian)
dirs = wrapTo180(rad2deg([azi(:) pi/2-inc(:)])); % measured directions (deg)
end