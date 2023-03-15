% Author: Sina Hafezi
% Enhancement processing script for SS-Hybrid beamformer
% Paper title: Subspace Hybrid Beamforming for head-worn microphone arrays
% Conference: ICASSP 2023 

% Version history: date/author/updates:
% 2023 Mar - Sina Hafezi - first matlab script to run SS-Hybrid beamforming enhancement

clear
close all;

setup;

% examle data
test_case=1; % test case number
dataPath = fullfile('..','example data',num2str(test_case)); % root path to example data
savePath = 'outputs/';mkdir(savePath); % where to optionally save the output audio
ToSave=0; % whether to save output audio in savePath


% processing setting
fs = 10e3; % sample rate
refChan = 2; % array reference channel
settings.Tau = 80e-3; % 80 ms for PCA smoothing
settings.dict = load("﻿sshybrid_dictionary.mat"); % load dictionary

% load array audio
[y,fs0]=audioread(fullfile(dataPath,'array.wav'));
y=resample(y,fs,fs0);
% load target DOA (wrt. head coordinate system)
tinfo=readcell(fullfile(dataPath,'info.txt'));
target_id=tinfo{find(contains(tinfo(:,1),'Target ID')),2};
target = read_participant_relative_position(fullfile(dataPath,'ht.json'),target_id);
target_pos = zeros(length(target.t),3); %[time(sec) azimuth(deg) elevation(deg]
target_pos(:,1)=target.t(:);
[target_pos(:,2) target_pos(:,3) ~]=cart2sph(target.x(target.t(:)),target.y(target.t(:)),target.z(target.t(:)));
target_pos(:,2:3) = wrapTo180(rad2deg(target_pos(:,2:3)));


% SS-Hybrid beamforming
[z_sshyb,z_iso,z_hyb] = fcn_mvdr_sshybrid(y,fs,target_pos,settings);
z_pass = y(:,refChan); % passthrough signal

% To listen
gain=20;
% sound(gain*z_pass,fs);
% sound(gain*z_iso,fs);
% sound(gain*z_hyb,fs);
% sound(gain*z_sshyb,fs);

% To save outupt audio
if ToSave
    audiowrite(fullfile(savePath,sprintf('test_%d_%s.wav',test_case,'pass')),gain*z_pass,fs);
    audiowrite(fullfile(savePath,sprintf('test_%d_%s.wav',test_case,'iso')),gain*z_iso,fs);
    audiowrite(fullfile(savePath,sprintf('test_%d_%s.wav',test_case,'hyb')),gain*z_hyb,fs);
    audiowrite(fullfile(savePath,sprintf('test_%d_%s.wav',test_case,'sshyb')),gain*z_sshyb,fs);
end


%% STFT plots
CL=[-100 -50];
figure;tiledlayout(4,1,'TileSpacing','tight','Padding','tight');
nexttile;fcn_plot_spec(z_pass,fs);title('Passthrough');clim(CL);
nexttile;fcn_plot_spec(z_iso,fs);title('Iso');clim(CL);
nexttile;fcn_plot_spec(z_hyb,fs);title('Hybrid');clim(CL);
nexttile;fcn_plot_spec(z_sshyb,fs);title('SS-Hybrid');clim(CL);
if ToSave exportgraphics(gcf,fullfile(savePath,sprintf('test_%d.png',test_case))); end

