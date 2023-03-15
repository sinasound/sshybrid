% Author: Sina Hafezi
% ﻿Dictionary generator script for SS-Hybrid beamformer
% Paper title: Subspace Hybrid Beamforming for head-worn microphone arrays
% Conference: ICASSP 2023 

% Version history: date/author/updates:
% ﻿2023 Mar - Sina Hafezi - first matlab script to create and save dictionary for SS-Hybrid beamformer

clear
close all;

setup;

%% Settings
% settings for array/processing
fs = 10e3; % processing sampling rate (Hz)
refChan = 2; % array reference channel
maxCond = 1e5; % maximum condition number for covariance matrix condition number limiting. use 0 to avoid limiting.
% settings for steering direction samples
steer_range_az=[-90 90]; % azimuth range for steering directios of interest
steer_range_el=[-30 30]; % elevation range for steering directios of interest
% setting for anisotropic models
pow_dyn = -[8:8:40]; % power dynamics (dB) of the unimodal anisotropic distributions
az_mode = [0:6:360-6]; % azimuths of the unimodal anisotropic distributions


%% ATFs
% load ATFs
atf_file = '../example data/Device_ATFs.h5';
[ir,fs_ir,dirs] = fcn_extract_IRs_EasyCom(atf_file); % AIRs
[H,f] = fcn_AIRs_to_ATFs(ir,fs_ir,fs); % ATFs
[nFreq,nChan,nDir]=size(H);
az=unique(dirs(:,1)); % unique azimuths (samples on X-axis)
el=unique(dirs(:,2)); % unique elevations (samples on Y-axis)
wq = fcn_quad_weights(dirs(:,1),dirs(:,2)); % quadrature weights [nDir x 1]
nAzi = length(az); % number of unique ATFs azimuth
nEl = length(el); % number of unique ATFs elevation

% plot quadrature weights and ATFs directions
figure;
imagesc(az,el(end:-1:1),reshape(wq,nEl,[]));axis xy;hold on;colorbar;
plot(dirs(:,1),dirs(:,2),'r.');grid on;pbaspect([2 1 1]);
xlim([-180 180]);ylim([-90 90]);
xlabel('Azimtuh [deg]');ylabel('Elevation [deg]');title('Quadrature weights for the ATF directions');
drawnow;

%% Isotropy Construction

% 2D (horizontal) Isotropies - Iso & unimodal Anisotropic
nPow = length(pow_dyn);
nMode = length(az_mode);
az_data = abs(wrapTo180(az(:)- az_mode(:)'))/180; %[nAzi x nMode]
models_combinations = repmat(pow_dyn(:)',nMode,1);
aniso_2D = repmat(az_data,1,1,nPow) .* ...
    permute(repmat(models_combinations,1,1,nAzi),[3 1 2]); % 2D anisotropies [nAzi x nMode x nPow]
iso_2D = zeros(nAzi,1); % [nAzi x 1] 0 dB at every azimuth

% Plot examples of 2D isotropy
figure;tiledlayout(1,2);nexttile;
fcn_plot_polar(az,iso_2D);hold on;
for n=1:nPow fcn_plot_polar(az,squeeze(aniso_2D(:,1,n))); end
rlim([min(pow_dyn) 0]);
title(sprintf('(2D) Horizontal Istoropies\n(Varying Power Dynamic, Fixed Mode = 0˚)'));
nexttile;
for n=1:4 fcn_plot_polar(az,squeeze(aniso_2D(:,1+(n-1)*3,end)));hold on; end
rlim([min(pow_dyn) 0]);
title(sprintf('(2D) Horizontal Istoropies\n(Fixed Power Dynamic = -40dB, Varying Mode)'));
drawnow;

% 3D (full-space) Anisotropies
iso_3D = reshape(repmat(iso_2D(:)',nEl,1),[],1); % [nDir x 1]
aniso_3D = reshape(permute(repmat(aniso_2D,1,1,1,nEl),[2 3 4 1]),nMode,nPow,[]); % [nMode x nPow x nDir]
aniso_3D = reshape(permute(aniso_3D,[3 1 2]),nDir,[]); % [nDir x nMode*nPow]
iso_3D = db2pow(iso_3D) .* wq(:); % quadrature weighting [nDir x 1]
aniso_3D = db2pow(aniso_3D) .* wq(:); % quadrature weighting [nDir x nMode*nPow]

% Plot example of 3D (isotropy)
figure;
imagesc(az,el,reshape(pow2db(aniso_3D(:,end)),nEl,[]));axis xy;colorbar;
xlim([-180 180]);ylim([-90 90]);pbaspect([2 1 1]);caxis([min(pow_dyn) 0]);cl=colorbar;cl.Title.String='[dB]';
xlabel('Azimtuh [deg]');ylabel('Elevation [deg]');title('Example of 3D Isotropy (unimodal anisotropic model)');
drawnow;
%% Dictionary Construction using ATFs and Isotropy models

% index of steering directions within range of interest
steer_dir_indx = find(dirs(:,1)>=steer_range_az(1) & dirs(:,1)<=steer_range_az(2) & ...
    dirs(:,2)>=steer_range_el(1) & dirs(:,2)<=steer_range_el(2)); % [nSteer x 1]
steer_dirs = dirs(steer_dir_indx,:); %(azi,ele) deg [nSteer x 1]
nSteer = size(steer_dirs,1); 

% Plane-wave covariance matrices [nChan x nChan x nFreq x nDir]
RHH = bsxfun(@times,permute(H,[2 4 1 3]),conj(permute(H,[4 2 1 3]))); 
% order of models: 1x Iso + (5x60) Aniso + 1x Identity = 302 models
isotropy_models = [iso_3D,aniso_3D]; % iso must always be the first model
nModel = size(isotropy_models,2)+1;
weights = zeros(nFreq,nChan,nModel,nSteer); % stores MVDR weights conjugate
fprintf('* Constructing the dictionary of weights:\n');
tic;
for fi=1:nFreq % iterate over freq band
    fprintf('freq band index: %d/%d\n',fi,nFreq);
    for mi=1:nModel % iterate over NCM model
        if mi<nModel
            % isotropy model (apply isotropy model and sum over all directions)
            R=squeeze(sum(permute(squeeze(RHH(:,:,fi,:)),[3 1 2]) .* isotropy_models(:,mi))); %[nChan x nChan]
            if maxCond>0 R = fcn_condition_limiting(R,maxCond); end
        else
            % idendity model (last model)
            R = eye(nChan);
        end

        % iterate over steering directions
        for si=1:nSteer
            steer_vec = squeeze(H(fi,:,steer_dir_indx(si)));
            steer_vec = steer_vec(:)./steer_vec(refChan); % steering vector
            weights(fi,:,mi,si) = conj(fcn_mvdr_weights(R,steer_vec)); % MVDR weights
        end 
    end
end
fprintf('Done!\n');
toc

save('﻿sshybrid_dictionary.mat','weights','f','fs','steer_dirs','-v7.3');
