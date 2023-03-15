function [z,z_iso,z_hyb,selected_models] = fcn_mvdr_sshybrid(y,fs,target_pos,settings)
% Subspace-Hybrid (SSH) MVDR beamformer
% version history, authors, updates:
% Sina Hafezi       2022 Sep    Initial version
% Sina Hafezi       2022 Oct    entirely re-structured. intakes data rather file

% -- INPUTS --  
%   y                   array time-domain signal [nSample  nChan]
%   fs                  sample rate of y (Hz)
%   target_pos          target position (doa or cartesian) over time wrt. head coordinate system. doa: [N 3] (time-sec,azi-deg,ele-deg) or cartesian: [N 4] (time-sec,x,y,z)  
%   settings            beamformer settings in a structure
%       .dict           dictionary
%           .steer_dirs sampled steering directions [nSteer x 2] (azi, ele) deg
%           .weights    beamformer weights [nChan nFreq nModel nSteer] for a reference mic
%       .Tau            time constant (sec) for exponential moving average used in PCA temporal smoothing   
% -- OUTPUTS --
%   z                   enhanced signal (SS-Hybrid) in time domain [nSample  1]
%   z_iso               output of Iso-MVDR beamformer in time domain [nSample 1]
%   z_hyb               output of Hybrid-MVDR beamformer in time domain (includes musical noise) [nSample 1]
%   selected_models     index of selected models in the STFT domain [nFreq nFrame]

fprintf('SS-Hybrid\n');
% STFT
[Y,t,f,pm] = stft_quick(y,fs);
[nFreq nChan nFrame]=size(Y);
% target interpolant function
pos_interp = get_pos_interpolant(target_pos(:,1),target_pos(:,2:end));
[target_az, target_el,~] = cart2sph(pos_interp.x(t(:)),pos_interp.y(t(:)),pos_interp.z(t(:)));
doas = rad2deg([target_az(:), target_el(:)]);
% PCA smoothing settings
ax=exp(-(t(2)-t(1))/settings.Tau);
%% main processing
% preallocate variables to be populated
w = zeros(nFreq,nChan,nFrame);  % beamformer weights for Hybrid MVDR

Z_iso = zeros(nFreq,1,nFrame); % output of Iso MVDR
Z_hyb = zeros(nFreq,1,nFrame); % output of Hybrid MVDR
Z_sshyb = zeros(nFreq,1,nFrame); % output of SS-Hybrid MVDR
selected_models = zeros(nFreq,nFrame);   % stores the index of selected model in the STFT domain

R_past = []; % last inter-method covariance matrix (from last frame)
nModel = size(settings.dict.weights,3); % total number of models 
reverseStr = '';

for iframe = 1:nFrame
    msg = sprintf('- SSHybrid Beamforming %2.2f s of %2.2f', pm.t(iframe), pm.t(end));
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));

    % nearest available look direction in the dictionary
    target_az = deg2rad(doas(iframe,1));
    target_el = deg2rad(doas(iframe,2));    

    [~,di] = min(get_angle_between(rad2deg([target_az,target_el]),settings.dict.steer_dirs));
    
    % Hybrid-MVDR
    Zt = repmat(squeeze(Y(:,:,iframe)),1,1,nModel) .* squeeze(settings.dict.weights(:,:,:,di));  %[nFreq nChan nModel]
    Zt = squeeze(sum(Zt,2)); % output of all MVDR beamformings [nFreq nModel]
    Z_iso(:,1,iframe) = Zt(:,1); % get a copy of Iso-MVDR

    % Model selection
    % Zt [nFreq x nModel] output of all models
    [~,best_model] = min(abs(Zt),[],2);
    selected_models(:,iframe) = best_model;
    for ifreq = 1:nFreq
        Z_hyb(ifreq,1,iframe)=Zt(ifreq,best_model(ifreq));
        w(ifreq,:,iframe) = settings.dict.weights(ifreq,:,best_model(ifreq),di);
    end

   
    % Wideband inter-method PCA
    Zf = [Z_hyb(:,1,iframe) Z_iso(:,1,iframe)].'; % [2 nFreq]
    R = Zf * Zf'; % [2 x 2]  inter-method covaraince matrix
    
    % R smoothing (EMA)
    if isempty(R_past)  R_past = R; end
    R = ax * R_past + (1-ax) * R;
    R_past = R;

    % EVD
    [U,S] = eig(R); 
    U = U(:,end); % [2 x 1] signal eigenvector

    % formulation 1
    %{
    Zf = U * U' * Zf; % projection & reconstruction
    Z_sshyb(:,1,iframe) = Zf(1,:); 
    %}

    % formulation 2 (equivalent to formulation 1)
    %%{
    c_hyb = U(1)*conj(U(1));
    c_iso = U(1)*conj(U(2));
    Z_sshyb(:,1,iframe) = c_hyb*Z_hyb(:,1,iframe) + c_iso*Z_iso(:,1,iframe);
    %}

end
Z = Z_sshyb;
z = stft_v2('inv',Z_sshyb,pm); % istft
z_hyb = stft_v2('inv',Z_hyb,pm); % istft
z_iso = stft_v2('inv',Z_iso,pm); % istft
fprintf('- Done\n');