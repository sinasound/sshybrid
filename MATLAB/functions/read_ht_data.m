function out = read_ht_data(file_path,mode)
% Main read function for head-tracking data
% implemented by Sina Hafezi, Sep 2021.
%   -- INPUTS --
%   file_path           (string) path to head-tracking file (.json or .wav' or '.csv')
%   mode                (string) 'ELOBES_WAV' or 'FRL_JSON' or 'SPEAR_CSV'
%   -- OUTPUT --
%   out                 (struct) structure where each (.w .x .y .z) field is an interpolated quaternion to be used for any artbitrary time(s)
%   out.w               (griddedInterpolant) w quaternion
%   out.x               (griddedInterpolant) x quaternion
%   out.y               (griddedInterpolant) y quaternion
%   out.z               (griddedInterpolant) z quaternion 
%   out.fs              (double) sample rate of head-track data
%   out.t               (vector) time vector of head-track data


%   -- USAGE --
%   out = read_ht_data(path);
%   my_times = [0:10]; % in seconds
%   my_q=[out.w(my_times(:)) out.x(my_times(:)) out.y(my_times(:)) out.z(my_times(:))];


% default assumption on 'mode' if it is unavailable (assign based on file format)
if nargin<2
    if strcmpi(file_path(end-2:end),'wav')
        mode = 'ELOBES_WAV';
    elseif strcmpi(file_path(end-3:end),'json')
        mode = 'FRL_JSON';
    elseif strcmpi(file_path(end-2:end),'csv')
        mode = 'SPEAR_CSV';
    else
        error('file format not found!');
    end
end

 % initial assumption
fs = 20; % frames per second
wearer_ID=2; % participant ID wearing AR glasses
switch mode
    case 'ELOBES_WAV'
        nom = pi; % un-normalized range limit (radian)
        %nom = 180; % deg rpy
        [rpy,fs] = audioread(file_path);
        if size(rpy,2)~=3
            error('Input file does not contain 3 channels - Should be normalized roll,pitch and yaw')
        end
        rpy = rpy*nom; %  scale to rad or deg
        t = ([1:size(rpy,1)].'-1)/fs;
        qr_in = v_roteu2qr('xyz',rpy.'); % [4 nSamples]: [w x y z]
    case 'SPEAR_CSV'
        d = readmatrix(file_path); %[index time qx qy qz qw]
        fs = 1/diff(d(1:2,2));
        t = d(:,2);
        qr_in = [d(:,6) d(:,3) d(:,4) d(:,5)].'; %[qw qx qy qz]
        
    case 'FRL_JSON'
        % FRL json files contains location and position data
        % location: xyz cartesian coordinates (based on AR-vision conventional xyz definition)
        % position (orientation): wxyz quaternions
        fid=fopen(file_path);
        raw=fread(fid,inf);
        str=char(raw');
        fclose(fid);
        vals = jsondecode(str);
        nframes=length(vals);
        t=([1:nframes].'-1)/fs;
        qr_in = zeros(4,nframes); % [4 nSamples]: [w x y z]
        temp=[0 cosd(45) cosd(45)]; % rotation vector for upside down (Their XY-plane or Our YZ-plane)
        for frame=1:nframes
            val=vals(frame);    
            ID=[val.Participants.Participant_ID];
            obs_i=find(ID==wearer_ID);
            %nSrc=length(ID);
            qx=val.Participants(obs_i).Quaternion_Z;
            qy=val.Participants(obs_i).Quaternion_X;
            qz=val.Participants(obs_i).Quaternion_Y;
            qw=val.Participants(obs_i).Quaternion_W;
            if val.Participants(obs_i).isUpSideDown
                q=quaternion(qw,qx,qy,qz);
                q2=quaternion(axang2quat([temp,pi]));
                q=q.*q2;
                ttt=compact(q);
                qw=ttt(1);
                qx=ttt(2);
                qy=ttt(3);
                qz=ttt(4);
            end    
            qr_in(:,frame) = [qw qx qy qz].';
        end
        
    otherwise
        error('mode not found!');
end

out.w = griddedInterpolant(t,qr_in(1,:),'linear');
out.x = griddedInterpolant(t,qr_in(2,:),'linear');
out.y = griddedInterpolant(t,qr_in(3,:),'linear');
out.z = griddedInterpolant(t,qr_in(4,:),'linear');
out.fs=fs;
out.t=t;
end

