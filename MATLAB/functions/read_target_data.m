function out = read_target_data(file_path,mode,target_id)
% Main read function for target location data
% implemented by Sina Hafezi, Sep 2021.
%   -- INPUTS --
%   file_path           (string) path to location data file (.json or .csv')
%   mode                (string) 'CART_CSV' or 'FRL_JSON' or 'SPEAR_CSV'
%   target_id           (int)    source if of target in case file contains multi-source info (for FRL mode)
%   -- OUTPUT --
%   out                 (struct) structure where each (.x .y .z) field is an interpolated cartesian coordinate to be used for any artbitrary time(s)
%   out.x               (griddedInterpolant) x 
%   out.y               (griddedInterpolant) y 
%   out.z               (griddedInterpolant) z  
%   out.fs              (double) sample rate of data
%   out.t               (vector) original time vector of data


%   -- USAGE --
%   out = read_target_data(path);
%   out = read_target_data(path,2);  
%   out = read_target_data(path,'FRL_JSON',2);
%   out = read_target_data(path,'CART_CSV');
%   my_times = [0:10]; % in seconds
%   my_cart=[out.x(my_times(:)) out.y(my_times(:)) out.z(my_times(:))];
isModeGiven=0;
if nargin>1
    if ~isnumeric(mode)
        isModeGiven=1;
        if (nargin==2) target_id = 1; end
    else
        target_id = mode;
    end
else
    target_id = 1;
end


% default assumption on 'mode' if it is unavailable (assign based on file format)
if ~isModeGiven
    if strcmpi(file_path(end-2:end),'csv')
        mode = 'CART_CSV';
    elseif strcmpi(file_path(end-3:end),'json')
        mode = 'FRL_JSON';
    else
        error('file format not found!');
    end
end
fs = 20; % frames per second
wearer_ID=2; % participant ID wearing AR glasses
switch mode
    case 'CART_CSV'
        
        target_pos = readtable(file_path,'ReadVariableNames',0);
        if size(target_pos,2)~=4
            error('csv file %s does not have 4 columns',...
                oracle_data.target_position_file)
        end
        target_pos.Properties.VariableNames = {'t','x','y','z'};
        fs=1/abs(target_pos.t(2)-target_pos.t(1));
    case 'SPEAR_CSV'
        % The only available metadata in SPEAR
        [fPath, fName, ~] = fileparts(file_path);
        prefix = fName(1:strfind(fName,'_M')+3);
        file = fullfile(fPath,sprintf('%s_ID%d.csv',prefix,target_id));
        num = readmatrix(file); %[index time azimuth elevation] in deg
        [target_pos.x target_pos.y target_pos.z] = sph2cart(deg2rad(num(:,3)),deg2rad(num(:,4)),ones(size(num,1),1));
        target_pos.t = num(:,2);
        fs = 1./diff(target_pos.t(1:2));
        
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
        target_pos.t=([1:nframes].'-1)/fs;
        target_pos.x=[];
        target_pos.y=[];
        target_pos.z=[];
        
        for frame=1:nframes
            val=vals(frame);    
            target_ind=find([val.Participants.Participant_ID]==target_id);
            wearer_ind=find([val.Participants.Participant_ID]==wearer_ID);
            x=val.Participants(target_ind).Position_Z;
            y=val.Participants(target_ind).Position_X;
            z=val.Participants(target_ind).Position_Y;
            
            xo=val.Participants(wearer_ind).Position_Z;
            yo=val.Participants(wearer_ind).Position_X;
            zo=val.Participants(wearer_ind).Position_Y;
            
            target_pos.x(frame)=x-xo;
            target_pos.y(frame)=y-yo;
            target_pos.z(frame)=z-zo;   
        end
        
    otherwise
        error('mode not found!');
end

out.x = griddedInterpolant(target_pos.t,target_pos.x,'linear');
out.y = griddedInterpolant(target_pos.t,target_pos.y,'linear');
out.z = griddedInterpolant(target_pos.t,target_pos.z,'linear');
out.fs= fs;
out.t = target_pos.t;
end

