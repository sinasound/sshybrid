function out = read_participant_relative_position(ht_file,participant_ID)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

mode='FRL_JSON';
nFrames=1200;
wearer_participant_ID=2;
ht = read_ht_data(ht_file,mode);
loc0=read_target_data(ht_file,mode,wearer_participant_ID);
t=loc0.t;
l0=[loc0.x(t) loc0.y(t) loc0.z(t)];
q=quaternion(ht.w(t),ht.x(t),ht.y(t),ht.z(t));
out.t = t;

loc = read_target_data(ht_file,mode,participant_ID);
l = [loc.x(t) loc.y(t) loc.z(t)];
l = l - l0;

cc=zeros(length(t),3);
for frame = 1:length(t)
    rotated_xyz=rotatepoint(q(frame),[1 0 0;0 1 0;0 0 1]);
    x=dot(rotated_xyz(1,:),l(frame,:));
    y=dot(rotated_xyz(2,:),l(frame,:));
    z=dot(rotated_xyz(3,:),l(frame,:));
    cc(frame,:)=[x y z];
end
out.x = griddedInterpolant(t,cc(:,1),'linear');
out.y = griddedInterpolant(t,cc(:,2),'linear');
out.z = griddedInterpolant(t,cc(:,3),'linear');  
out.t = t;
end

