function out = get_pos_interpolant(t,pos)
%Intakes directions over time and output interpolant functions for each
%cartesian coordinate
%   Sina Hafezi, Oct 2022, initial version
%   -- INPUTS --
%   t           time samples (sec) [N x 1]
%   pos         position (doa or cartesian) over time wrt. head coordinate system. doa: [N 3] (time-sec,azi-deg,ele-deg) or cartesian: [N 4] (time-sec,x,y,z)  
%   -- OUTPUT --
%   out.x       interpolant function for x cartesian coordinate
%   out.y       interpolant function for y cartesian coordinate
%   out.z       interpolant function for z cartesian coordinate
%   out.t       original time samples used

if size(pos,2)==2
    % doa
    pos = deg2rad(pos);
    [x,y,z]=sph2cart(pos(:,1),pos(:,2),ones(size(pos,1),1));
    pos = [x,y,z];
end

out.x = griddedInterpolant(t,pos(:,1),"linear","nearest");
out.y = griddedInterpolant(t,pos(:,2),"linear","nearest");
out.z = griddedInterpolant(t,pos(:,3),"linear","nearest");
out.t = t;
end