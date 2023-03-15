function out = get_angle_between(ref,in,mode)
% Author: Sina Hafezi
% DESCRIPTION: get angular distance between one/multiple reference direction point(s) and many others 
% --- INPUTS ---
% ref   reference point(s) direction or cartesian position [1x2] (azi_deg,ele_deg) or [1x3] (x,y,z) or [Nx2] or [Nx3]
% in    subject point(s) direction or cartesian position [Nx2] (azi_deg, ele_deg) or [N x 3] (x,y,z)
% mode 'd' (degree- default) or 'r' (radian) if input is direction and not cartesian

if size(ref,1) ~= size(in,1)
    ref=repmat(ref,size(in,1),1);
end

if nargin<3 mode='d'; end
if size(ref,2)<3
    % direction input
    if strcmpi(mode,'d')
        ref=deg2rad(ref);
        in=deg2rad(in);
    end
    [xr,yr,zr]=sph2cart(ref(:,1),ref(:,2),ones(size(ref(:,1))));
    [x,y,z]=sph2cart(in(:,1),in(:,2),ones(size(in(:,1))));
    ref = [xr(:) yr(:) zr(:)];
    in = [x(:) y(:) z(:)];
end
% now ref and in are cartesian and size size
out=atan2d(vecnorm(cross(ref,in,2),2,2),dot(ref,in,2));
end

