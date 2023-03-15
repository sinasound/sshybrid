function R = fcn_condition_limiting(R,threshold)
% Author: Sina Hafezi
% DESCRIPTION: limit maximum condition number of a covariance matrix
%   --- INPUTS ---
%   R               Original covariance matrix [Q x Q]
%   threshold       maximum condition number to cap on
%   --- OUTPUTS ---
%   R               condition limited covariance matrix

Q = size(R,1);
cond_num=cond(R);    
if cond_num>threshold
    e=eig(R);   % eigenvalues
    R=R+(max(e)-threshold*min(e))/(threshold-1)*eye(Q); % add a multiple of the identity
end
end