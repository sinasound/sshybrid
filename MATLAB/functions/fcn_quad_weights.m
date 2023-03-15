function w = fcn_quad_weights(az,el)
% Author: Sina Hafezi
% DESCRIPTION: get quadrature weights for a vectorized grid set of directions
%   --- INPUTS ---
%   az            vectorized grid of azimuths (deg) [nDir x 1]
%   el            vectorized grid of elevations (deg) [nDir x 1]
%   NOTE: nDir = nAzi * nEle where nAzi and nEle are respectively the
%   number of unique azimuth and elevations
%   --- OUTPUTS ---
%   w           normalzied quadrature weights [nDir x 1]

% NOTE: quadrature formulation assumes symmetric distribution of elevations
% around 0. If the the set of unique elevations are not symmetirc, we first
% perform symmetrization, calculates the complete set of weights and then
% crop to the original range of elevations. It is common to have assymetric
% elevation sampling (e.g. typically bottom elevation stops earliers than
% than top elevation due to difficulty of IR measurements below the horizon

% keep min and max of original elevation
minEl = min(el);
maxEl = max(el);

az = deg2rad(az);
inc = deg2rad(90-el); % elevation to inclination
nAzi = length(unique(az)); % number of unique azimuths
nInc = length(unique(inc)); % number of unique inclinations

ele = unique(sort(el)); % vector of unique elevation samples
nEle = length(ele); % number of elevation samples

numPos = sum(ele>0); % number of positive elevations
numNeg = sum(ele<0); % number of negative elevations

if numPos ~= numNeg
    % assymetric distribution of elevation around 0 (symmetrization requried)
    if numPos > numNeg
        % positive elevations are more complete (use them)
        ind = find(ele>0);
    else
        ind = find(ele<0);
    end
    el2 = sort(unique(abs(ele(ind(:))))); % hald-sided (positive) complete set of unique elevations
    if (numPos+numNeg<length(ele)) % check if zero is included
        ele2 = [-el2(end:-1:1);0;el2];
    else
        ele2 = [-el2(end:-1:1);el2];
    end
    % now ele2 is a symmetric set of elevations
    ele2 = ele2(end:-1:1); % [nEle x 1] descending (+90 -> -90)
    nEle = length(ele2);
    inc = deg2rad(90- reshape(repmat(ele2(:),1,nAzi),[],1) ); % [nDir2 x 1] turning into grid and conversion from elevation to inclination
    
end
p = [0:(.5*nEle-1)];
w=2*sin(inc) .* sum(sin((2*p+1).*inc)./(2*p+1),2) / (nAzi*nEle);
w=w./max(w(:)); 

if numPos~=numNeg
    % symmetrization was done. Now remove added elements to match the size
    % of original elevation range and nDir
    w = reshape(w,nEle,[]); % 1D rolled vector to 2D spatial grid [nEle2 x nAzi]
    w = w(find((ele2>=minEl) & (ele2<=maxEl)),:); % [nEle x nAzi] keep the original range for elevation
    w = w(:); % 2D grid to 1D vector (vecotirzation) [nDir x 1]
end
end

