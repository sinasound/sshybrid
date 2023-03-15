function fcn_plot_polar(ang,r)
% Author: Sina Hafezi
% DESCRIPTION: polar plot horizontal isotropy
%   --- INPUTS ---
%   ang             azimuths (deg) [nAzi x 1]
%   r               radius [nAzi x 1]

ang=deg2rad(ang);
polarplot(ang([1:end 1]),r([1:end 1]),'LineWidth',2); % close the wrap
ax=gca;
ax.ThetaZeroLocation='top';
ax.ThetaTickLabel=cellstr(num2str(wrapTo180(ax.ThetaTick(1:end-1)')));
%rlim([min(r) max(r)]);
end