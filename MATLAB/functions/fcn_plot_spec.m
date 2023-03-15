function fcn_plot_spec(y,fs)
% Author: Sina Hafezi
% DESCRIPTION: pplot spectrogram of input signal
%   --- INPUTS ---
%   y             time-domain signal [nSample x 1]
%   fs             sample rate (Hz)

[Y,t,f] = stft_quick(y,fs);
imagesc(t,f/1e3,mag2db(abs(squeeze(Y))));
axis xy;xlabel('[sec]');ylabel('[kHz]');
v_colormap('v_thermliny');cl=colorbar;cl.Title.String='[dB]';
end