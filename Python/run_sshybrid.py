# Author: Sina Hafezi
# Enhancement processing script for SS-Hybrid beamformer
# Paper title: Subspace Hybrid Beamforming for head-worn microphone arrays
# Conference: ICASSP 2023 

# Version history: date/author/updates:
# 2023 Mar - Sina Hafezi - first Python script to run SS-Hybrid beamforming enhancement


from SSHybrid import *
from EasyCom_Array import *
from DataLoader import *
import os
import numpy as np
import sounddevice as sd # only used to play audio
import soundfile as sf # only used to save audio

# settings for processing
fs = int(10e3)
test_case = 1
out_path = 'outputs'
ToSave = True # whether to save the output audio in out_path folder
if not os.path.exists(out_path):
    os.makedirs(out_path) 

# path setup
atf_file = os.path.join('..', 'example data','Device_ATFs.h5')
data_path = os.path.join('..', 'example data',str(test_case))

AIR = EasyCom_Array(atf_file) # get EasyCom array AIR (Acoustic Impulse Response)
sshyb = SSHybrid(AIR, fs) # initialize SS-Hybrid for this array

# (OPTIONAL) get and plot few sample 2D & 3D noise field isotropy models
pow_dyns=np.arange(-8,-48,-8) # dynamic ranges for horizontally unimodal anisotropic noise field models
ps = sshyb.generate_2D_anisotropies(pow_dyns) # 2D isotropies
ps3 = sshyb.convert_2D_to_3D_isotropy(ps) # 3D isotropies (quadrature weighted along elevation)
sshyb.plot_2D_isotropy(ps[:,0,:],sshyb.azi) # plot few examples of 2D isotropies
sshyb.plot_3D_isotropy(ps3[:,0,0], sshyb.azi, sshyb.ele) # plot an example of 3D isotropy

# Loading or Making Dictionary
#sshyb.make_dictionary(pow_dyns=pow_dyns,steer_az_limit=[-30,30],steer_el_limit=[-90,90])
#sshyb.save_dictionary()
sshyb.load_dictionary()

# READ in target DOA (target doas)
dl = DataLoader(data_path)
dl.plot_VAD() # plot voice activity labels
target_doas = dl.get_target_doa() # (azi_deg, ele_deg) [N x 2]
target_doas = np.concatenate((dl.t[:,None],target_doas),axis=1) # (t_sec, azi_deg, ele_deg) [N x 3]

# READ in naudio
x, _ = sshyb.read_audio(os.path.join(data_path,'array.wav'),fs=fs)
y_pass = x[sshyb.out_chan[0],:]

# PROCESS audio
y_sshyb, y_iso, y_hyb = sshyb.process_signal(x,target_doas)

# PLAY output
#sd.play(y_pass.T, fs) # passthrough signal (no processing)
#sd.play(y_iso.T, fs) # superdiredtive baseline enhanced
#sd.play(y_sshyb.T, fs) # enhaned output (SS-Hybrid) 

# Export audio
if ToSave:
    gain=20;
    sf.write(os.path.join(out_path,'case_%d_%s.wav'%(test_case,'pass')), gain*y_pass.T, fs)
    sf.write(os.path.join(out_path,'case_%d_%s.wav'%(test_case,'iso')), gain*y_iso.T, fs)
    sf.write(os.path.join(out_path,'case_%d_%s.wav'%(test_case,'hyb')), gain*y_hyb.T, fs)
    sf.write(os.path.join(out_path,'case_%d_%s.wav'%(test_case,'sshyb')), gain*y_sshyb.T, fs)

# PLOT STFTs
sshyb.plot_stft_sig(y_pass,'Passthrough')
sshyb.plot_stft_sig(y_iso,'Iso MVDR (Superdirective)')
sshyb.plot_stft_sig(y_hyb,'Hybrid MVDR (containing musical noise)')
sshyb.plot_stft_sig(y_sshyb,'SS-Hybrid MVDR (proposed)')

