# Author: Sina Hafezi
# Dictionary generator script for SS-Hybrid beamformer
# Paper title: Subspace Hybrid Beamforming for head-worn microphone arrays
# Conference: ICASSP 2023 

# Version history: date/author/updates:
# 2023 Mar - Sina Hafezi - first Python script to create and save dictionary for SS-Hybrid beamformer


from SSHybrid import *
from EasyCom_Array import *
import os
import numpy as np
import time

# settings for processing
fs = int(10e3)

# path setup
atf_file = os.path.join('..', 'example data','Device_ATFs.h5')

AIR = EasyCom_Array(atf_file) # get EasyCom array AIR (Acoustic Impulse Response)
sshyb = SSHybrid(AIR, fs) # initialize SS-Hybrid for this array

# (OPTIONAL) get and plot few sample 2D & 3D noise field isotropy models
pow_dyns=np.arange(-8,-48,-8) # dynamic ranges for horizontally unimodal anisotropic noise field models
ps = sshyb.generate_2D_anisotropies(pow_dyns) # 2D isotropies
ps3 = sshyb.convert_2D_to_3D_isotropy(ps) # 3D isotropies (quadrature weighted along elevation)
sshyb.plot_2D_isotropy(ps[:,0,:],sshyb.azi) # plot few examples of 2D isotropies
sshyb.plot_3D_isotropy(ps3[:,0,0], sshyb.azi, sshyb.ele) # plot an example of 3D isotropy

# Loading or Making models Dictionary
t0=time.time()
sshyb.make_dictionary(pow_dyns=pow_dyns,steer_az_limit=[-30,30],steer_el_limit=[-90,90])
t = time.time()-t0
print('It took %3.1f minute(s)'%(t/60))
sshyb.save_dictionary()
