# Author: Sina Hafezi
# EasyCom AR glasses array loader for SS-Hybrid beamformer
# Paper title: Subspace Hybrid Beamforming for head-worn microphone arrays
# Conference: ICASSP 2023 

# Version history: date/author/updates:
# 2023 Mar - Sina Hafezi - first Python version of EasyCom array loader for SS-Hybrid beamformer

import glob
import h5py
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

def EasyCom_Array(atf_file):
    # DESCRIPTION: load the array ATF (.h5) file containing array's Acoustic Impulse Responses (AIRs) assuming a grid sampling scheme for directions
    # *** OUTPUTS ***
    # AIR        (dict) dictionary {'IR': (nSample,nDirection,nChannel),'fs': (int),'directions': (N,2),'nChan': (int)}
    AIR = {'IR': [],'fs': [],'directions': [],'nChan': [], 'azi': [], 'ele': []}
    # IR: (ndarray) Impulse Responses [nSample x nDirection x nChan]
    # fs: (int) sample rate in Hz
    # directions: (ndarray) (azimuth,elevation) in radians [nDirection x 2] 
    # nChan: (int) number of array's sensor/channel
    # azi: sorted unique azimuths (radians) [nDirection x 1]
    # ele: sorted unique elevations (radians) [nDirection x 1]
    f = h5py.File(atf_file,'r')
    #groups = list(f.keys())
    AIR['fs'] = int(list(f['SamplingFreq_Hz'])[0][0])
    AIR['IR'] = np.array(f['IR']) # (ndarray) [nSample x nDirection x nChan]
    AIR['ele'] = (np.pi/2)-np.array(f['Theta']) # (ndarray) elevation in radians [1 x nDirection]
    AIR['azi'] = np.array(f['Phi']) # (ndarray) azimuth in radians [1 x nDirection]
    AIR['directions'] = np.concatenate((AIR['azi'],AIR['ele']),axis=0).T # (ndarray) [nDirection x 2]
    AIR['ele'] = np.sort(np.unique(AIR['ele'])) # (ndarray) [nElevation x 1]
    AIR['azi'] = np.sort(np.unique(AIR['azi'])) # (ndarray) [nAzimuth x 1]
    AIR['nChan'] = AIR['IR'].shape[-1]
    f.close()
    return AIR

