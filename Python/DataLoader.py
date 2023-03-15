# Author: Sina Hafezi
# Data loader for SS-Hybrid beamformer
# Paper title: Subspace Hybrid Beamforming for head-worn microphone arrays
# Conference: ICASSP 2023 

# Version history: date/author/updates:
# 2023 Mar - Sina Hafezi - first Python version of data loader for SS-Hybrid beamformer

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

class DataLoader:
    fps = 20 # OptiTrack rate
    file_duration = 6 # sec
    total_frames = file_duration * fps # total number of OptiTrack frames per file
    total_sources = 7 # total number of possible participants
    wearer_ID = 2 # participant ID wearing the array
    ref_chan = 2 # array reference channel
    root_path = '' # path to folder containing data
    array_file = 'array.wav' # array audio file
    ht_file = 'ht.json' # head tracking data file
    info_file = 'info.txt' # case study info (chunk info in EasyCom dataset)
    vad_file = 'vad.json' # voice activity labels file
    target_ID = 0 # to be determined later
    IDs = [] # list of present participant IDs
    ht = [] # interpolant function of array rotation quaternions over time
    t = [] # time samples for array rotation data
    
    
    def __init__(obj,inpath):
        obj.root_path = inpath
        obj.set_info()
        obj.set_participant_IDs()
        obj.set_wearer_ht()
        
        
    def set_info(obj):
        # Description: extract and set the target ID from the file info
        c = open(os.path.join(obj.root_path, obj.info_file),'r')
        obj.target_ID = int([match for match in c if 'Target ID' in match][0].split('Target ID: ')[1])
        
    def set_wearer_ht(obj):
        # Description: extract and set the array rotation that is head-rotation of the Ego-centric subjet (participant wearing the array) over time
        obj.ht, obj.t = obj.get_ht(obj.wearer_ID)
        
    def set_participant_IDs(obj):
        # Description: extract and set the ID of participants present in a file
        f = open(os.path.join(obj.root_path,obj.ht_file))
        data = json.load(f)
        obj.IDs = [n['Participant_ID'] for n in data[0]['Participants']]
        f.close()
        
    def get_target_doa(obj):
        # Description: return target DOA over time
        # *** OUTPUTS ***
        # doas      # (ndarray) DOA (azimuth,elevation) in radians wrt to wearer [N x 2] 
        doas = obj.get_doa(obj.target_ID,obj.t)
        doas = np.rad2deg(doas)
        return doas
        
    def get_doa(obj,src_ID,t):
        # Description: returns the DOAs of requested participant (except wearer) wrt. wearer over requested times
        # *** INPUTS ***
        # src_ID       (int)   participant ID of interest
        # t            (ndarray) times of interest in sec [N x 1] 
        # *** OUTPUTS ***
        # doas      # (ndarray) DOA (azimuth,elevation) in radians wrt to wearer [N x 2] 
        src_pos = obj.get_pos() # dictionary of position interpolant function
        q = obj.ht(t) # [N x 4] wearer head-rotation
        pos0 = src_pos[obj.wearer_ID](t) # [N x 3] wearer position
        pos = src_pos[src_ID](t) # [N x 3]  source position
        pos = pos - pos0 # [N x 3] relative source position wrt to wearer
        doas = np.zeros((len(t),2)) # [N x 2] relative doa (azimuth,elevation) in radians wrt to wearer in head-rotated coordiante system
        for n in range(0,len(t)):
            rot_pos = obj.rotate_sys(q[n,:],pos[n,:]) # [x y z] cartesian position in head-rotated coordiantes
            az, el, _ = obj.cart2sph(rot_pos[0],rot_pos[1],rot_pos[2]) 
            doas[n,:] = [az,el]
        return doas
    
    def rotate_sys(obj,q,point):
        # Description: return a point cartesian in a rotated cartesian system
        # *** INPUTS ***
        # q       (ndarray)  rotation quartenions (xyzw) of the system [1 x 4]
        # point   (ndarray) cartesian position (xyz) of a point [1 x 3] in default world coordinate system
        # *** OUTPUTS ***
        # point   (ndarray) cartesian position (xyz) of a point [1 x 3] in rotated world coordinate system
        Rot = Rotation.from_quat(q)
        point = np.dot(point,Rot.apply(np.eye(3)).transpose())
        return point
    
    
    def get_ht(obj,src_ID):
            # Description: returns the head-rotation of a particular participant over time
            # *** INPUTS ***
            # src_ID      (int)    participant ID of interest
            # *** OUTPUTS ***
            # t          (ndarray) OptiTrack time samples in sec [total_frames x 1]
            # q          (SciPy 1D-interpolant) quaternions interpolant function. q(t) returns the OptiTrack quaternions (qx qy qz qw) as [total_frames x 4] ndarray
            # *** Calling Example ***
            # e.g. q(t) returns ndarray [total_frames x 4] that is interpolated quarternions (xyzw) values over time samples in vector t for requested Participant 
            f = open(os.path.join(obj.root_path,obj.ht_file))
            data = json.load(f)
            nFrame = len(data)
            qv = np.zeros((nFrame,4)) # quaternions samples for src_ID
            IDs = [n['Participant_ID'] for n in data[0]['Participants']]
            t = np.arange(0,nFrame)/obj.fps
            if src_ID in IDs:
                m = IDs.index(src_ID)
                for n in range(0,nFrame):
                    qx = data[n]['Participants'][m]['Quaternion_Z']
                    qy = data[n]['Participants'][m]['Quaternion_X']
                    qz = data[n]['Participants'][m]['Quaternion_Y']
                    qw = data[n]['Participants'][m]['Quaternion_W']
                    qv[n,:] = [qx,qy,qz,qw]        
            q = interp1d(t,qv,kind='linear',axis=0)    
            f.close()
            return q, t
    
    def get_pos(obj):
        # Description: returns the cartesian position of all present participants wrt. world coordinate
        # *** OUTPUTS ***
        # src_pos    (dict) dictionary where key (int) is participant ID and value is SciPy 1D-interpolant cartesian interpolant function. 
        # *** Output Example ***
        # e.g. src_pos[4](t) returns ndarray [N x 3] that is interpolated cartesian (xyz) values over time samples in vector t (ndarray N x 1) for Participant ID 4
        f = open(os.path.join(obj.root_path,obj.ht_file))
        data = json.load(f)
        nFrame = len(data)
        src_pos = dict.fromkeys(np.array(obj.IDs))
        for si in range(0,len(obj.IDs)):
            pos = np.zeros((nFrame,3)) #[nFrame x (x,y,z)]
            for fi in range(0,nFrame):
                frame = data[fi]['Participants'][si]
                pos[fi,0] = frame['Position_Z'] # X
                pos[fi,1] = frame['Position_X'] # Y
                pos[fi,2] = frame['Position_Y'] # Z
            src_pos[obj.IDs[si]] = interp1d(obj.t,pos,kind='linear',axis=0)
        f.close()
        return src_pos
    
    def cart2sph(obj,x,y,z):
        # Description: converts cartesian to spherical coordinate
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    def sph2cart(obj,az,el,r):
        # Description: converts spherical to cartesian coordinate
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z
    
    def get_VAD(obj):
        # Description: returns the vad of all sources over time
        # *** OUTPUTS ***
        # vads          (ndarray) VAD binary matrix [toral_sources x total_frames] 
        vads = np.zeros((obj.total_sources,obj.total_frames), dtype='int')
        f = open(os.path.join(obj.root_path,obj.vad_file))
        data = json.load(f)
        for n in range(0,len(data)):
            s1=data[n]['Start_Frame']
            s2=data[n]['End_Frame']
            ID=data[n]['Participant_ID']
            vads[ID-1,(s1-1):(s2-1)]=1
        f.close()
        return vads
    
    def plot_VAD(obj):
        # Description: plots the VADs activity of all sources over over time
        vads =  obj.get_VAD()
        nSrc, nFrame = vads.shape
        plt.figure()
        plt.pcolormesh(np.arange(0,nFrame)/obj.fps,np.arange(0,nSrc)+1,vads, shading='auto', cmap='binary', vmin = 0, vmax = 1)
        plt.xlabel('[sec]')
        plt.ylabel('Participant ID')
        plt.grid(linestyle = ':')
        plt.title('Voice Activity (VAD), target ID: %d' %(obj.target_ID))
    