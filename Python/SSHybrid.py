# Author: Sina Hafezi
# Processor class for SS-Hybrid beamformer
# Paper title: Subspace Hybrid Beamforming for head-worn microphone arrays
# Conference: ICASSP 2023 

# Version history: date/author/updates:
# 2023 Mar - Sina Hafezi - first Python version of processor class for SS-Hybrid beamformer

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal
import librosa
import h5py
import numpy.matlib
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate import interp1d

class SSHybrid:
    out_chan = [2] # (list) channels at which to get enhanced audio (indexing from 1)
    
    total_sensor = [] # (int) total number of sensors
    dirs = [] # (ndarray)available ATFs directions (azimuth,elevation) in radians [nDirection x 2] 
    azi = [] # (ndarray) unique azimuths (x-axis) in radian [nAzimuth x 1]
    ele = [] # (ndarray) unqiue elevations (y-axis) in radian [nElevation x 1]
    nAzi =[] # (int) length of azi
    nEle = [] # (int) length of ele
    
    IR = [] # (ndarray) AIR directory [nSample x nDirection x nChan]
    ATF = [] # (ndarray) fft of IR [nFreq x nDirection x nChan] (Acoustic Transfer Function)
    f = [] # (ndarray) frequency axis
    t = [] # (ndarray) time axis
    
    mics = [] # (list) mics subset
    fs = [] # (int) sample rate in Hz
    nChan = [] # (int) number of sensors in mics subset
    nDir = [] # (int) number of measrued directions in AIR/ATF
    nFreq = [] # (int) number of frequencies
    nSample = [] # (int) number of time-domain samples in AIR
    nOut = [] # (int) number of output channels
    
    pca_time_const = 80e-3 # (sec) time constant for exponential moving average used in PCA temporal smoothing
    
    # STFT params (dict)
    stft_params = {'window': 'hann', 'winL': 16e-3, 'stepL': 8e-3}
    nfft = [] # FFT size in sample
    noverlap = [] # step size in sample
    
    # dictionary parameters
    dict_default_name = 'sshybrid_dictionary.h5' # default file name used to load/save dictioinary
    dictw_keys = ['w','f','steer_dirs','fs','out_chan'] # keys used in the dictionary (see below for more details)
    dictw = dict.fromkeys(dictw_keys) # (dict) dictionary of weights for Hybrid-MVDR beamformer
    # dictw['w']: (ndarray) table of beamformer (conjugate) weights [nChan x nFreq x nModel x nSteer x nOut]. first model is always Isotropic
    # dictw['f']: (ndarray) vector of frequency bands (Hz) [nFreq x 1]
    # dictw['steer_dirs'] (ndarray) steering directions (azimuth,elevation) in deg [nSteer x 2]
    # dictw['fs']: (int) sample rate (Hz)
    # dictw['out_chan']: (list) list of channel index (starting from 1) at which the weights are calculated
    
    w = []# Hybrid-MVDR beamforning STFT conj weights [nChan x nFreq x nFrame x nOut]
    Y_hyb = [] # output of Hybrid-MVDR in STFT domain [nOut x nFreq x nFrame]
    Y_iso = [] # output of Iso-MVDR in STFT domain [nOut x nFreq x nFrame]
    Y = [] # enhanced signal in STFT domain [nOut x nFreq x nFrame]
    best_model = [] # index of selected model in STFT domain [nOut x nFreq x nFrame]
        
    # for condition limiting on noise covariance matrix models (not used by default - see 'make_dictionary' function if needed)
    diag_load_mode = 'cond' # type of diagonal loading. 'cond': limit maximum condition number. 'const': add a constant identity. [] to disable diag loading
    diag_load_val = int(1e5) # max condition number for 'cond' mode, constant value for 'const' mode
    
    
    def __init__(obj,air,fs = [],mics = [],out_chan = out_chan):
        # DESCRIPTION: Intakes array's AIR and initialises the parameters
        # *** INPUTS *** 
        # air  (dict) dictionary of AIR (output of get_all_AIRs() in dataset class)
        # fs (int) processing sample rate. [] for default
        # mics (list) list of mic subsets to be used for beamforming (indexing from 1). [] for all
        # out_chan (list) list of mic subsets at which the beamforming output are obtained (indexing from 1)
        obj.IR = air['IR'] # (ndarray) Impulse Responses [nSample x nDirection x nChan]
        fs0 = air['fs'] 
        obj.total_sensor = air['nChan']
        obj.dirs = air['directions'] #  (ndarray) (azimuth,elevation) in radians [nDirection x 2] 
        obj.dirs[:,0] = obj.wrapToPi(obj.dirs[:,0])
        obj.dirs[:,1] = obj.wrapToPi(obj.dirs[:,1])
        obj.azi = obj.wrapToPi(air['azi']) # (ndarray) [nAzimuth x 1]
        obj.ele = obj.wrapToPi(air['ele']) # (ndarray) [nElevation x 1]
        if not fs:
            obj.fs = fs0
        else:
            obj.fs = int(fs)
        if not mics:
            obj.mics = list(np.arange(obj.total_sensor)+1)
        else:
            obj.mics = mics
        obj.nChan = len(obj.mics)
        
        obj.IR = obj.IR[:,:,list(np.array(obj.mics)-1)] # keeping the IRs for required mics subset
        nSample, nDir, nChan = obj.IR.shape
        if obj.fs!=fs0: # resamplilng IRs if needed
            obj.IR = np.transpose(obj.IR) # [nChan x nDir x nSample ]
            obj.IR = obj.IR.reshape((-1,nSample)) # [nChan*nDir x nSample ]
            obj.IR = librosa.resample(obj.IR,orig_sr=fs0,target_sr=obj.fs) # [nChan*nDir x nSample2 ]
            obj.IR = obj.IR.reshape((nChan,nDir,-1))  # [nChan x nDir x nSample2 ]
            obj.IR = np.transpose(obj.IR) # [nSample2 x nDir x nChan]
            nSample, nDir, nChan = obj.IR.shape
        
        obj.nSample = nSample
        obj.nChan = nChan
        obj.nDir = nDir
        
        obj.nAzi = np.prod(obj.azi.shape)
        obj.nEle = np.prod(obj.ele.shape)
        
        obj.set_out_channels(out_chan)
        obj.prepare_ATF()
          
    
    def set_out_channels(obj,c):
        # DESCRIPTION: update the requested output channels for beamformer
        # *** INPUTS ***
        # c   (list) list of mic subsets at which the beamforming output are obtained (indexing from 1)
        obj.out_chan = c 
        obj.nOut = len(c)
         
    def interpolate_target_doa(obj,target_doa,new_t):
        # DESCRIPTION: finds the interpolated DOAs at requested time samples
        # *** INPUTS ***
        # target_doa   (ndarray) target_doa over time samples (t_sec, azi_deg, ele_deg) as [total_frames x 3]
        # new_t        (ndarray) vector of requested time samples [N x 1]
        # *** OUTPUT ***
        # doas         (ndarray) interpolated doas at new times (azi,ele) in radians [N x 2]
        t = target_doa[:,0]
        x, y, z = obj.sph2cart(np.deg2rad(target_doa[:,1]), np.deg2rad(target_doa[:,2]), np.ones(t.shape))
        target_pos_interp = interp1d(t,np.concatenate([x[:,None],y[:,None],z[:,None]],axis=1),kind='previous',axis=0,fill_value='extrapolate')
        pos = target_pos_interp(new_t)
        az, el, _ = obj.cart2sph(pos[:,0],pos[:,1],pos[:,2]) 
        doas = np.concatenate([az[:,None],el[:,None]],axis=1)
        return doas
    
        
    def set_stft_params(obj,window='hann', winL=16e-3, stepL=8e-3):
        # DESCRIPTION: set the settings for STFT
        # *** INPUTS ***
        # window (str)  type of window
        # winL (float) time frame size in sec
        # stepL (float) time frame step in sec
        obj.stft_params['window'] = window
        obj.stft_params['winL'] = winL
        obj.stft_params['stepL'] = stepL
        obj.prepare_ATF()
    
    def prepare_ATF(obj):
        # DESCRIPTION: obtains array's Acoustic Transfer Function (ATF) from Acoustic Impulse Response (AIR) using FFT
        obj.nfft = round(obj.stft_params['winL'] * obj.fs)
        obj.noverlap = round(obj.stft_params['stepL'] * obj.fs)
        
        # AIR to ATF (fft)
        obj.ATF = rfft(obj.IR,n=obj.nfft,axis=0)  # [nFreq x nDirection x nChan]
        obj.f = rfftfreq(obj.nfft,1/obj.fs)
        obj.nFreq = len(obj.f)
    
    
    def get_linear_unimodal_aniso(obj,pow_dyn=-10,azi_mode=0):
        # DESCRIPTION: return a horizontally linear unimodal anisotropic isotropy
        # *** INPUTS ***
        # pow_dyn       (float) power dynamic range of distribution in (dB)
        # azi_mode       (float) mode or peak azimuth of directionality for noise field isotropy
        # *** OUTPUT ***
        # p         (ndarray) 2D horizontal isotropy [nAzi x 1] in (dB)
        p = pow_dyn * np.abs(obj.wrapToPi((obj.azi-np.deg2rad(azi_mode)))/np.pi)
        return p
    
    def generate_2D_anisotropies(obj,pow_dyns=[-10],azi_modes=np.arange(0,360,6)):
        # DESCRIPTION: returns a series of 2D isotrpies for requested set of power dynamic(s) and azimuth mode(s)
        # *** INPUTS ***
        # pow_dyns       (list) of power dynamic range(s) of distribution in (dB)
        # azi_modes       (list) of mode or peak azimuth(s) of directionality for noise field isotropy
        # *** OUTPUT ***
        # ps    (ndarray) horizontal isotropies in dB. [nAzi x nMode x nPow]
        nPow = len(pow_dyns)
        nMode = len(azi_modes)
        nAzi = obj.nAzi
        ps = np.zeros((nAzi,nMode,nPow)) # (ndarray) horizontal isotopy of models
        for azi_mode, mi in zip(azi_modes,range(nMode)):
            for pow_dyn, di in zip(pow_dyns,range(nPow)):
                ps[:,mi,di] = obj.get_linear_unimodal_aniso(pow_dyn,azi_mode)
        return ps
    
    def convert_2D_to_3D_isotropy(obj,ps):
        # DESCRIPTION: converts 2D to 3D isotropies (by repeating and quarature weighting along elevation)
        # *** INPUTS ***
        # ps    (ndarray) horizontal (2D) isotropies in dB. [nAzi x ...]
        # *** OUTPUT ***
        # ps3   (ndarray) full-space (3D) isotropies [nDir x ...]
        if len(ps.shape)<2:
            ps = ps[:,None]
        in_shape = list(ps.shape)
        ps = ps.reshape((ps.shape[0],-1),order='F')
        
        nModel = ps.shape[1]
        ps3 = np.repeat(ps[:,:,None],obj.nEle,axis=2) # [nAzi x nModel x nEle]
        ps3 = np.transpose(ps3,axes=[2,0,1]).reshape((-1,nModel),order='F') # [nDir x nModel]
        w_quad = obj.get_quadrature_weights() # [nDir x 1]
        ps3 = 10*np.log10( (10**(ps3/10)) * w_quad ) # quadrature weighting in linear domain and back to dB again
        in_shape[0] = ps3.shape[0]
        ps3 = ps3.reshape(tuple(in_shape),order='F')
        return ps3
        
    
    def make_dictionary(obj,pow_dyns=[-10],azi_modes=np.arange(0,360,6),include_eye=True,steer_az_limit=None,steer_el_limit=None):
        # DESCRIPTION: create a dictionary for Hybrid beamformer using Isotropic and unimodal Anisotropic noise field model (optionally inlucde identity model)
        # *** INPUTS ***
        # pow_dyns          (list) of power dynamic range(s) of anisotropic distribution in (dB)
        # azi_modes         (list) of mode or peak azimuth(s) of directionality for noise field isotropy
        # include_eye       (bool) wether to include identity model among noise covariance matrix models
        # steer_az_limit    (list) min and max of steering range for azimuth [1 x 2]. None means full coverage [-180,180]
        # steer_el_limit    (list) min and max of steering range for elevation [1 x 2]. None means full coverage [-90,90]
        if not steer_az_limit:
            steer_az_limit = [-180,180]
        if not steer_el_limit:
            steer_el_limit = [-90,90]
        steer_az_limit = obj.wrapToPi(np.deg2rad(steer_az_limit))
        steer_el_limit = obj.wrapToPi(np.deg2rad(steer_el_limit))
        steer_dirs = obj.dirs #[nDir x 1]
        steer_dirs = steer_dirs[np.logical_and(steer_dirs[:,0]>=steer_az_limit[0],steer_dirs[:,0]<=steer_az_limit[1]),:] # trimming azimuths
        steer_dirs = steer_dirs[np.logical_and(steer_dirs[:,1]>=steer_el_limit[0],steer_dirs[:,1]<=steer_el_limit[1]),:] # trimming elevations
        
        steer_ind = np.squeeze(np.where( (steer_dirs[:,0]>=steer_az_limit[0]) & (steer_dirs[:,0]<=steer_az_limit[1]) & (steer_dirs[:,1]>=steer_el_limit[0]) & (steer_dirs[:,1]<=steer_el_limit[1]) ))
        
        nSteer = len(steer_ind)
        nFreq = obj.nFreq
        nChan = obj.nChan
        nOut = obj.nOut
        nDir = obj.nDir
        outChan_ind = np.array(obj.out_chan)-1

        ps_ansio = obj.generate_2D_anisotropies(pow_dyns,azi_modes) # [nAzi x nMode x nPow] (dB0)
        ps_aniso = ps_ansio.reshape((obj.nAzi,-1),order='F') # [nAzi x nMode*nPow] (dB)
        ps_iso = np.zeros((obj.nAzi,1)) # [nAzi x 1] (dB)
        ps = np.concatenate((ps_iso,ps_aniso),axis=1) # [nAzi x 1+nMode*nPow]
        ps3 = obj.convert_2D_to_3D_isotropy(ps) # full-space isotropies [nDir x 1+nMode*nPow]
        ps3 = 10**(ps3/10) # dB to pow isotropies
        nModel = ps3.shape[1]
        if include_eye:
            nModel += 1
                
        RHH = obj.ATF[:,:,:,None] @ np.conj(obj.ATF[:,:,None,:]) # PW covariance matrices [nFreq x nDir x nChan x nChan]
        
        obj.dictw['f'] = obj.f
        obj.dictw['steer_dirs'] = np.rad2deg(steer_dirs)
        obj.dictw['fs'] = obj.fs  
        obj.dictw['out_chan'] = obj.out_chan
        obj.dictw['w'] = np.zeros((nChan,nFreq,nModel,nSteer,nOut),dtype=np.complex64) # conjugate weights        
        print_cycle=50 # number of interations to wait before updating progress print (keep it high as the loop is fast)
        cntr = 0
        for di in range(nSteer): # iterate over steering directions
            h = np.squeeze(obj.ATF[:,steer_ind[di],:]) #  ATFs at steering direction [nFreq x nChan]
            for fi in range(nFreq): # iterate over freq bands
                if (cntr%print_cycle==0):
                    print('\r','Preparing Dictionary %%%2.2f (Steering: %d/%d, Freq: %d/%d)' % (100*cntr/(nSteer*nFreq),di+1,nSteer,fi+1,nFreq),end='\r')
                for mi in range(nModel): # iterate over models
                    if (mi==(nModel-1) and include_eye): # if it's identity model
                        R = np.eye(nChan) # [nChan x nChan]
                    else:
                        pow_isotropy = ps3[:,mi] # [nDir x 1]
                        R = np.squeeze(RHH[fi,:,:,:]) * pow_isotropy[:,None,None] # [nDir x nChan x nChan]
                        R = np.squeeze(np.sum(R,axis=0)) # [nChan x nChan]
                        # R = obj.diag_load_cov(R)  # in case condition number limiting needed (not needed in our example)
     
                    for oi in range(nOut): # iterate over output channel
                        steer_vec = h[fi,:] / h[fi,outChan_ind[oi]] # [1 x nChan]
                        obj.dictw['w'][:,fi,mi,di,oi] = np.conj(obj.get_mvdr_weights(R,steer_vec)) # [nChan x 1]
                cntr+=1
        print('\r','Preparing Dictionary %%%2.2f - Done!' % (100),end='\n')  
        
    def save_dictionary(obj,file_name=None):
        # DESCRIPTION: save current dictionary as a .h5 file for future use
        # *** INPUTS ***
        # file_name    (str) full path to file name to be saved by. Leave it for default file name 
        print('Saving dictionary')
        if not file_name:
            file_name = obj.dict_default_name
        fields = list(obj.dictw.keys())
        hf = h5py.File(file_name,'w')
        for field in fields:
            hf.create_dataset(field,data=obj.dictw[field])
        hf.close()
        print('- Done')
        
    def load_dictionary(obj,file_name=None):
        # DESCRIPTION: load a previously-saved dictionary as from a .h5 file
        # *** INPUTS ***
        # file_name    (str) full path to file name to be read from. Leave it for default file name
        print('Loading dictionary')
        if not file_name:
            file_name = obj.dict_default_name
        hf = h5py.File(file_name,'r')
        try:
            for field in obj.dictw_keys:
                obj.dictw[field] = np.array(hf[field])
            obj.dictw['fs']=int(obj.dictw['fs']) 
            print('- Done')
        except:
           print('\n[Error] Invalid File: File does not contain the correct dataset names') 
        hf.close()
        
    def wrapToPi(obj,angles):
        # DESCRIPTION: wrap angles to [-Pi,Pi] radian range
        # *** INPUTS ***
        # angles    (ndarray) vector of angles in radians [N x 1]
        return np.arctan2(np.sin(angles),np.cos(angles))
      
        
    def diag_load_cov(obj,R):
        # DESCRIPTION: diagonal loading of covariance matrix
        # *** INPUTS ***
        # R (ndarray)  covariance matrix [nChan x nChan]
        if obj.diag_load_mode=='const':
            R = R + obj.diag_load_val * np.eye(obj.nChan)
        else:
            cn0 = np.linalg.cond(R) # original condition number
            threshold = obj.diag_load_val 
            if cn0>threshold:
                ev = np.linalg.eig(R)[0] # eigenvalues only
                R = R + np.eye(obj.nChan) * (ev.max() - threshold * ev.min()) / (threshold-1)
        return R
                
    def get_quadrature_weights(obj):
        # DESCRIPTION: returns the quadrature weights for the elevations samples
        # NOTE: These weights are needed to compensate for higher density of points closer to the poles due to uniform grid sampling of full-space. For non-grid sampling scheme you would need different weighting strategy
        minEl = np.min(obj.dirs[:,1])
        maxEl = np.max(obj.dirs[:,1])
        ele = obj.ele
        inc = (np.pi/2)-obj.dirs[:,1] # [nDir x 1]
        nEle = obj.nEle
        nAzi = obj.nAzi
        
        numPos = np.sum(obj.ele>0)
        numNeg = np.sum(obj.ele<0)
        if numPos!=numNeg:
            # assymetric elevation sampling (symmetrization required)
            if numPos>numNeg:
                # positive elevations are more complete (use them)
                ind = np.where(obj.ele>0)
            else:
                ind = np.where(obj.ele<0)    
            el=np.sort(np.abs(obj.ele[ind])) # half-sided (positive) complete set of unique elevations
            if (numPos+numNeg)<len(obj.ele): # check if zero is included
                ele2 = np.concatenate([-el[::-1],np.array([0]),el])
            else:
                ele2 = np.concatenate([-el[::-1],el])
            # now ele2 is a symmetric set of elevations
            ele2 = ele2[::-1] # descending (+90 --> -90)
            nEle = len(ele2)
            inc = (np.pi/2) - np.tile(ele2,obj.nAzi) # [nDir2 x 1]
        
        m = np.arange(nEle/2)
        m = m[:,None].T # (1,M)
        w = np.sum(np.sin(inc[:,None] * (2*m+1)) / (2*m+1),axis=1) * np.sin(inc) * 2 / (nEle*nAzi) # [nDir2 x 1]
        w = w / np.max(w)
        
        if numPos!=numNeg:
            # symmetrization was done. now remove added elements to match the size of original nDir
            w = w.reshape(len(ele2),-1,order='F') # 1D [nDir2 x 1] vector to 2D matrix [nEle2 x nAzi] for quadrature weights
            w = w[np.where( (ele2>=minEl) & (ele2<=maxEl) ),:] # [nEle x nAzi]
            w = w.reshape(-1,1,order='F') # [nDir x 1]
        
        return w
              
       
    def get_angle_between(obj,ref,x,unit='radian'):
        # DESCRIPTION: calculates the angular separation between the directions (in either cartesian or DOA format)
        # *** INPUTS *** 
        # ref       (ndarray) reference direction(s) [1x2] or [Nx2] (azimuth,elevation) or [1x3] or [Nx3] (x,y,z)
        # x         (ndarray) subject direction(s) [Nx2] (azimuth,elevation) or [Nx3] (x,y,z)
        # unit      (str)   unit of DOA if ref and x inputs are DOAs 'radian' or 'degree'
        # *** OUTPUTS ***
        # a         (ndarray) angle between direction(s) [Nx1] in the same unit as input
        if ref.shape[0] != x.shape[0]:
            # there is one ref and multiple subject. repeat ref to match the number of subjects
            ref = np.matlib.repmat(ref,x.shape[0],1)
        if ref.shape[1]>2:
            # inputs are in cartesian. convert to spherical angles
            ref = np.array(obj.cart2sph(ref[:,0], ref[:,1], ref[:,2]))[[0,1],:].T # [N x 2] (az,el)
            x = np.array(obj.cart2sph(x[:,0], x[:,1], x[:,2]))[[0,1],:].T # [N x 2] (az,el)
        else:
            # inputs are in spherical angles. check unit
            if unit=='degree':
                ref = np.deg2rad(ref)
                x = np.deg2rad(x)
                
        # ref and x are now both in spherical angles (radian) and match in size
        # use Haversine formula to get angle between
        dlon = ref[:,0] - x[:,0] # azimuth differences
        dlat = ref[:,1] - x[:,1] # elevation differences 
        
        a = np.sin(dlat/2) ** 2 + np.cos(x[:,1]) * np.cos(ref[:,1]) * np.sin(dlon/2) ** 2
        a = 2 * np.arcsin(np.sqrt(a))
        if unit=='degree': # convert unit to match input unit
                a = np.rad2deg(a)
        return a
   
    def cart2sph(obj,x,y,z):
        # DESCRIPTION: converts cartesian to spherical coordinate
        # *** INPUTS ***
        # x  (ndarray) x-coordinate(s) [N x 1]
        # y  (ndarray) y-coordinate(s) [N x 1]
        # z  (ndarray) z-coordinate(s) [N x 1]
        # *** OUTPUTS ***
        # az  (ndarray) azimuth(s) in radians [N x 1]
        # el  (ndarray) elevation(s) in radians [N x 1]
        # r   (ndarray) range(s) in radians [N x 1]
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    def sph2cart(obj,az,el,r):
        # DESCRIPTION: converts spherical to cartesian coordinate
        # *** INPUTS ***
        # az  (ndarray) azimuth(s) in radians [N x 1]
        # el  (ndarray) elevation(s) in radians [N x 1]
        # r   (ndarray) range(s) in radians [N x 1]
        # *** OUTPUTS ***
        # x  (ndarray) x-coordinate(s) [N x 1]
        # y  (ndarray) y-coordinate(s) [N x 1]
        # z  (ndarray) z-coordinate(s) [N x 1]
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z     
   
    def process_stft(obj,X,target_doas):
        # DESCRIPTION: performs SS-Hybrid beamforming including Hybrid- and Is- (Superdirective) MVDR beamforming in the STFT domain 
        # *** INPUTS ***
        # X  (ndarray) signal in the stft domain [nChan x nFreq x nFrame]
        # target_doas  (ndarray) target DOA (azimuth, elevation) in radians [nFrame x 2]
        nChan, nFreq, nFrame = X.shape # [nChan x nFreq x nFrame]
        nOut = len(obj.out_chan)
        print_cycle=100 # number of interations to wait before updating progress print (keep it high as the loop is fast)
        
        obj.w = np.zeros((nChan,nFreq,nFrame,nOut),dtype=np.complex64) # beamforning STFT conj weights [nChan x nFreq x nFrame x nOut]
        obj.Y_hyb = np.zeros((nOut,nFreq,nFrame),dtype=np.complex64) # output of Hybrid-MVDR in STFT domain [nOut x nFreq x nFrame]
        obj.Y_iso = np.zeros((nOut,nFreq,nFrame),dtype=np.complex64) # output of Iso-MVDR in STFT domain [nOut x nFreq x nFrame]
        obj.Y = np.zeros((nOut,nFreq,nFrame),dtype=np.complex64) # enhanced signal in STFT domain [nOut x nFreq x nFrame]
        obj.best_model = np.zeros((nOut,nFreq,nFrame),dtype=np.complex64) # index of selected model in STFT domain [nOut x nFreq x nFrame]
    
        
        R_past = dict.fromkeys(np.arange(nOut)) # last inter-method cov matrix for each output channel used for PCA temporal smoothing
        dictw_steer_dirs = np.deg2rad(obj.dictw['steer_dirs']) # dictionary stearing directions
        alpha = np.exp(-obj.stft_params['stepL']/obj.pca_time_const)
        # STFT processing
        for framei in range(nFrame):
            if (framei%print_cycle==0):
                print('\r','Processing %%%2.2f ' % (100*framei/nFrame),end='\r')
                
            # find the closest dictionary steering direction to the target DOA
            dist = obj.get_angle_between(target_doas[framei,:],dictw_steer_dirs)
            di = np.where(dist == dist.min())[0][0] # nearest neighbour to target doa
            
            
            
            for oi in range(nOut): # iterate over output channels
                
                # obj.dictw['w'] weight dictionary [nChan x nFreq x nModel x nSteer x nOut]
                ws = np.squeeze(obj.dictw['w'][:,:,:,di,oi]) # [nChan x nFreq x nModel]
            
                # Hybrid-MVDR
                Zs = np.squeeze(np.sum(X[:,:,framei,None] * ws,axis=0)) # [nFreq x nModel] out of all models
                model_ind = np.argmin(np.abs(Zs),axis=1) # [nFreq x 1] index of minimum-power (selected) model at each freq band
                obj.best_model[oi,:,framei] = model_ind
                obj.Y_iso[oi,:,framei] = Zs[:,0] # assuming first model is always Iso  
                for freqi in range(nFreq):
                    obj.w[:,freqi,framei,oi] = ws[:,freqi,model_ind[freqi]]
                    obj.Y_hyb[oi,freqi,framei] = Zs[freqi,model_ind[freqi]]
                 
                # inter-method wideband PCA
                Z = np.concatenate([np.squeeze(obj.Y_hyb[oi,:,framei])[:,None], np.squeeze(obj.Y_iso[oi,:,framei])[:,None]],axis=1).T # [2 x nFreq]
                R = Z @ np.conj(Z.T) # inter-method wideband covariance matrix [2x2]
                
                # temporal smoothing (via exponential moving average)
                if R_past[oi] is None:
                    R_past[oi] = R
               
                R = alpha*R_past[oi] + (1-alpha)*R
                R_past[oi] = R
                
                eigenValues, eigenVectors = np.linalg.eig(R)
                idx = eigenValues.argsort()[::-1]   
                #eigenValues = eigenValues[idx]
                eigenVectors = eigenVectors[:,idx]
                U=(eigenVectors[:,0])[:,None] # signal eigen vector [2 x 1]
                Z_ss = U @ np.conj(U.T) @ Z # signal subspace of Z [2 x nFreq]
                obj.Y[oi,:,framei] = Z_ss[0,:]
                
        print('\r','Processing %%%2.2f - Done!' % (100),end='\n')
    
    def get_mvdr_weights(obj,R,d):
        # DESCRIPTION: calculates the MVDR weights
        # *** INPUTS ***
        # R  (ndarray) covariance matrix [nChan x nChan]
        # d  (ndarray) steering vector or Relative Transfer Function (RTF) [nChan x 1]
        # *** OUTPUTS ***
        # w  (ndarray) beamformer conjugate weights in the stft domain  [nChan x 1]
        invRd = np.matmul(np.linalg.pinv(R),d)
        w = invRd/np.matmul(np.conj(d).T,invRd)
        return w
        
    def do_stft(obj,x):
        # DESCRIPTION: convert signal from time domain to STFT domain
        # *** INPUTS ***
        # x  (ndarray) signal in time doamin [nChan x nSample]
        # *** OUTPUTS ***
        # X  (ndarray) signal in STFT domain [nChan x nFreq x nFrame]
        # f  (ndarray) frequency vector [nFreq x 1]
        # t  (ndarray) time vector [nFrame x 1]
        f, t, X = signal.stft(x, fs=obj.fs, window=obj.stft_params['window'], nperseg=round(obj.fs*obj.stft_params['winL']),noverlap=round(obj.fs*obj.stft_params['stepL']))
        obj.f = f
        obj.t = t
        return X, f, t
    
    def do_istft(obj,X):
        # DESCRIPTION: convert signal from STFT to time domain 
        # *** INPUTS ***
        # X  (ndarray) signal in STFT domain [nChan x nFreq x nFrame]
        # *** OUTPUT ***
        # x  (ndarray) signal in time doamin [nChan x nSample]
        # t  (ndarray) time vector [nFrame x 1]
        t, x = signal.istft(X, fs=obj.fs, window=obj.stft_params['window'], nperseg=round(obj.fs*obj.stft_params['winL']),noverlap=round(obj.fs*obj.stft_params['stepL']))
        return x, t
    
    def process_signal(obj,x,target_doa=[]):
        # DESCRIPTION: applies the SS-Hybrid beamforming on a time-doamin signal
        # *** INPUTS ***
        # x  (ndarray) signal in time doamin [nChan x nSample]
        # target_doa (ndarray) target DOA (azimuth,elevation) in degree [nFrame x 2]
        # *** OUTPUT ***
        # y      (ndarray) SS-Hybrid enhanced signal (proposed) [nOut x nSample]
        # y_iso  (ndarray) superdirective enhancedsignal (baseline) [nOut x nSample]
        # y_hyb  (ndarray) Hybrid signal (containing the musical noise) [nOut x nSample]
        
        X, _, t = obj.do_stft(x)
        target_doa = obj.interpolate_target_doa(target_doa,t)
        obj.process_stft(X,target_doa)
        y, _ = obj.do_istft(obj.Y)
        y_iso, _ = obj.do_istft(obj.Y_iso)
        y_hyb, _ = obj.do_istft(obj.Y_hyb)
        return y, y_iso, y_hyb
    
    
    def read_audio(obj,in_file,fs=None,start_t=0.0,duration=None,mics=[]):
        # DESCRIPTION: returns the time-domain audio signal
        # *** INPUTS ***
        # in_file  (str) full path to the audio file
        # fs       (int) optional - output sample rate in Hz, default: original sample rate
        # start_t   (float) optional - start time of segment in sec, default: beginning of the file
        # duration   (float) optional - duration of segment in sec, default: entirely to the end of the file
        # mics      (list) optional - mics/channels subset (indexing from 1), default: all channels
        # *** OUTPUTS ***
        # y   (ndarray) output array audio signal [nChan x nSample]
        # fs  (int)     output sample rate in Hz
        y, fs = librosa.load(in_file,sr=fs,mono=False,offset=start_t,duration=duration)
        if mics:
            y = y[np.array(mics)-1,:]
        return y, fs

    def plot_2D_isotropy(obj,ps,azi):
        # DESCRIPTION: plots the horizontal (2D) isotropy models
        # *** INPUTS ***
        # ps    (ndarray) horizontal (2D) isotropies in dB. [nAzi x ...]
        # azi   (ndarray) vector of azimuths in degrees [nAzi x 1]
        nAzi = ps.shape[0]
        ps = ps.reshape(nAzi,-1,order='F')
        ids = np.concatenate([np.arange(0,nAzi),np.array([0])])
        azi = azi[ids]
        ps = ps[ids,:]
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(azi, ps)
        ax.set_rmax(0)
        ax.set_rmin(np.min(ps.reshape(-1,1)))
        #ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
        ax.set_rlabel_position(0)  # Move radial labels away from plotted line
        ax.grid(True)
        ax.set_theta_zero_location("N")
        
        ax.set_title("Horizontal Isotropy", va='bottom')
        plt.show()
        
    def plot_3D_isotropy(obj,ps3,azi,ele):
        # DESCRIPTION: plots the horizontal (2D) isotropy model
        # *** INPUTS ***
        # ps3   (ndarray) full-space (3D) isotropies [nDir x 1] (nDir = nEle * nAzi)
        # azi   (ndarray) vector of azimuths in degrees [nAzi x 1]
        # ele   (ndarray) vector of elevations in degrees [nEle x 1]
        ele = np.sort(ele)[::-1]
        fig, ax = plt.subplots()
        ps3 = np.squeeze(ps3).reshape((len(ele),-1),order='F')
        ids = np.argsort(azi)
        azi = azi[ids]
        ps3 = ps3[:,ids]
        plt.pcolormesh(np.rad2deg(azi),np.rad2deg(ele),ps3, shading='auto', cmap='magma', vmin=np.min(ps3.reshape(-1,1)), vmax=np.max(ps3.reshape(-1,1)))
        plt.title('Full-space Isotropy')
        plt.ylabel('Elevation [deg]')
        plt.xlabel('Azimuth [deg]')
        plt.gca().invert_xaxis()
        plt.xticks(np.linspace(-180,180,9))
        plt.yticks(np.linspace(-90,90,9))
        plt.colorbar(label = '')
        plt.show()
            
    def plot_stft_sig(obj,x,title='',CL=[-100,-50]):
        # DESCRIPTION: plots the signal spectrogram
        # *** INPUTS ***
        # x     (ndarray) signal in time-domain 
        # title (str) optional title
        # CL    (list) optional min and max for the color axis limit (dB)
        X, f, t = obj.do_stft(x)
        if len(X.shape)==3:
            X = X[0,:,:] # first channel (in case of multi-channel signal)
        fig, ax = plt.subplots()
        plt.pcolormesh(t,f/1e3,20*np.log10(np.abs(X)), shading='auto', cmap='magma', vmin=CL[0], vmax=CL[1])
        plt.ylabel('Freq [kHz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label = '[dB]')
        plt.title(title)
        plt.show()
        
        