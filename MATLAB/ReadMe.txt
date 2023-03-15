Subspace-Hybrid MVDR Beamforming README for MATLAB

Author: Sina Hafezi
Paper: Subspace Hybrid Beamforming for Head-worn Microphone Arrays
Conference: ICASSP 2023


SUMMARY: 
A new adaptive beamformer outperforming conventional superdirective beamformer. The beamformer is based on a dictionary of pre-calculated weights for various noise covariance matrix models assuming different noise field isotropies.

TEST DATA:
A test array ATF (from EasyCom dataset [1]) and two example test cases from EasyCom [1] dataset are provided. Some functions from VOICEBOX [2] toolbox (included) is used.

HOW TO RUN:
1) Either download the Dictionary for MATLAB 'sshybrid_dictionary.mat' (480 MB) from
https://www.dropbox.com/s/4e11j5d52wab7m7/%EF%BB%BFsshybrid_dictionary.mat?dl=0
and place it in the same directory as the MATLAB codes OR create the dictionary by running 'run_create_dictionary.m' (takes about 1 minute) to generate and save the same dictionary file.
2) Run 'run_sshybrid.m' which returns the output audio, plots the results and optionally saves the output audio/plots.

USE IT WITH YOUR OWN ARRAY/DATASET:
You would need to replace 'fcn_extract_IRs_EasyCom.m' function in 'run_create_dictionary.m' script and re-generate a new dictionary. Make sure the output of your own customized 'fcn_extract_IRs_' function follows the same format as the 'fcn_extract_IRs_EasyCom.m'. For dataset, you just need to read-in and pass your target DOAs over time as a matrix of [time_sec, azimuth_deg, elevation_deg]. The rest is compatible.

CONTACT:
sina.clamet@gmail.com (private - preferred)
sina.hafezi@imperial.ac.uk (academic)

REFERENCES:
[1] J. Donley, etc. "EasyCom: An Augmented Reality Dataset to Support Algorithms for Easy Communication in Noisy Environments" 
https://doi.org/10.48550/arXiv.2107.04174
[2] M. Brookes. "VOICEBOX" 
https://github.com/ImperialCollegeLondon/sap-voicebox
http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

