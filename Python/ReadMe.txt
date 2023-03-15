Subspace-Hybrid MVDR Beamforming README for Python

Author: Sina Hafezi
Paper: Subspace Hybrid Beamforming for Head-worn Microphone Arrays
Conference: ICASSP 2023


SUMMARY: 
A new adaptive beamformer outperforming conventional superdirective beamformer. The beamformer is based on a dictionary of pre-calculated weights for various noise covariance matrix models assuming different noise field isotropies.

TEST DATA:
A test array ATF (from EasyCom dataset [1]) and two example test cases from EasyCom [1] dataset are provided.

HOW TO RUN:
1) Either download the Dictionary for Python 'sshybrid_dictionary.h5' (200 MB) from 
https://www.dropbox.com/s/8ffyefd40kpjr6z/sshybrid_dictionary.h5?dl=0
and place it in the same directory as the Python codes OR create the dictionary by running 'run_create_dictionary.py' (takes about 10 minute) to generate and save the same dictionary file.
2) Run 'run_sshybrid.py' which returns the output audio, plots the results and optionally saves the output audio/plots.

USE IT WITH YOUR OWN ARRAY/DATASET:
You would need to replace 'EasyCom_Array.py' function (for array) and 'DataLoader.py' class (for dataset) with your own codes and re-generate a new dictionary using new function/class. Make sure the output of your own customized function/class follows the same format as the 'EasyCom_Array.py' function and 'DataLoader.py' class.

CONTACT:
sina.clamet@gmail.com (private - preferred)
sina.hafezi@imperial.ac.uk (academic)

REFERENCES:
[1] J. Donley, etc. "EasyCom: An Augmented Reality Dataset to Support Algorithms for Easy Communication in Noisy Environments" 
https://doi.org/10.48550/arXiv.2107.04174

