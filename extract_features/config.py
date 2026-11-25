import numpy as np

signals_path = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/eeg/eeg.dat" #all accepted recordings after processing
metadata_path = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/meta/exams_processed.csv" #corresponding labels and metadata 

PREPROCESSED_CONN_DIR ="/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/connectivity/"

s00_conn_path =  "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/connectivity/S00_ffDTF.dat"
all_sites_conn_path = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/connectivity/all_sites_ffDTF.dat"

channels = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','P3','Pz','P4','O1','O2']
montage = "colin27_1020" #new name for standard_1020 in MNE

#after all processings (metadata & eeg data)
N_exams = 34831 
N_windows = 150
N_channels = 17
Window_len = 6 #s
Fs = 128 #Hz
dtype = np.float32

freq_bands = {'delta': (1.0, 4.0),
              'theta': (4.0, 8.0),
              'alpha': (8.0, 12.0),
              'beta':  (13.0, 30.0),
              'gamma': (31.0, 40.0)}

f = np.arange(1, 40, 1)
p = 9 #set mvar model order