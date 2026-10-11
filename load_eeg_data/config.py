import numpy as np

signals_path = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/eeg/eeg.dat" #all accepted recordings after processing
metadata_path = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/meta/exams_processed.csv" #corresponding labels and metadata 
#both accessed by indices

N_exams = 38542
N_windows = 150
N_channels = 17
Window_len = 6 #s
Fs = 128 #Hz
dtype = np.float32