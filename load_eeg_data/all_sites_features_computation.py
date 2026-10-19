from lib import *
import numpy as np
import os


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [logging.FileHandler("ffDTF_extraction.log"),
                                logging.StreamHandler()])
mne.set_log_level("WARNING")

def run_pipeline():
    i_exams = np.arange(0, c.N_exams, 1)
    eeg = load_eeg_memmap()
    #initial allocation
    mm = np.memmap(filename = os.path.join(c.PREPROCESSED_CONN_DIR, "all_sites_ffDTF.dat"), 
                    dtype = np.float32,
                    mode = 'w+', 
                    shape = (len(i_exams), c.N_windows, c.N_channels, c.N_channels, len(c.freq_bands)))

    try:    
        for i_exam in i_exams:
            for j, window in enumerate(eeg[i_exam]):
                mm[i_exam, j, :, :, :] = feature_pipeline(window, feature_type = 'ffDTF', aggregation_method = np.sum)
            if (i_exam + 1) % 100 == 0: 
                logger.info("Processed %d/%d recordings", i_exam + 1, len(i_exams))

    finally:
        mm.flush()
        del mm

if __name__ == "__main__":
    run_pipeline()  