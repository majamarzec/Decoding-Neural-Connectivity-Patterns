"""
The main idea is to use a one big np.memmap for all preprocessed eegs in original time-series windowed & preprocessed representation.
This then enables PyTorch to read DIRECTLY FORM DISC and not load the heavy arrays/ tensors into RAM.

There are reported problems with n_morkers in DataLoaders, but it can be resolved by giving them separate views of the memmap.
The downside it also zero compression, just serialization. 

For the sake of methodological tidiness, I resigned from parallelizing during processing.
This is not optimized in terms of computational resorces, but is easily tracable for neuroscientist.

The explicit dimension of the memmap is exam.
(N_exams, N_windows = 150, N_channels = 17, N_timepoints = 6s x 128Hz)

The memmap can't be appended to so the plan is to:
(1) prealocate the mm with N_exams = all exams prior to preprocessing // during preprocessing a lot of exams are discarded due to issues described in dc/wsp modules
(2) write to it sequentially exam by exam (with removing the ones that outputted None in fsp)
(3) truncate the mm at the end
"""

import numpy as np
import mne
import os
 
import config as c
from fsp_module import process
from metadata_module import read_metadata

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [logging.FileHandler("batch_preprocessing.log"),
                                logging.StreamHandler()])
mne.set_log_level("WARNING")


def run_pipeline():
    metadata = read_metadata()
    n_exams_before_preprocessing = metadata.shape[0]
    processed_eeg_path = os.path.join(c.PREPROCESSED_EEG_DIR, "eeg.dat")

    #initial allocation
    mm = np.memmap(filename = processed_eeg_path,
                   dtype = np.float32, #reducing precision because of memory issues
                   mode = 'w+', 
                   shape = (n_exams_before_preprocessing, c.TARGET_WINDOWS, len(c.VALID_CHANNELS_AFTER_DROP),  int(c.DEFAULT_WINDOW_LEN * c.DEFAULT_SFREQ)))
    
    bytes_per_exam = mm[0].nbytes
    counter = 0

    try:
        for row in metadata.to_dict(orient = "records"): #‘records’ : list like [{column -> value}, … , {column -> value}]
            processed_eeg = process(row)
            if processed_eeg is None:
                logger.info("Exam at metadata row %d discarded", counter)
                continue
            mm[counter] = processed_eeg #index has to be counted dynamically due to disarded exams
            counter+=1 #incrementation
    finally:
        mm.flush()
        del mm

    n_valid = counter

    if n_valid == 0:
        os.remove(processed_eeg_path)
        raise RuntimeError("No exams survived preprocessing.")

    os.truncate(path = processed_eeg_path,
                length = n_valid * bytes_per_exam) #len in bytes
    
    assert os.path.getsize(processed_eeg_path) == n_valid * bytes_per_exam


if __name__ == "__main__":
    run_pipeline()


    



    
