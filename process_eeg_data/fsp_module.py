"""
Full signal processing module per recording
(1) dcsp -> signal in RawDataset, appended metadata: new main matadata (or None) + technical metadata (even for dropped)
(2) dwsp -> ndarray of shape (150, 17, 768) of ready to use windowed preprocessed eeg (or None)

"""
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Montage name 'standard_1020' is deprecated",
    category=FutureWarning)

from dcsp_module import load_and_process_continuous_eeg
from dwsp_module import window_continuous_eeg
from metadata_module import append_good_eeg_metadata, append_technical_info_metadata

import mne
import numpy as np
from typing import Optional

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [logging.FileHandler("batch_preprocessing.log"),
                                logging.StreamHandler()],
                    force = True)

mne.set_log_level("WARNING")



def process(row: dict) -> Optional[np.ndarray]:
    eeg, info = load_and_process_continuous_eeg(row)
    if eeg is None:
        logger.warning("Recording not loaded for preprocessing!")
        return None
    append_good_eeg_metadata(eeg.description) #bad signal excluded
    windowed_eeg_ndarray = window_continuous_eeg(eeg.raw, info)
    append_technical_info_metadata(info) #bad signal included
    return windowed_eeg_ndarray


    