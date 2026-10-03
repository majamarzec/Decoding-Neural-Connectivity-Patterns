"""
Digital windowed signal processing module:

(1) cut the continuous signal in non-overlapping fixed-length windows
(2) apply criterion of flat & too artifacted windows drop (+ register the drops)
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Montage name 'standard_1020' is deprecated",
    category=FutureWarning)

import config as c

import pandas as pd
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

def window_continuous_eeg(eeg: mne.io.Raw, 
                          info: dict, 
                          window_len: float = c.DEFAULT_WINDOW_LEN,
                          max_ptp: float = c.BIG_ARTIFACT_PTP,
                          flat_ptp: float = c.FLAT_PTP,
                          windows_limit: int = c.TARGET_WINDOWS) -> Optional[np.ndarray]:

    #create events of fixed length 
    windows_events = mne.make_fixed_length_events(eeg,
                                                  duration = window_len,
                                                  overlap = 0)
    # instantiate epochs with PTP amplitude rejection criterion
    windows = mne.Epochs(eeg,
                         tmin = 0,
                         tmax = window_len - 1/c.DEFAULT_SFREQ, #inclusive so it produces 1 samle too much
                         baseline = None,
                         events = windows_events)

    windows.drop_bad(reject = dict(eeg = max_ptp), #type:ignore
                     flat = dict(eeg = flat_ptp),
                     verbose = False) #type:ignore
    
    drop_info = pd.Series([reason for log in windows.drop_log for reason in log]).value_counts().to_dict()

    n_good_windows = len(windows)
    info["n_good_windows"] =  n_good_windows
    info["why_bad_windows"] = drop_info

    if n_good_windows < windows_limit:
        info["status"] = "too short"
        return None
    
    windows = windows[:windows_limit]
    data = windows.get_data() #array of shape (n_epochs, n_channels, n_times)
    assert data.shape == (c.TARGET_WINDOWS, len(c.VALID_CHANNELS_AFTER_DROP),  c.DEFAULT_WINDOW_LEN * c.DEFAULT_SFREQ), f"{data.shape} != expected"
    return data

