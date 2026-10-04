"""
Digital continous signal processing module:
(1) read metadata once and then row-by-row
(2) load eeg raw signals from EDF files mapped from already preprocessed metadata
(3) store each eeg as a braindecode.datasets.RawDataset with desc aligned with original metadata file exams.csv and specified target_name
(4) process each continous eeg & store informations about initial properties (fs, lenght, channels) in info dict
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Montage name 'standard_1020' is deprecated",
    category=FutureWarning)

import config as c 

import json
import pandas as pd
from typing import Tuple, Optional
import mne

from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, buttord
from braindecode.datasets import RawDataset


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [logging.FileHandler("batch_preprocessing.log"),
                                logging.StreamHandler()],
                    force = True)

mne.set_log_level("WARNING")


def _load_raw_from_row(row: dict) -> Tuple[Optional[RawDataset], dict]:
    """
    Load and wrap one eeg recording from edf file, by metadata row index.
    Add metadata to be in the mne object for easier clustering by features.
    """
    exam_id = row["exam_id"]
    path = c.EDF_DIR + f"/{exam_id}.edf"
    raw  = read_raw_edf(path,
                        infer_types = True,
                        verbose = False)

    all_channels = list(raw.ch_names)
    info = {"site": row["site"],
            "exam_id": exam_id,
            "initial_fs": float(raw.info["sfreq"]),
            "n_all_windows": int((raw.n_times / raw.info["sfreq"]) // c.DEFAULT_WINDOW_LEN)}
    
    raw.pick(picks=["eeg"])
    info["ch_non_eeg"] = json.dumps(sorted(set(all_channels) - set(raw.ch_names)))

    if not raw.ch_names:
        info["status"] = "no_eeg" 
        logger.warning("Skipping %s before loading data: No eeg channels found.", exam_id)
        return None, info

    if info["n_all_windows"] < c.TARGET_WINDOWS:
        info["status"] = "too_short_raw"
        logger.warning("Skipping %s before loading data: The recording is too short.", exam_id)
        return None, info

    
    info["status"] = "ok"
    raw.load_data() 
    age_bins = list(range(16, 67, 10))
    age_labels = [f"{i}-{i+9}" for i in range(16, 57, 10)]
    age_group = str(pd.cut([row["age"]], bins=age_bins, labels=age_labels, right=False)[0])
    
    return RawDataset(raw=raw, 
                      description={"site": row["site"],
                                   "patient_id": row["patient_id"],
                                   "exam_id": exam_id,
                                   "age": row["age"],
                                   "age_group": age_group,
                                   "female": row["female"],
                                   "pathology": row["pathology"]},
                      target_name = "pathology"), info


def apply_specified_filters(data: RawDataset) -> None:
    raw = data.raw
    sfreq = raw.info["sfreq"] #varies across recordings
    
    b, a = iirnotch(c.DEFAULT_NOTCH_FREQ, c.DEFAULT_NOTCH_Q, fs=sfreq)
    raw.apply_function(lambda d: filtfilt(b, a, d, axis=-1), verbose=False) #labda is a wrapped to match filtfilt output with apply_function demaned input
    
    wp_hp = c.DEFAULT_HP_CUTOFF
    ws_hp = wp_hp * 0.5
    N_hp, Wn_hp = buttord(wp=wp_hp, ws=ws_hp, gpass=c.DEFAULT_GPASS, gstop=c.DEFAULT_GSTOP, fs=sfreq)
    sos_hp = butter(min(N_hp, c.DEFAULT_MAX_ORDER), Wn_hp, btype="highpass", fs=sfreq, output="sos")
    raw.apply_function(lambda d: sosfiltfilt(sos_hp, d, axis=-1), verbose=False)
    
    wp_lp = c.DEFAULT_LP_CUTOFF
    ws_lp = wp_lp + (wp_lp * 0.25)
    N_lp, Wn_lp = buttord(wp=wp_lp, ws=ws_lp, gpass=c.DEFAULT_GPASS, gstop=c.DEFAULT_GSTOP, fs=sfreq)
    sos_lp = butter(min(N_lp, c.DEFAULT_MAX_ORDER), Wn_lp, btype="lowpass", fs=sfreq, output="sos")
    raw.apply_function(lambda d: sosfiltfilt(sos_lp, d, axis=-1), verbose=False)

def apply_resample(data: RawDataset) -> None:
    raw = data.raw
    raw.resample(float(c.DEFAULT_SFREQ))

def apply_rereference_and_drop(data: RawDataset) -> None:
    raw = data.raw
    raw.set_eeg_reference(ref_channels = c.LINKED_TEMPORAL) #type:ignore
    raw.drop_channels(ch_names = c.LINKED_TEMPORAL)


def apply_channels_rename(data: RawDataset) -> None:
    raw = data.raw
    for mapping in c.CHNAMES_MAPPING:
                if set(mapping.keys()).issubset(set(raw.ch_names)):
                    raw.rename_channels(mapping)

def apply_channels_cleaning(data: RawDataset, info: dict) -> bool:
    raw = data.raw
    keep = set(c.VALID_CHANNELS_AFTER_DROP) | set(c.LINKED_TEMPORAL)
    to_drop = [ch for ch in raw.ch_names if ch not in keep]
    raw.drop_channels(to_drop)
    info["ch_drop"] = json.dumps(to_drop)
    missing = sorted(keep - set(raw.ch_names))
    if missing:
        info["ch_missing"] = json.dumps(missing)
        info["status"] = "missing_channels"
        logger.warning("Skipping %s: missing channels %s", info["exam_id"], missing)
        return False
    return True

def apply_montage(data: RawDataset):
    raw = data.raw
    montage = make_standard_montage("colin27_1020")
    raw.set_montage(montage, 
                    match_case=False, 
                    on_missing='ignore')

# for each recording
def load_and_process_continuous_eeg(row_from_original_metadata: dict) -> Tuple[Optional[RawDataset], dict]:
    signal, technical_info = _load_raw_from_row(row_from_original_metadata)
    if signal is None:
        return None, technical_info
    apply_channels_rename(signal)
    if not apply_channels_cleaning(signal, technical_info):
            return None, technical_info

    apply_specified_filters(signal)
    apply_resample(signal)
    apply_montage(signal)
    apply_rereference_and_drop(signal)
    if signal is not None:
        signal.raw.reorder_channels(c.VALID_CHANNELS_AFTER_DROP)

    return signal, technical_info

