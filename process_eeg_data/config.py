import numpy as np
import os

## Here user can amend all preprocessing parameters ##

BASE_DIR = "/dmj/fizmed/mmarzec/licencjat_neuro" 
PROJECT_DIR = os.path.join(BASE_DIR, "DL-NS") 
BASE_CSV_PATH = os.path.join(BASE_DIR, "baza_elm19/ELM19_info.csv")
EDF_DIR = os.path.join(BASE_DIR, "baza_elm19/ELM19_edfs")
PREPROCESSED_EEG_DIR = os.path.join(PROJECT_DIR, "data/eeg")


# ============================================================================
# METADATA
# ============================================================================

META_EVENTS_PATH = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/meta/exams.csv"
META_EVENTS_PROCESSED_PATH = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/meta/exams_processed.csv"
META_EVENTS_PROCESSED_TECH_PATH = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/meta/exams_processed_technical_info.csv"
META_WINDOWS_PATH = "/dmj/fizmed/mmarzec/licencjat_neuro/DL-NS/data/meta/windows.csv"


# ============================================================================
# DEFAULT PREPROCESSING PARAMETERS, SPECIFIC DESIGN PRIOR MVAR
# ============================================================================

UNIT_SCALE = 1e-6
DEFAULT_NOTCH_FREQ = 50.0
DEFAULT_NOTCH_Q = 5
DEFAULT_IIR_ORDER = 4
DEFAULT_HP_CUTOFF = 1.0
DEFAULT_LP_CUTOFF = 40.0
DEFAULT_GPASS = 1.0
DEFAULT_GSTOP = 20.0
DEFAULT_MAX_ORDER = 4
DEFAULT_SFREQ = 128
LINKED_TEMPORAL= ["T5", "T6"]

# ============================================================================
# WINDOWING DETAILS
# ============================================================================

DEFAULT_WINDOW_LEN = 6.0 # seconds
FLAT_PTP = UNIT_SCALE #flat, ptp criterion
BIG_ARTIFACT_PTP = 1000 * UNIT_SCALE # very large, should exceed blinking or muscles artifacts, but not the huge ones (movement, electrode detachment, drifts), ptp criterion
TARGET_WINDOWS = 150 # we assume that longer recordings come from sleep depriviation exams ("delta contaminated")


###############
VALID_CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
VALID_CHANNELS_AFTER_DROP = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','P3','Pz','P4','O1','O2']

CHNAMES_MAPPING = [
    {
        "EEG Fp1": "Fp1",
        "EEG Fp2": "Fp2",
        "EEG F7": "F7",
        "EEG F3": "F3",
        "EEG Fz": "Fz",
        "EEG F4": "F4",
        "EEG F8": "F8",
        "EEG T3": "T3",
        "EEG C3": "C3",
        "EEG Cz": "Cz",
        "EEG C4": "C4",
        "EEG T4": "T4",
        "EEG T5": "T5",
        "EEG P3": "P3",
        "EEG Pz": "Pz",
        "EEG P4": "P4",
        "EEG T6": "T6",
        "EEG O1": "O1",
        "EEG O2": "O2",
    },
 {
        "EEG Fp1": "Fp1",
        "EEG Fp2": "Fp2",
        "EEG F7": "F7",
        "EEG F3": "F3",
        "Fz_nowe": "Fz",
        "EEG F4": "F4",
        "EEG F8": "F8",
        "EEG T3": "T3",
        "EEG C3": "C3",
        "EEG Cz": "Cz",
        "EEG C4": "C4",
        "EEG T4": "T4",
        "EEG T5": "T5",
        "EEG P3": "P3",
        "EEG Pz": "Pz",
        "EEG P4": "P4",
        "EEG T6": "T6",
        "EEG O1": "O1",
        "EEG O2": "O2",
    },
    {
        "EEG FP1-REF": "Fp1",
        "EEG FP2-REF": "Fp2",
        "EEG F3-REF": "F3",
        "EEG F4-REF": "F4",
        "EEG C3-REF": "C3",
        "EEG C4-REF": "C4",
        "EEG P3-REF": "P3",
        "EEG P4-REF": "P4",
        "EEG O1-REF": "O1",
        "EEG O2-REF": "O2",
        "EEG F7-REF": "F7",
        "EEG F8-REF": "F8",
        "EEG T3-REF": "T3",
        "EEG T4-REF": "T4",
        "EEG T5-REF": "T5",
        "EEG T6-REF": "T6",
        "EEG A1-REF": "A1",
        "EEG A2-REF": "A2",
        "EEG FZ-REF": "Fz",
        "EEG CZ-REF": "Cz",
        "EEG PZ-REF": "Pz",
    },
    {
        'C3-REF': "C3",
        'C4-REF': "C4",
        'Cz-REF': "Cz",
        'EKG-REF': "EKG",
        'F3-REF': "F3",
        'F4-REF': "F4",
        'F7-REF': "F7",
        'F8-REF': "F8",
        'Fp1-REF': "Fp1",
        'Fp2-REF': "Fp2",
        'Fz-REF': "Fz",
        'O1-REF': "O1",
        'O2-REF': "O2",
        'P3-REF': "P3",
        'P4-REF': "P4",
        'Photic-REF': "Photic",
        'Pz-REF': "Pz",
        'T3-REF': "T3",
        'T4-REF': "T4",
        'T5-REF': "T5",
        'T6-REF': "T6"
    },
]
