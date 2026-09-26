import numpy as np
import os

## Here user can amend all preprocessing parameters ##

BASE_DIR = "/dmj/fizmed/mmarzec/licencjat_neuro" 
PROJECT_DIR = os.path.join(BASE_DIR, "DL-NS") 
BASE_CSV_PATH = os.path.join(BASE_DIR, "baza_elm19/ELM19_info.csv")
EDF_DIR = os.path.join(BASE_DIR, "baza_elm19/ELM19_edfs")
OUTPUT_PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "dataset")

# Models
MODELS_DIR = os.path.join(PROJECT_DIR, "models_files")


VALID_CHANNELS = {'Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2'}

# ============================================================================
# DEFAULT FILETRING PARAMETERS, SPECIFIC DESIGN PRIOR MVAR
# ============================================================================

DEFAULT_NOTCH_FREQ = 50.0
DEFAULT_NOTCH_Q = 5
DEFAULT_IIR_ORDER = 4
DEFAULT_HP_CUTOFF = 1.0
DEFAULT_LP_CUTOFF = 40.0
DEFAULT_GPASS = 1.0
DEFAULT_GSTOP = 20.0
DEFAULT_MAX_ORDER = 4

# ============================================================================
# DEFAULT OTHERS
# ============================================================================

DEFAULT_SFREQ = 128
DEFAUL_OUTPUT = "dataset"

# ============================================================================
# REFERENCE AND CORRESPONDING CHANNEL REMOVAL (PRIOR MVAR)
# ============================================================================

# symmetry in 10-20 system
LINKED_TEMPORAL= ["T5", "T6"]

# ============================================================================
# WINDOWING DETAILS
# ============================================================================

DEFAULT_WINDOW_LEN = 6.0 # seconds
FLAT_PTP = 1e-6 #flat, ptp criterion
BIG_ARTIFACT_PTP = 1000e-6 # very large, should exceed blinking or muscles artifacts, but not the huge ones (movement, electrode detachment, drifts), ptp criterion
MAX_WINDOWS_PER_REC = 150 # we assume that longer recordings come from sleep depriviation exams ("delta contaminated")
N_JOBS = -3 #leave 3 CPU cores free


