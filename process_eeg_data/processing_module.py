import json
import numpy as np
from pathlib import Path
import zarr
# for serialization
from zarr.codecs import BloscCodec

from pandas import DataFrame, read_csv
from mne.io import BaseRaw, read_raw_edf
from mne.channels import make_standard_montage
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, buttord

#local config
import config as c
#utilize braindecode's wrappers for internal parallelization
from braindecode.datasets import RawDataset, BaseConcatDataset, WindowsDataset
from braindecode.preprocessing import Preprocessor, create_fixed_length_windows, preprocess

#for parallelization
from collections import Counter
from itertools import chain
from joblib import Parallel, delayed

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


BR = BaseRaw
DF = DataFrame
RD = RawDataset
BCD = BaseConcatDataset
WD = WindowsDataset

def _read_metadata(metadata_path: str = f"{c.BASE_DIR}/magisterka/metadata/metadata_exams.csv") -> DF:
    """
    Load exam metadata once.
    """
    return read_csv(metadata_path)


def _load_raw_eeg_from_metadata(metadata_df: DF, idx: int) -> RD:
    """
    Load and wrap one eeg recording from edf file, by metadata row index.
    Add metadata to be in the mne object for easier clustering by features.
    """
    exam_id = metadata_df["exam_id"].iloc[idx]
    clean_exam_id = f"exam-{idx:05d}"
    path = c.EDF_DIR + f"/{exam_id}.edf"
    raw  = read_raw_edf(path,
                        infer_types = True,
                        verbose = 0,
                        preload = True)

    raw.pick(picks=["eeg"])
    to_drop = [ch for ch in raw.ch_names if ch not in c.VALID_1020]
    
    if to_drop:
        raw.drop_channels(to_drop)

    if not raw.ch_names:
        print(f"Skipping {exam_id}: No valid 10-20 channels found.")
        return None  

    if_patho = metadata_df["pathology"].iloc[idx]
    site = metadata_df["site"].iloc[idx]
    age = metadata_df["age"].iloc[idx]
    if_female = metadata_df["female"].iloc[idx]
    sub_id = metadata_df["sub_id"].iloc[idx]

    return RawDataset(raw = raw, 
                      description={"exam_id": clean_exam_id, 
                                   "original_exam_id": exam_id,
                                   "sub_id": sub_id,
                                   "target": if_patho,
                                   "site": site, 
                                   "age": age, 
                                   "sex": if_female})


def load_raw_eegs_from_metadata(metadata_path: str, 
                                idxs: list | None = None, 
                                n_jobs: int = c.N_JOBS) -> BCD:
    """
    Read metadata, load row for given indices in parallel (default None = all).
    Out of all unpreprocess raws create a BaseConcatDataset.
    """
    metadata_df = _read_metadata(metadata_path)
    if idxs is None:
         idxs = list(range(len(metadata_df)))
    datasets = Parallel(n_jobs = n_jobs, 
                        backend = "loky", 
                        verbose = 5)(delayed(_load_raw_eeg_from_metadata)(metadata_df, idx) for idx in idxs)
    datasets = [d for d in datasets if d is not None]
    return BaseConcatDataset(datasets)


def rename_and_montage(raw: BR) -> BR:
    for mapping in c.CHNAMES_MAPPING:
                if set(mapping.keys()).issubset(set(raw.ch_names)):
                    raw.rename_channels(mapping)
    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, 
                    match_case=False, 
                    on_missing='ignore')
    return raw

def apply_specified_filters(raw: BR) -> BR:
    sfreq = raw.info["sfreq"] #varies across recordings
    
    b, a = iirnotch(c.DEFAULT_NOTCH_FREQ, c.DEFAULT_NOTCH_Q, fs=sfreq)
    raw.apply_function(lambda d: filtfilt(b, a, d, axis=-1), verbose=False)
    
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
    return raw

def apply_rereference_drop_and_resample(raw: BR) -> BR:
    raw.set_eeg_reference(ref_channels = c.LINKED_TEMPORAL)
    raw.drop_channels(ch_names = c.LINKED_TEMPORAL)
    raw.resample(float(c.DEFAULT_SFREQ))
    return raw

def _log_windows_info_and_standarize_recording_len(ds: RD, 
                               windows_limit: int = c.MAX_WINDOWS_PER_REC) -> tuple[RD, int, dict, bool]:
    """
    Dropping flats and too artifacted windows is handled internally by create_fixed_length_windows, but this function logs it to metadata.
    Crop recording length (in windows) to selected threshold (here 150 defined in config) + log if too short.
    """
    windows = ds.windows
    drop_counts = Counter(chain.from_iterable(windows.drop_log))
    n_good_windows = len(windows)
    windows.metadata.loc[:, "target"] = ds.description["target"]

    is_too_short = n_good_windows < windows_limit

    if is_too_short:
        ds.windows = None  #overwriting on purpose
    else:
        ds.windows = windows[:windows_limit] #overwriting on purpose
        ds.y = ds.windows.metadata["target"].to_numpy() #resync
    return ds, n_good_windows, dict(drop_counts), is_too_short

def log_windows_info_and_standarize_recordings_len(windows_dataset: BCD, 
                               windows_limit: int = c.MAX_WINDOWS_PER_REC,
                               n_jobs: int = c.N_JOBS,
                               stats_path: str = f"{c.BASE_DIR}/magisterka/metadata/windows_stats.csv") -> BCD:
    """
    Full dataset applicable. Parallelized across recodings.
    Sort the dataset by sites for easier chunks later on.
    """
    results = Parallel(n_jobs = n_jobs,
                              backend = "loky",
                              verbose = 5)(delayed(_log_windows_info_and_standarize_recording_len)(ds, windows_limit) for ds in windows_dataset.datasets)

    stats_df = DataFrame([
        {   "site": ds.description["site"],
            "target": ds.description["target"],
            "age": ds.description["age"],
            "exam_id": ds.description["exam_id"],
            "original_exam_id": ds.description["original_exam_id"],
            "good_windows_count": n_good,
            "bad_windows_reasons": drop_reasons,
            "too_short": too_short,
        }
        for ds, n_good, drop_reasons, too_short in results
    ])

    stats_df.to_csv(stats_path, index=False)
    logger.info("Wrote per-recording window stats to %s", stats_path)

    n_dropped = int(stats_df["too_short"].sum())
    if n_dropped:
        logger.info("Dropped %d/%d recordings as too short (< %d windows).",
                    n_dropped, len(results), windows_limit)
    
    valid_datasets = [ds for ds, _, _, too_short in results if not too_short]
    
    return BaseConcatDataset(valid_datasets)

###################
def build_preprocessed_windows_dataset(metadata_path: str = f"{c.BASE_DIR}/magisterka/metadata/metadata_exams.csv",
                                       idxs: list | None = None,
                                       n_jobs: int = c.N_JOBS) -> BCD:
    """
    Full preprocessing pipeline with Braindecode wrappers:
    Load raws from EDFs aligned with preprocessed metadata (metadata_exams.csv).
    Preprocess EEGs.
    Window.
    Reject and log bad windows.
    """

    logger.info("Loading raws from metadata: %s", metadata_path)
    concat_ds = load_raw_eegs_from_metadata(metadata_path, idxs = idxs, n_jobs = n_jobs)
    N = len(concat_ds.datasets)
    logger.info("Loaded %d recordings.", N)


    preprocessors = [
         Preprocessor(rename_and_montage, apply_on_array = False),
         Preprocessor(apply_specified_filters, apply_on_array = False),
         Preprocessor(apply_rereference_drop_and_resample, apply_on_array = False)
    ]

    logger.info("Preprocessing %d recordings: ", N)
    preprocess(concat_ds, preprocessors, n_jobs = n_jobs)
    logger.info("Preprocessing done.")

    window_size_in_samples = int(c.DEFAULT_WINDOW_LEN * c.DEFAULT_SFREQ)
    logger.info("Windowing into %d-sample windows.", window_size_in_samples)
    windows_ds = create_fixed_length_windows(
        concat_ds = concat_ds,
        window_size_samples = window_size_in_samples,
        window_stride_samples = None, #no overlap
        reject = dict(eeg = c.BIG_ARTIFACT_PTP),
        flat = dict(eeg = c.FLAT_PTP),
        drop_bad_windows= True,
        on_last_window = 'drop',
        use_mne_epochs = True,
        preload = True
    )
    N2 = len(windows_ds.datasets)
    logger.info("Windowing done: %d recordings.", N2)

    logger.info("Standarizing recordings length...")
    windows_ds = log_windows_info_and_standarize_recordings_len(windows_ds, 
                                                                n_jobs = n_jobs)
    N2 = len(windows_ds.datasets) #overwriting on purpose
    logger.info("Done: %d saved recordings of standarized length [windows]", N2)

    return windows_ds

#####
 
def _write_one_exam(data_array, 
                    i: int, 
                    ds) -> None:
    exam = ds.windows.get_data(copy=False)
    data_array[i] = exam.astype(np.float32, copy=False)

def save_site_in_chunks(site_ds,
                        site_name: str,
                        output_path: str = c.OUTPUT_PREPROCESSED_DIR,
                        N_exams_per_chunk: int = 25,
                        n_jobs: int = 1) -> list[dict]:
    """
    Save a site's preprocessed WindowsDatasets into exam-level zarr chunks locally on disk
    (not pushed to Hugging Face).
 
    With a dataset this large (e.g. ELM19, 50k+ EEGs), using braindecode's internal
    BIDS-format save on a BaseConcatDataset would be an overkill.
 
    Storage structure of the dataset:
        site-i/
            chunk-0000/
                data.zarr/     shape: (n_exams_in_chunk, n_windows, n_channels, n_samples)
                metadata.parquet   (exam-level rows of metadata)
                manifest.json
            chunk-0001/
                data.zarr/
                metadata.parquet
                manifest.json
            ...
 
    Each physical Zarr chunk stores N_exams_per_chunk exams; each exam is one contiguous (1, n_windows, n_channels, n_samples) within it.
    """

    site_path = Path(output_path) / site_name
    site_path.mkdir(parents=True, exist_ok=True)
 
    datasets = [ds for ds in site_ds.datasets if ds is not None and ds.windows is not None]
    N_exams = len(datasets)
 
    if N_exams == 0:
        logger.warning("No valid exams for site %s; nothing written.", site_name)
        return []
 
    first_exam = datasets[0].windows.get_data(copy=False)
    n_windows, n_channels, n_samples = first_exam.shape
 
    chunk_info = []
 
    for chunk_idx, start in enumerate(range(0, N_exams, N_exams_per_chunk)):
        end = min(start + N_exams_per_chunk, N_exams)
        chunk = datasets[start:end]
        N_chunk = len(chunk)
 
        chunk_path = site_path / f"chunk_{chunk_idx:04d}"
        chunk_path.mkdir(parents=True, exist_ok=True)
        zarr_path = chunk_path / "data.zarr"
 
        root = zarr.open(str(zarr_path), mode="w")
        root.create_array(
            "data",
            shape=(N_chunk, n_windows, n_channels, n_samples),
            dtype=np.float32,  # reduced from float64
            chunks=(1, n_windows, n_channels, n_samples),
            compressors=[
                BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"),
            ],
        )
        data_array = root["data"]
 
        if n_jobs == 1:
            for i, ds in enumerate(chunk):
                _write_one_exam(data_array, i, ds)
        else:
            Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_write_one_exam)(data_array, i, ds) for i, ds in enumerate(chunk))
 
        metadata = DataFrame([ds.description.to_dict() for ds in chunk])
        metadata.insert(0, "site_exam_idx", np.arange(start, end))
        metadata.to_parquet(chunk_path / "metadata.parquet")
 
        chunk_manifest = {
            "site": site_name,
            "chunk_idx": chunk_idx,
            "exam_start": start,
            "exam_end": end,
            "N_exams": N_chunk,
            "shape": [N_chunk, n_windows, n_channels, n_samples],
            "dtype": "float32",
        }
 
        with open(chunk_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(chunk_manifest, f, indent=2)
 
        chunk_info.append(chunk_manifest)
        logger.info("Wrote chunk %d for site %s: exams [%d:%d)", chunk_idx, site_name, start, end)
 
    return chunk_info