import config as c 
from pandas import DataFrame, read_csv

def read_metadata(metadata_path: str = c.META_EVENTS_PATH) -> DataFrame:
    """
    Load preprocessed exam metadata once (then access row by row).
    """
    return read_csv(metadata_path)

def append_good_eeg_metadata(eeg_description: dict, 
                            exams_processed: str = c.META_EVENTS_PROCESSED_PATH) -> None:
    DataFrame([eeg_description]).to_csv(exams_processed, mode="a", header = False, index=False)


def append_technical_info_metadata(info: dict, 
                                   exams_processed_technical_info : str = c.META_EVENTS_PROCESSED_TECH_PATH) -> None:
    DataFrame([info]).to_csv(exams_processed_technical_info, mode="a", header = False, index=False)