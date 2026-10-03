"""
The script executing the whole processing + saving pipeline.
"""


import logging
from pipeline_module import run_pipeline

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [logging.FileHandler("batch_preprocessing.log"),
                                logging.StreamHandler()],
                    force = True)
logger.info("Starting processing.")
run_pipeline()
    