import os
import urllib.request as request
import zipfile
from src.BirdImageClassification import logger
from kaggle.api.kaggle_api_extended import KaggleApi
from src.BirdImageClassification.utils.common import (
    read_yaml,
    create_directory,
    save_json,
)

from src.BirdImageClassification.entity.config import DataIngestionConfig

from pathlib import Path
from src.BirdImageClassification import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_files_dir):
            try:
                dataset_name = "gpiosenka/100-bird-species"
                KaggleApi().dataset_download_files(
                    dataset_name, path=self.config.local_files_dir, force=True
                )
                logger.info("File downloaded successfully from Kaggle.")
            except Exception as e:
                # logger.error(f"Error downloading file from Kaggle: {e}")
                raise e
        else:
            logger.info("File already exists.")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        # filepath = (
        #     r"C:\Users\ramak\OneDrive\Desktop\P2\Birds_Classification\Data\data.zip"
        # )
        filepath = os.path.join("Data", "data.zip")
        logger.info("Extracting zip files............")
        # with zipfile.ZipFile(self.config.local_files_dir, 'r') as zip_ref:
        #     zip_ref.extractall(unzip_path)
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info("Zip files extracted successfully......................")
