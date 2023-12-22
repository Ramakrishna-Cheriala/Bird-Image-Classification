from src.BirdImageClassification.constants import *
from src.BirdImageClassification.utils.common import read_yaml, create_directory
from src.BirdImageClassification.entity.config import (
    DataIngestionConfig,
    TrainingModelConfig,
    ModelPreparationConfig,
)


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directory([self.config.main_dir])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directory([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            kaggle_api=config.kaggle_api_key,
            local_files_dir=config.local_files_dir,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def base_model_config(self) -> ModelPreparationConfig:
        config = self.config.model_preparation
        create_directory([config.root_dir])

        model_preparation_config = ModelPreparationConfig(
            root_dir=Path(config.root_dir),
            saved_model_dir=Path(config.model_saved_dir),
            input_shape=self.params.INPUT_SHAPE,
            include_top=self.params.INCLUDE_TOP,
            weights=self.params.WEIGHTS,
            classes=self.params.CLASSES,
            learning_rate=self.params.LEARNING_RATE,
        )

        return model_preparation_config

    def training_config(self) -> TrainingModelConfig:
        config = self.config.training_model
        create_directory([config.root_dir])

        prepare_training_model = TrainingModelConfig(
            root_dir=Path(config.root_dir),
            trained_model_dir=Path(config.trained_model_dir),
            saved_model_dir=Path(config.saved_model_dir),
            train_data=Path(config.train_data),
            validation_data=Path(config.validation_data),
            test_data=Path(config.test_data),
            batch_size=self.params.BATCH_SIZE,
            epochs=self.params.EPOCHS,
            input_shape=self.params.INPUT_SHAPE,
        )

        return prepare_training_model
