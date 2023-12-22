from src.BirdImageClassification.config.configuration import ConfigurationManager
from src.BirdImageClassification.pipeline.Data_ingestion_pipeline import (
    DataIngestionPipeline,
)
from src.BirdImageClassification.pipeline.Model_preparation_pipeline import (
    ModelPreparationPipeline,
)
from src.BirdImageClassification.pipeline.Training_pipeline import TrainingPipeline

from src.BirdImageClassification import logger


try:
    STAGE_NAME = "Data Ingestion"
    logger.info(f"{STAGE_NAME} started")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f"{STAGE_NAME} finished")

    STAGE_NAME = "Model Preparation"
    logger.info(f"{STAGE_NAME} started")
    obj1 = ModelPreparationPipeline()
    obj1.main()
    logger.info(f"{STAGE_NAME} finished")

    STAGE_NAME = "Model Training"
    logger.info(f"{STAGE_NAME} started")
    obj2 = TrainingPipeline()
    obj2.main()
    logger.info(f"{STAGE_NAME} finished")
except Exception as e:
    logger.exception(e)
    raise e
