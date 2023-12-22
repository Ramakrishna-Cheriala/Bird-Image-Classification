from src.BirdImageClassification.components.Training import TrainingModel
from src.BirdImageClassification.config.configuration import ConfigurationManager


class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        training_config = config_manager.training_config()
        training = TrainingModel(config=training_config)
        training.load_model()
        # training.data_preparation()
        training.prepare_training_model()
