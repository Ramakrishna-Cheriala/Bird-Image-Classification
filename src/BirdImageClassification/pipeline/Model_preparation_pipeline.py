from src.BirdImageClassification.components.Model_preparation import ModelPreparation
from src.BirdImageClassification.config.configuration import ConfigurationManager


class ModelPreparationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_preparation_config = config.base_model_config()
        model_preparation = ModelPreparation(config=model_preparation_config)
        # model_preparation.get_prepared_model()
        model_preparation.prepare_base_model()
