import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, VGG16, DenseNet201
from tensorflow.keras.layers import (
    Conv2D,
    Add,
    MaxPooling2D,
    Dense,
    BatchNormalization,
    Input,
    Flatten,
    Dropout,
    GlobalMaxPooling2D,
    Lambda,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.BirdImageClassification.utils.common import (
    read_yaml,
    create_directory,
    save_json,
)

from src.BirdImageClassification.entity.config import ModelPreparationConfig
from pathlib import Path
from src.BirdImageClassification import logger


class ModelPreparation:
    def __init__(self, config: ModelPreparationConfig):
        self.config = config
        print(self.config.input_shape)
        self.base_model = InceptionV3(
            weights=self.config.weights,
            include_top=self.config.include_top,
            input_shape=(self.config.input_shape, self.config.input_shape, 3),
        )

    def prepare_base_model(self):
        logger.info("Preparing base model................")
        model = Sequential()
        model.add(self.base_model)

        for layer in self.base_model.layers:
            layer.trainable = False

        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(self.config.classes, activation="softmax"))

        # lr = 0.005

        optimizers = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        model.compile(
            loss="categorical_crossentropy", optimizer=optimizers, metrics=["accuracy"]
        )

        model.summary()

        logger.info("Model saving................................")
        self.save_model(path=self.config.saved_model_dir, model=model)

    def save_model(self, path: Path, model=tf.keras.Model):
        model.save(path)
        logger.info("Model Saved.............................")
