from src.BirdImageClassification import logger
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.BirdImageClassification import logger
from src.BirdImageClassification.entity.config import TrainingModelConfig
from src.BirdImageClassification.utils.common import (
    read_yaml,
    create_directory,
    save_json,
)


class TrainingModel:
    def __init__(self, config=TrainingModelConfig):
        self.config = config

    def load_model(self):
        self.model = tf.keras.models.load_model(self.config.saved_model_dir)

    def prepare_training_model(self):
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            self.config.train_data,
            target_size=(self.config.input_shape, self.config.input_shape),
            batch_size=self.config.batch_size,
            class_mode="categorical",
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.config.validation_data,
            target_size=(self.config.input_shape, self.config.input_shape),
            batch_size=self.config.batch_size,
            class_mode="categorical",
        )

        test_generator = test_datagen.flow_from_directory(
            self.config.test_data,
            target_size=(self.config.input_shape, self.config.input_shape),
            batch_size=self.config.batch_size,
            class_mode="categorical",
        )

        self.model.fit(
            train_generator,
            epochs=self.config.epochs,
            #    steps_per_epoch = len(train_generator),
            validation_data=validation_generator,
            #    validation_steps = len(validation_generator) * 0.25,
            verbose=1,
            callbacks=[early_stopping],
        )

        scores = self.model.evaluate(
            validation_generator, batch_size=self.config.batch_size
        )

        score = {"loss": scores[0], "accuracy": scores[1]}
        save_json(path=Path("scores.json"), data=score)

        self.save_model(path=self.config.trained_model_dir, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
