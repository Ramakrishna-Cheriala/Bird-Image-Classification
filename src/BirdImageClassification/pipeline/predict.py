import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        dir_train = os.path.join("artifacts", "data_ingestion", "data", "train/")
        classes = os.listdir(dir_train)
        model = load_model(os.path.join("artifacts", "training_model", "model.h5"))
        image_path = self.filename
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_classname = classes[predicted_class]
        max_score = np.max(prediction)

        return [{"image": predicted_classname, "Prediction_score": str(max_score)}]
