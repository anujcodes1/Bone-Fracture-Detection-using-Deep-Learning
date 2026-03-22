from pathlib import Path
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from preprocess_data import IMAGE_SIZE


MODEL_PATH = Path(__file__).resolve().parent / "bone_fracture_model.h5"


def preprocess_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_fracture(image_path):
    model = load_model(MODEL_PATH)
    processed_image = preprocess_image(image_path)

    prediction = float(model.predict(processed_image, verbose=0)[0][0])

    if prediction >= 0.5:
        result = "Fracture"
        confidence = prediction * 100
    else:
        result = "Normal"
        confidence = (1 - prediction) * 100

    return result, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_single_image.py <image_path>")
        sys.exit(1)

    image_file = Path(sys.argv[1])
    result, confidence = predict_fracture(image_file)

    print(f"Result: {result}")
    print(f"Confidence: {confidence:.2f}%")
