from pathlib import Path
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.models import Model, load_model

from preprocess_data import IMAGE_SIZE


MODEL_PATH = Path(__file__).resolve().parent / "bone_fracture_model.h5"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "gradcam"


def preprocess_image(image_path):
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, IMAGE_SIZE)
    normalized_image = resized_image.astype("float32") / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return original_image, input_image


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (Conv2D, DepthwiseConv2D)):
            return layer.name

        layer_output = getattr(layer, "output", None)
        layer_shape = getattr(layer_output, "shape", None)
        if layer_shape is not None and len(layer_shape) == 4:
            return layer.name

    # Fallback for MobileNetV2 transfer-learning models saved with Keras 3.
    for fallback_name in ("Conv_1", "out_relu", "block_16_project"):
        try:
            model.get_layer(fallback_name)
            return fallback_name
        except ValueError:
            continue

    raise ValueError("No convolutional layer found for Grad-CAM.")


def make_gradcam_heatmap(input_image, model, last_conv_layer_name):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_image)
        class_channel = predictions[:, 0]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) == 0.0:
        return np.zeros(heatmap.shape, dtype=np.float32)
    heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 1 - alpha, color_heatmap, alpha, 0)
    return overlay


def generate_gradcam(image_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(MODEL_PATH)
    original_image, input_image = preprocess_image(image_path)
    last_conv_layer_name = find_last_conv_layer(model)

    heatmap = make_gradcam_heatmap(input_image, model, last_conv_layer_name)
    overlay_image = overlay_heatmap(original_image, heatmap)

    output_path = OUTPUT_DIR / f"{Path(image_path).stem}_gradcam.jpg"
    cv2.imwrite(str(output_path), overlay_image)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_gradcam.py <image_path>")
        sys.exit(1)

    image_file = Path(sys.argv[1])
    saved_path = generate_gradcam(image_file)
    print(f"Grad-CAM image saved at: {saved_path}")
