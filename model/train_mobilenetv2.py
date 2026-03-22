from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from preprocess_data import IMAGE_SIZE, create_data_generators


def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )

    # Freeze pretrained feature-extraction layers for the first training stage.
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_history(history, output_dir):
    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png")
    plt.close()


if __name__ == "__main__":
    dataset_folder = Path(__file__).resolve().parents[1] / "dataset"
    train_generator, val_generator, _ = create_data_generators(dataset_folder)
    model_dir = Path(__file__).resolve().parent

    model = build_model()
    model.summary()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
    )

    plot_training_history(history, model_dir)
    model.save(model_dir / "bone_fracture_model.h5")
    print("Model and training graph saved successfully.")
