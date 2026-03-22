from pathlib import Path

from PIL import ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Some public medical datasets contain partially truncated JPGs.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_data_generators(dataset_dir):
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    test_dir = dataset_path / "test"

    # Training generator includes augmentation to help the model generalize.
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        zoom_range=0.2,
    )

    # Validation and test images should only be rescaled.
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=True,
    )

    val_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    dataset_folder = Path(__file__).resolve().parents[1] / "dataset"
    train_data, val_data, test_data = create_data_generators(dataset_folder)

    print("Class labels:", train_data.class_indices)
    print("Training samples:", train_data.samples)
    print("Validation samples:", val_data.samples)
    print("Test samples:", test_data.samples)
