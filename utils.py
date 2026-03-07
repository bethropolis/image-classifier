from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
BACKBONE_CHOICES = ("custom", "mobilenetv2", "efficientnetb0")


def discover_classes(directory: Path) -> list[str]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}. Create class folders in this location."
        )

    class_names = sorted([d.name for d in directory.iterdir() if d.is_dir()])
    if not class_names:
        raise ValueError(
            f"No class folders found in {directory}. "
            "Add at least one class sub-folder with images."
        )
    return class_names


def has_class_directories(directory: Path) -> bool:
    return directory.exists() and any(child.is_dir() for child in directory.iterdir())


def ensure_class_directories(directory: Path, class_names: list[str]) -> None:
    missing = [cls for cls in class_names if not (directory / cls).is_dir()]
    if missing:
        missing_csv = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing class folders in {directory}: {missing_csv}. "
            "Folder names must match training classes."
        )


def count_images(directory: Path, class_names: list[str]) -> int:
    count = 0
    for cls in class_names:
        cls_path = directory / cls
        if not cls_path.is_dir():
            continue
        for fpath in cls_path.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
    return count


def create_image_dataset(
    directory: Path,
    class_names: list[str],
    img_size: tuple[int, int],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int | None,
    validation_split: float | None = None,
    subset: str | None = None,
    cache_path: Path | None = None,
) -> tf.data.Dataset:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = dataset.cache(str(cache_path))

    return dataset.prefetch(AUTOTUNE)


def load_dataset(
    directory: Path,
    class_names: list[str],
    img_size: tuple[int, int],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    cache_path: Path | None = None,
) -> tf.data.Dataset:
    return create_image_dataset(
        directory=directory,
        class_names=class_names,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        cache_path=cache_path,
    )


def save_class_names(path: Path, class_names: list[str]) -> None:
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing class names file: {path}. Run training first to generate model artifacts."
        )

    class_names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise ValueError(f"No class names found in {path}.")
    return class_names


def _augmentation_block() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.15, value_range=(0.0, 255.0)),
        ],
        name="data_augmentation",
    )


def _build_custom_model(num_classes: int, img_size: tuple[int, int], augment: bool) -> keras.Model:
    data_augmentation = _augmentation_block()

    inputs = keras.Input(shape=(*img_size, 3))
    x = layers.Rescaling(1.0)(inputs)
    if augment:
        x = data_augmentation(x)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def _build_transfer_model(
    num_classes: int,
    img_size: tuple[int, int],
    augment: bool,
    backbone: str,
    pretrained_weights: str | None,
    freeze_backbone: bool,
) -> keras.Model:
    data_augmentation = _augmentation_block()

    if backbone == "mobilenetv2":
        backbone_ctor = keras.applications.MobileNetV2
        preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    elif backbone == "efficientnetb0":
        backbone_ctor = keras.applications.EfficientNetB0
        preprocess_input = keras.applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base = backbone_ctor(
        input_shape=(*img_size, 3),
        include_top=False,
        weights=pretrained_weights,
    )
    base.trainable = not freeze_backbone

    inputs = keras.Input(shape=(*img_size, 3))
    x = layers.Rescaling(1.0)(inputs)
    if augment:
        x = data_augmentation(x)
    x = preprocess_input(x)
    x = base(x, training=not freeze_backbone)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def build_model(
    num_classes: int,
    img_size: tuple[int, int],
    augment: bool = True,
    backbone: str = "custom",
    pretrained_weights: str | None = "imagenet",
    freeze_backbone: bool = True,
) -> keras.Model:
    if backbone not in BACKBONE_CHOICES:
        choices = ", ".join(BACKBONE_CHOICES)
        raise ValueError(f"Invalid backbone '{backbone}'. Expected one of: {choices}")

    if backbone == "custom":
        return _build_custom_model(num_classes=num_classes, img_size=img_size, augment=augment)

    return _build_transfer_model(
        num_classes=num_classes,
        img_size=img_size,
        augment=augment,
        backbone=backbone,
        pretrained_weights=pretrained_weights,
        freeze_backbone=freeze_backbone,
    )
