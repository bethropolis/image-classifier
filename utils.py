import json
import logging
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
BACKBONE_CHOICES = ("custom", "custom_v2", "mobilenetv2", "efficientnetb0", "resnet50")


class CleanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            return f"[{record.levelname}] {message}"
        return message


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(CleanFormatter())
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )


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


def count_images_per_class(directory: Path, class_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls in class_names:
        cls_path = directory / cls
        cls_count = 0
        if cls_path.is_dir():
            for fpath in cls_path.iterdir():
                if fpath.is_file() and fpath.suffix.lower() in IMAGE_EXTENSIONS:
                    cls_count += 1
        counts[cls] = cls_count
    return counts


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


def _augmentation_block(
    *,
    flip: bool,
    rotation: float,
    zoom: float,
    brightness: float,
    contrast: float,
) -> keras.Sequential | None:
    aug_layers: list[layers.Layer] = []
    if flip:
        aug_layers.append(layers.RandomFlip("horizontal"))
    if rotation > 0:
        aug_layers.append(layers.RandomRotation(rotation))
    if zoom > 0:
        aug_layers.append(layers.RandomZoom(zoom))
    if brightness > 0:
        aug_layers.append(layers.RandomBrightness(brightness, value_range=(0.0, 255.0)))
    if contrast > 0:
        aug_layers.append(layers.RandomContrast(contrast))

    if not aug_layers:
        return None

    return keras.Sequential(aug_layers, name="data_augmentation")


def _build_custom_model(
    num_classes: int,
    img_size: tuple[int, int],
    augment: bool,
    augmentation: keras.Sequential | None,
    variant: str,
) -> keras.Model:

    inputs = keras.Input(shape=(*img_size, 3))
    x = inputs
    if augment and augmentation is not None:
        x = augmentation(x)
    x = layers.Rescaling(1.0 / 255.0)(x)

    if variant == "v1":
        x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    elif variant == "v2":
        x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
    else:
        raise ValueError(f"Unsupported custom variant: {variant}")
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def _build_transfer_model(
    num_classes: int,
    img_size: tuple[int, int],
    augment: bool,
    augmentation: keras.Sequential | None,
    backbone: str,
    pretrained_weights: str | None,
    freeze_backbone: bool,
) -> keras.Model:
    if backbone == "mobilenetv2":
        backbone_ctor = keras.applications.MobileNetV2
        preprocess_input = keras.applications.mobilenet_v2.preprocess_input
    elif backbone == "efficientnetb0":
        backbone_ctor = keras.applications.EfficientNetB0
        preprocess_input = keras.applications.efficientnet.preprocess_input
    elif backbone == "resnet50":
        backbone_ctor = keras.applications.ResNet50
        preprocess_input = keras.applications.resnet.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base = backbone_ctor(
        input_shape=(*img_size, 3),
        include_top=False,
        weights=pretrained_weights,
    )
    base.trainable = not freeze_backbone

    inputs = keras.Input(shape=(*img_size, 3))
    x = inputs
    if augment and augmentation is not None:
        x = augmentation(x)
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
    *,
    aug_flip: bool = True,
    aug_rotation: float = 0.1,
    aug_zoom: float = 0.1,
    aug_brightness: float = 0.15,
    aug_contrast: float = 0.1,
) -> keras.Model:
    if backbone not in BACKBONE_CHOICES:
        choices = ", ".join(BACKBONE_CHOICES)
        raise ValueError(f"Invalid backbone '{backbone}'. Expected one of: {choices}")

    augmentation = (
        _augmentation_block(
            flip=aug_flip,
            rotation=aug_rotation,
            zoom=aug_zoom,
            brightness=aug_brightness,
            contrast=aug_contrast,
        )
        if augment
        else None
    )

    if backbone in ("custom", "custom_v2"):
        variant = "v2" if backbone == "custom_v2" else "v1"
        return _build_custom_model(
            num_classes=num_classes,
            img_size=img_size,
            augment=augment,
            augmentation=augmentation,
            variant=variant,
        )

    return _build_transfer_model(
        num_classes=num_classes,
        img_size=img_size,
        augment=augment,
        augmentation=augmentation,
        backbone=backbone,
        pretrained_weights=pretrained_weights,
        freeze_backbone=freeze_backbone,
    )


def load_leaderboard(models_dir: Path) -> list[dict]:
    entries: list[dict] = []
    if not models_dir.exists():
        return entries

    for run_dir in sorted(models_dir.glob("run-*")):
        if not run_dir.is_dir():
            continue
        report_path = run_dir / "evaluation_report.json"
        if not report_path.exists():
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload.setdefault("run_name", run_dir.name)
            payload.setdefault("run_dir", str(run_dir))
            entries.append(payload)
        except Exception:  # noqa: BLE001
            continue

    return sorted(entries, key=lambda e: float(e.get("test_accuracy", 0.0)), reverse=True)
