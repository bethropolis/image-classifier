import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from utils import (
    BACKBONE_CHOICES,
    build_model,
    count_images,
    create_image_dataset,
    discover_classes,
    ensure_class_directories,
    has_class_directories,
    save_class_names,
)

LOGGER = logging.getLogger(__name__)

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "classifier.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
SEED = 42
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 2
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6
COSINE_ALPHA = 0.01


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--class-names-path", type=Path, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=list(IMG_SIZE))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-split", type=float, default=VALIDATION_SPLIT)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    parser.add_argument("--lr-patience", type=int, default=LR_PATIENCE)
    parser.add_argument("--lr-factor", type=float, default=LR_FACTOR)
    parser.add_argument("--min-learning-rate", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--cosine-alpha", type=float, default=COSINE_ALPHA)
    parser.add_argument("--backbone", choices=list(BACKBONE_CHOICES), default="custom")
    parser.add_argument("--pretrained", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def build_optimizer(args: argparse.Namespace, train_ds: tf.data.Dataset) -> keras.optimizers.Optimizer:
    if args.lr_scheduler != "cosine":
        return keras.optimizers.Adam(learning_rate=args.learning_rate)

    cardinality_value = int(tf.data.experimental.cardinality(train_ds).numpy())
    steps_per_epoch = cardinality_value if cardinality_value > 0 else 1
    decay_steps = max(1, steps_per_epoch * args.epochs)
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=decay_steps,
        alpha=args.cosine_alpha,
    )
    return keras.optimizers.Adam(learning_rate=schedule)


def build_callbacks(args: argparse.Namespace, model_path: Path) -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    if args.lr_scheduler == "plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=args.lr_factor,
                patience=args.lr_patience,
                min_lr=args.min_learning_rate,
            )
        )

    return callbacks


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    img_size = (args.img_size[0], args.img_size[1])
    train_dir: Path = args.train_dir
    val_dir: Path = args.val_dir
    model_dir: Path = args.model_dir
    model_path: Path = args.model_path or (model_dir / MODEL_PATH.name)
    class_names_path: Path = args.class_names_path or (model_dir / CLASS_NAMES_PATH.name)

    tf.keras.utils.set_random_seed(args.seed)

    class_names = discover_classes(train_dir)
    LOGGER.info("Found %d classes: %s", len(class_names), class_names)

    if has_class_directories(val_dir):
        ensure_class_directories(val_dir, class_names)
        train_ds = create_image_dataset(
            train_dir,
            class_names,
            img_size,
            args.batch_size,
            shuffle=True,
            seed=args.seed,
            cache_path=model_dir / ".cache" / "train.cache",
        )
        val_ds = create_image_dataset(
            val_dir,
            class_names,
            img_size,
            args.batch_size,
            shuffle=False,
            seed=args.seed,
            cache_path=model_dir / ".cache" / "val.cache",
        )
        LOGGER.info("Using dedicated validation set from %s", val_dir)
    else:
        train_ds = create_image_dataset(
            train_dir,
            class_names,
            img_size,
            args.batch_size,
            shuffle=True,
            seed=args.seed,
            validation_split=args.validation_split,
            subset="training",
            cache_path=model_dir / ".cache" / "train.cache",
        )
        val_ds = create_image_dataset(
            train_dir,
            class_names,
            img_size,
            args.batch_size,
            shuffle=True,
            seed=args.seed,
            validation_split=args.validation_split,
            subset="validation",
            cache_path=model_dir / ".cache" / "val.cache",
        )
        LOGGER.info(
            "No data/val class folders found. Using seeded split from train data "
            "(validation_split=%s, seed=%s).",
            args.validation_split,
            args.seed,
        )

    train_samples = count_images(train_dir, class_names)
    val_samples = count_images(val_dir, class_names) if has_class_directories(val_dir) else "split from train"
    LOGGER.info("Training samples: %s", train_samples)
    LOGGER.info("Validation samples: %s", val_samples)

    pretrained_weights = None if args.pretrained == "none" else "imagenet"
    freeze_backbone = not args.unfreeze_backbone
    model = build_model(
        len(class_names),
        img_size,
        augment=not args.disable_augmentation,
        backbone=args.backbone,
        pretrained_weights=pretrained_weights,
        freeze_backbone=freeze_backbone,
    )
    model.summary()

    optimizer = build_optimizer(args, train_ds)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    callbacks = build_callbacks(args, model_path)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Keeps disk artifact aligned with best weights restored by EarlyStopping.
    model.save(model_path)
    save_class_names(class_names_path, class_names)

    LOGGER.info("Best model saved to %s", model_path)
    LOGGER.info("Class names saved to %s", class_names_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    curves_path = model_dir / "training_curves.png"
    plt.savefig(curves_path)
    LOGGER.info("Training curves saved to %s", curves_path)
    plt.show()


if __name__ == "__main__":
    main()
