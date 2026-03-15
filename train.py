import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from utils import (
    BACKBONE_CHOICES,
    build_model,
    configure_logging,
    count_images,
    count_images_per_class,
    create_image_dataset,
    discover_classes,
    ensure_class_directories,
    has_class_directories,
    save_class_names,
)

LOGGER = logging.getLogger(__name__)

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
MODELS_DIR = Path("models")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--models-dir", "--model-dir", dest="models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
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
    parser.add_argument("--backbone", choices=list(BACKBONE_CHOICES), default="custom_v2")
    parser.add_argument("--pretrained", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--aug-flip", action="store_true", default=True)
    parser.add_argument("--no-aug-flip", dest="aug_flip", action="store_false")
    parser.add_argument("--aug-rotation", type=float, default=0.1)
    parser.add_argument("--aug-zoom", type=float, default=0.1)
    parser.add_argument("--aug-brightness", type=float, default=0.15)
    parser.add_argument("--aug-contrast", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--class-weights", choices=["none", "auto"], default="none")
    parser.add_argument("--no-plot", action="store_true", help="Save plots but do not open interactive windows")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def generate_run_name(models_dir: Path, backbone: str) -> str:
    existing = (
        [p.name for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
        if models_dir.exists()
        else []
    )
    next_n = len(existing) + 1
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run-{next_n:04d}-{now}-{backbone}"


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


def save_latest_pointer(models_dir: Path, payload: dict) -> None:
    latest_path = models_dir / "latest.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_history(path: Path, history: keras.callbacks.History) -> None:
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def plot_training_curves(
    history: keras.callbacks.History,
    output_path: Path,
    *,
    run_name: str,
    backbone: str,
    train_samples: int,
    val_samples: int | str,
) -> None:
    epochs = list(range(1, len(history.history.get("loss", [])) + 1))
    val_acc = history.history.get("val_accuracy", [])
    val_loss = history.history.get("val_loss", [])
    lr_history = history.history.get("learning_rate", [])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].plot(epochs, history.history.get("accuracy", []), label="Train", marker="o", linewidth=1.8)
    axes[0].plot(epochs, val_acc, label="Val", marker="o", linewidth=1.8)
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    if val_acc:
        best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
        axes[0].scatter(best_idx + 1, val_acc[best_idx], color="red", zorder=3)
        axes[0].annotate(
            f"best={val_acc[best_idx]:.3f}",
            (best_idx + 1, val_acc[best_idx]),
            textcoords="offset points",
            xytext=(6, 8),
        )

    axes[1].plot(epochs, history.history.get("loss", []), label="Train", marker="o", linewidth=1.8)
    axes[1].plot(epochs, val_loss, label="Val", marker="o", linewidth=1.8)
    axes[1].set_title("Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    if val_loss:
        best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
        axes[1].scatter(best_idx + 1, val_loss[best_idx], color="red", zorder=3)
        axes[1].annotate(
            f"min={val_loss[best_idx]:.3f}",
            (best_idx + 1, val_loss[best_idx]),
            textcoords="offset points",
            xytext=(6, 8),
        )

    if lr_history:
        axes[2].plot(epochs, lr_history, marker="o", linewidth=1.8)
        axes[2].set_yscale("log")
    axes[2].set_title("Learning Rate by Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        f"Run: {run_name} | Backbone: {backbone} | Train: {train_samples} | Val: {val_samples}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    img_size = (args.img_size[0], args.img_size[1])
    train_dir: Path = args.train_dir
    val_dir: Path = args.val_dir
    models_dir: Path = args.models_dir

    tf.keras.utils.set_random_seed(args.seed)

    run_name = args.run_name or generate_run_name(models_dir, args.backbone)
    run_dir = models_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.keras"
    class_names_path = run_dir / "class_names.txt"
    curves_path = run_dir / "training_curves.png"
    history_path = run_dir / "training_history.json"
    metadata_path = run_dir / "run_metadata.json"

    class_names = discover_classes(train_dir)
    LOGGER.info("Run %s", run_name)
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
            cache_path=run_dir / ".cache" / "train.cache",
        )
        val_ds = create_image_dataset(
            val_dir,
            class_names,
            img_size,
            args.batch_size,
            shuffle=False,
            seed=args.seed,
            cache_path=run_dir / ".cache" / "val.cache",
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
            cache_path=run_dir / ".cache" / "train.cache",
        )
        val_ds = create_image_dataset(
            train_dir,
            class_names,
            img_size,
            args.batch_size,
            shuffle=False,
            seed=args.seed,
            validation_split=args.validation_split,
            subset="validation",
            cache_path=run_dir / ".cache" / "val.cache",
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
        aug_flip=args.aug_flip,
        aug_rotation=args.aug_rotation,
        aug_zoom=args.aug_zoom,
        aug_brightness=args.aug_brightness,
        aug_contrast=args.aug_contrast,
    )
    model.summary()

    optimizer = build_optimizer(args, train_ds)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"],
    )

    callbacks = build_callbacks(args, model_path)

    class_weights = None
    if args.class_weights == "auto":
        class_counts = count_images_per_class(train_dir, class_names)
        total = sum(class_counts.values())
        if total == 0:
            LOGGER.warning("No training images found for class-weight calculation. Skipping class weights.")
        else:
            class_weights = {}
            num_classes = len(class_names)
            for idx, cls in enumerate(class_names):
                cls_count = class_counts.get(cls, 0)
                if cls_count == 0:
                    LOGGER.warning("Class '%s' has 0 images; assigning weight 0.0", cls)
                    class_weights[idx] = 0.0
                else:
                    class_weights[idx] = total / (num_classes * cls_count)
            LOGGER.info("Using auto class weights: %s", class_weights)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    model.save(model_path)
    save_class_names(class_names_path, class_names)
    save_history(history_path, history)

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    best_val_loss = min(history.history.get("val_loss", [float("inf")]))

    metadata = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "class_names_path": str(class_names_path),
        "history_path": str(history_path),
        "curves_path": str(curves_path),
        "backbone": args.backbone,
        "image_size": list(img_size),
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "train_samples": train_samples,
        "val_samples": val_samples,
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_latest_pointer(models_dir, metadata)

    LOGGER.info("Best model saved to %s", model_path)
    LOGGER.info("Class names saved to %s", class_names_path)
    LOGGER.info("Run metadata saved to %s", metadata_path)
    LOGGER.info("Updated latest pointer: %s", models_dir / "latest.json")

    plot_training_curves(
        history,
        curves_path,
        run_name=run_name,
        backbone=args.backbone,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    if not args.no_plot:
        plt.show()
    else:
        plt.close("all")
    LOGGER.info("Training curves saved to %s", curves_path)


if __name__ == "__main__":
    main()
