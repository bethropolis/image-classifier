import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from utils import create_image_dataset, ensure_class_directories, load_class_names

LOGGER = logging.getLogger(__name__)

TEST_DIR = Path("data/test")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "classifier.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image classifier")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--class-names-path", type=Path, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=list(IMG_SIZE))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    img_size = (args.img_size[0], args.img_size[1])
    test_dir: Path = args.test_dir
    model_dir: Path = args.model_dir
    model_path: Path = args.model_path or (model_dir / MODEL_PATH.name)
    class_names_path: Path = args.class_names_path or (model_dir / CLASS_NAMES_PATH.name)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model: {model_path}. Run `uv run train` first."
        )

    class_names = load_class_names(class_names_path)
    ensure_class_directories(test_dir, class_names)
    num_classes = len(class_names)
    LOGGER.info("Classes: %s", class_names)

    model = keras.models.load_model(model_path)

    test_ds = create_image_dataset(
        test_dir,
        class_names,
        img_size,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        cache_path=model_dir / ".cache" / "test.cache",
    )

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    LOGGER.info("Test samples: %d", len(y_true))

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    LOGGER.info("Test Loss    : %.4f", loss)
    LOGGER.info("Test Accuracy: %.2f%%", accuracy * 100)

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix (Accuracy: {accuracy * 100:.1f}%)")

    thresh = conf_matrix.max() / 2 if conf_matrix.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                conf_matrix[i, j],
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )

    plt.tight_layout()
    cm_path = model_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    LOGGER.info("Confusion matrix saved to %s", cm_path)
    plt.show()

    LOGGER.info("Per-Class Accuracy:")
    for i, cls in enumerate(class_names):
        total = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        pct = (correct / total * 100) if total > 0 else 0
        LOGGER.info("  %-20s %d/%d (%.1f%%)", cls, correct, total, pct)


if __name__ == "__main__":
    main()
