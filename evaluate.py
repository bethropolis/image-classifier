import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from utils import configure_logging, create_image_dataset, ensure_class_directories, load_class_names

LOGGER = logging.getLogger(__name__)

TEST_DIR = Path("data/test")
MODELS_DIR = Path("models")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image classifier")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--models-dir", "--model-dir", dest="models_dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--class-names-path", type=Path, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=list(IMG_SIZE))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-plot", action="store_true", help="Save plots but do not open interactive windows")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def resolve_run_artifacts(args: argparse.Namespace) -> tuple[Path, Path, str, Path]:
    models_dir: Path = args.models_dir

    if args.model_path is not None:
        model_path = args.model_path
        class_names_path = args.class_names_path or model_path.parent / "class_names.txt"
        run_name = model_path.parent.name
        run_dir = model_path.parent
        return model_path, class_names_path, run_name, run_dir

    if args.run_name:
        run_dir = models_dir / args.run_name
        model_path = run_dir / "model.keras"
        class_names_path = run_dir / "class_names.txt"
        return model_path, class_names_path, args.run_name, run_dir

    latest_path = models_dir / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest run pointer found at {latest_path}. "
            "Run training first or pass --run-name/--model-path."
        )

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    model_path = Path(payload["model_path"])
    class_names_path = Path(payload["class_names_path"])
    run_name = payload.get("run_name", model_path.parent.name)
    run_dir = Path(payload.get("run_dir", model_path.parent))
    return model_path, class_names_path, run_name, run_dir


def save_evaluation_report(
    path: Path,
    *,
    run_name: str,
    test_samples: int,
    loss: float,
    accuracy: float,
    backbone: str | None,
    evaluated_at_utc: str,
    class_names: list[str],
    conf_matrix: np.ndarray,
) -> None:
    per_class = {}
    for i, cls in enumerate(class_names):
        total = int(conf_matrix[i].sum())
        correct = int(conf_matrix[i, i])
        pct = (correct / total * 100.0) if total > 0 else 0.0
        per_class[cls] = {
            "correct": correct,
            "total": total,
            "accuracy_pct": pct,
        }

    payload = {
        "run_name": run_name,
        "test_samples": test_samples,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "backbone": backbone,
        "evaluated_at_utc": evaluated_at_utc,
        "class_names": class_names,
        "confusion_matrix": conf_matrix.tolist(),
        "per_class": per_class,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)
    test_dir: Path = args.test_dir

    model_path, class_names_path, run_name, run_dir = resolve_run_artifacts(args)
    run_metadata_path = run_dir / "run_metadata.json"

    if run_metadata_path.exists():
        try:
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            image_size = run_metadata.get("image_size")
            if isinstance(image_size, list) and len(image_size) == 2:
                img_size = (int(image_size[0]), int(image_size[1]))
                LOGGER.info("Image size loaded from run metadata: %s", img_size)
            else:
                img_size = (args.img_size[0], args.img_size[1])
                LOGGER.warning("run_metadata.json missing valid image_size, using --img-size: %s", img_size)
        except Exception as exc:  # noqa: BLE001
            img_size = (args.img_size[0], args.img_size[1])
            LOGGER.warning("Failed to parse run_metadata.json (%s), using --img-size: %s", exc, img_size)
    else:
        img_size = (args.img_size[0], args.img_size[1])
        LOGGER.warning("No run_metadata.json found, using --img-size: %s", img_size)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    class_names = load_class_names(class_names_path)
    ensure_class_directories(test_dir, class_names)
    num_classes = len(class_names)

    LOGGER.info("Evaluating run: %s", run_name)
    LOGGER.info("Model: %s", model_path)
    LOGGER.info("Class names: %s", class_names_path)
    LOGGER.info("Classes: %s", class_names)

    model = keras.models.load_model(model_path)

    test_ds = create_image_dataset(
        test_dir,
        class_names,
        img_size,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
        cache_path=run_dir / ".cache" / f"test-{img_size[0]}x{img_size[1]}-b{args.batch_size}.cache",
    )

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    test_samples = len(y_true)
    LOGGER.info("Test samples: %d", test_samples)

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    loss_vector = keras.losses.sparse_categorical_crossentropy(y_true, y_pred_probs)
    loss = float(np.mean(loss_vector))
    accuracy = float(np.mean(y_pred == y_true))
    LOGGER.info("Test Loss    : %.4f", loss)
    LOGGER.info("Test Accuracy: %.2f%%", accuracy * 100)

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        conf_matrix[true, pred] += 1

    row_totals = conf_matrix.sum(axis=1, keepdims=True)
    conf_pct = np.divide(
        conf_matrix,
        row_totals,
        out=np.zeros_like(conf_matrix, dtype=float),
        where=row_totals != 0,
    )

    support = conf_matrix.sum(axis=1)
    y_labels = [f"{cls} (n={support[i]})" for i, cls in enumerate(class_names)]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(conf_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized proportion")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(
        f"Confusion Matrix | run={run_name}\n"
        f"acc={accuracy * 100:.2f}% | loss={loss:.4f} | test_samples={test_samples}"
    )

    for i in range(num_classes):
        for j in range(num_classes):
            pct = conf_pct[i, j] * 100
            count = conf_matrix[i, j]
            label = f"{count}\n{pct:.1f}%"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if conf_pct[i, j] > 0.5 else "black",
                fontsize=9,
            )

    plt.tight_layout()
    cm_path = run_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    LOGGER.info("Confusion matrix saved to %s", cm_path)
    if not args.no_plot:
        plt.show()
    else:
        plt.close(fig)

    LOGGER.info("Per-Class Accuracy:")
    for i, cls in enumerate(class_names):
        total = conf_matrix[i].sum()
        correct = conf_matrix[i, i]
        pct = (correct / total * 100) if total > 0 else 0
        LOGGER.info("  %-20s %d/%d (%.1f%%)", cls, correct, total, pct)

    report_path = run_dir / "evaluation_report.json"
    backbone = None
    if run_metadata_path.exists():
        try:
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            backbone = run_metadata.get("backbone")
        except Exception:  # noqa: BLE001
            backbone = None
    save_evaluation_report(
        report_path,
        run_name=run_name,
        test_samples=test_samples,
        loss=float(loss),
        accuracy=float(accuracy),
        backbone=backbone,
        evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
        class_names=class_names,
        conf_matrix=conf_matrix,
    )
    LOGGER.info("Evaluation report saved to %s", report_path)


if __name__ == "__main__":
    main()
