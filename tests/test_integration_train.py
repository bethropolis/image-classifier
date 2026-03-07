from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

import train


def write_dummy_rgb_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    image.save(path)


def _populate_split(root: Path, classes: list[str], images_per_class: int) -> None:
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(images_per_class):
            write_dummy_rgb_image(cls_dir / f"{cls}_{idx}.png")


def test_train_one_epoch_dummy_data(tmp_path: Path, monkeypatch) -> None:
    classes = ["cats", "dogs"]
    train_dir = tmp_path / "data" / "train"
    val_dir = tmp_path / "data" / "val"
    model_dir = tmp_path / "model"

    _populate_split(train_dir, classes, images_per_class=4)
    _populate_split(val_dir, classes, images_per_class=2)

    monkeypatch.setattr(train.plt, "show", lambda: None)

    train.main(
        [
            "--train-dir",
            str(train_dir),
            "--val-dir",
            str(val_dir),
            "--model-dir",
            str(model_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--img-size",
            "32",
            "32",
            "--learning-rate",
            "0.001",
            "--backbone",
            "custom",
            "--lr-scheduler",
            "none",
            "--pretrained",
            "none",
            "--log-level",
            "WARNING",
        ]
    )

    assert (model_dir / "classifier.keras").exists()
    assert (model_dir / "class_names.txt").exists()
    assert (model_dir / "training_curves.png").exists()
