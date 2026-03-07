import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import train


def _populate_split(root: Path, classes: list[str], images_per_class: int, make_dummy_image) -> None:
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(images_per_class):
            make_dummy_image(cls_dir / f"{cls}_{idx}.png")


def test_train_one_epoch_dummy_data(tmp_path: Path, monkeypatch, make_dummy_image) -> None:
    classes = ["cats", "dogs"]
    train_dir = tmp_path / "data" / "train"
    val_dir = tmp_path / "data" / "val"
    model_dir = tmp_path / "model"

    _populate_split(train_dir, classes, images_per_class=4, make_dummy_image=make_dummy_image)
    _populate_split(val_dir, classes, images_per_class=2, make_dummy_image=make_dummy_image)

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
            "--no-plot",
            "--log-level",
            "WARNING",
        ]
    )

    latest_path = model_dir / "latest.json"
    assert latest_path.exists()

    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
    run_dir = Path(latest_payload["run_dir"])

    assert (run_dir / "model.keras").exists()
    assert (run_dir / "class_names.txt").exists()
    assert (run_dir / "training_curves.png").exists()
