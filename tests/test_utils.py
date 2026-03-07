from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image

from utils import build_model, discover_classes, load_dataset


def write_dummy_rgb_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    image.save(path)


def _make_dataset_tree(root: Path) -> list[str]:
    classes = ["cats", "dogs"]
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            write_dummy_rgb_image(cls_dir / f"{cls}_{idx}.png")
    return classes


def test_discover_classes_sorted(tmp_path: Path) -> None:
    (tmp_path / "dogs").mkdir()
    (tmp_path / "cats").mkdir()

    result = discover_classes(tmp_path)

    assert result == ["cats", "dogs"]


def test_load_dataset_batches(tmp_path: Path) -> None:
    class_names = _make_dataset_tree(tmp_path)

    ds = load_dataset(
        directory=tmp_path,
        class_names=class_names,
        img_size=(32, 32),
        batch_size=2,
        shuffle=False,
        seed=42,
    )

    images, labels = next(iter(ds))
    assert images.shape[1:] == (32, 32, 3)
    assert labels.shape[0] == 2
    assert labels.dtype in (tf.int32, tf.int64)


def test_build_model_output_shape_custom() -> None:
    model = build_model(num_classes=3, img_size=(32, 32), backbone="custom")
    assert model.output_shape == (None, 3)


def test_build_model_output_shape_transfer_no_pretrained() -> None:
    model = build_model(
        num_classes=4,
        img_size=(64, 64),
        backbone="mobilenetv2",
        pretrained_weights=None,
        freeze_backbone=True,
    )
    assert model.output_shape == (None, 4)
