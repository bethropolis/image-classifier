# Dynamic Image Classifier (uv)

Image classifier starter using:
- Pillow
- TensorFlow
- Keras
- NumPy
- Matplotlib

Class labels are discovered dynamically from folder names.

## Project Layout

```text
.
├── train.py
├── evaluate.py
├── utils.py
├── pyproject.toml
├── data/
│   ├── train/
│   │   └── <class_name>/
│   ├── val/                  # optional, preferred for stable validation
│   │   └── <class_name>/
│   └── test/
│       └── <class_name>/
└── model/
```

## Setup with uv

```bash
uv sync
```

## Train

```bash
uv run train
```

Training now uses:
- `tf.data` pipeline (`image_dataset_from_directory`)
- prefetching + cache files under `model/.cache/`
- built-in data augmentation (flip/rotation/zoom/brightness)
- early stopping + best-model checkpointing
- learning-rate scheduling (`plateau`, `cosine`, or `none`)
- optional transfer learning backbones (`mobilenetv2`, `efficientnetb0`)
- structured logging (use `--log-level`)

Validation behavior:
- If `data/val/` contains class folders, that directory is used for validation.
- Otherwise a deterministic seeded split from `data/train/` is used.

Outputs:
- `model/classifier.keras` (best checkpoint)
- `model/class_names.txt`
- `model/training_curves.png`

Useful training args:

```bash
uv run train --epochs 30 --batch-size 64 --img-size 160 160 --learning-rate 0.0005
```

Transfer-learning example:

```bash
uv run train --backbone mobilenetv2 --pretrained imagenet
```

Cosine scheduler example:

```bash
uv run train --lr-scheduler cosine --learning-rate 0.001
```

## Evaluate

```bash
uv run evaluate
```

Evaluation uses the same shared `tf.data` utilities and writes:
- `model/confusion_matrix.png`
- terminal metrics (loss, accuracy, per-class accuracy)

Useful evaluation args:

```bash
uv run evaluate --batch-size 64 --img-size 160 160
```

Optional dispatcher entry point:

```bash
uv run classifier train
uv run classifier evaluate
```

## Tests

Pytest suite includes unit tests for class discovery/dataset/model shape and a 1-epoch integration train test on dummy images.

```bash
uv run pytest -q
```

## Notes

- Class folder names must match across train/val/test.
- Python is constrained to `>=3.10,<3.13` for TensorFlow compatibility.
