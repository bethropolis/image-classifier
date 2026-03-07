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
├── app.py
├── utils.py
├── pyproject.toml
├── data/
│   ├── train/
│   │   └── <class_name>/
│   ├── val/                  # optional, preferred for stable validation
│   │   └── <class_name>/
│   └── test/
│       └── <class_name>/
└── models/
    ├── latest.json           # points to most recent training run
    └── run-YYYYMMDD-HHMMSS/
        ├── model.keras
        ├── class_names.txt
        ├── training_curves.png
        ├── training_history.json
        ├── run_metadata.json
        ├── confusion_matrix.png
        └── evaluation_report.json
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
- prefetching + cache files under each run dir (`models/<run>/.cache/`)
- built-in data augmentation (flip/rotation/zoom/brightness)
- early stopping + best-model checkpointing
- learning-rate scheduling (`plateau`, `cosine`, or `none`)
- optional transfer learning backbones (`mobilenetv2`, `efficientnetb0`)
- structured logging (use `--log-level`)

Validation behavior:
- If `data/val/` contains class folders, that directory is used for validation.
- Otherwise a deterministic seeded split from `data/train/` is used.

Outputs:
- `models/run-YYYYMMDD-HHMMSS/model.keras` (best checkpoint)
- `models/run-YYYYMMDD-HHMMSS/class_names.txt`
- `models/run-YYYYMMDD-HHMMSS/training_curves.png`
- `models/latest.json` (the run used by default for evaluation)

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
- `models/<run>/confusion_matrix.png`
- `models/<run>/evaluation_report.json`
- terminal metrics (loss, accuracy, per-class accuracy)

Useful evaluation args:

```bash
uv run evaluate --batch-size 64 --img-size 160 160
```

Evaluate a specific run:

```bash
uv run evaluate --run-name run-20260307-044444
```

Optional dispatcher entry point:

```bash
uv run classifier train
uv run classifier evaluate
uv run classifier gui
```

## GUI (Gradio)

Launch the interactive app:

```bash
uv run gui
```

Or with explicit host/port:

```bash
uv run python app.py --host 127.0.0.1 --port 7860
```

The GUI provides:
- Dataset summary (train/val/test class counts)
- Run-aware training controls (creates versioned runs under `models/`)
- Evaluation for latest run or a selected run
- Single-image prediction with class probability display
```

## Tests

Pytest suite includes unit tests for class discovery/dataset/model shape and a 1-epoch integration train test on dummy images.

```bash
uv run pytest -q
```

## Notes

- Class folder names must match across train/val/test.
- Python is constrained to `>=3.10,<3.13` for TensorFlow compatibility.
