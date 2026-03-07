# Dynamic Image Classifier — Build Guide
**TensorFlow · Keras · NumPy · Pillow · Matplotlib**  
Folder-based Dynamic Labels · Train & Evaluate · Full Source Code

---

## Table of Contents
1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Training Script](#4-training-script-trainpy)
5. [Evaluation Script](#5-evaluation-script-evaluatepy)
6. [CNN Architecture Explained](#6-cnn-architecture-explained)
7. [Running the Project](#7-running-the-project)
8. [Tips & Best Practices](#8-tips--best-practices)
9. [Output Files](#9-output-files)

---

## 1. Overview

This guide walks you through building a fully dynamic image classification model using Python. The system automatically discovers class labels from sub-folder names — no hard-coded labels needed. The same workflow is used for both training and evaluation.

### How It Works

- Training images are stored in sub-folders inside a `train/` directory
- Each sub-folder name becomes a class label (e.g. `cats/`, `dogs/`, `cars/`)
- The model is a Convolutional Neural Network (CNN) built with Keras
- A separate `test/` directory — same structure — is used for evaluation
- After evaluation, accuracy metrics and a confusion matrix are displayed

### Technology Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Build, compile, and train the CNN model |
| NumPy | Array manipulation and numerical operations |
| Pillow (PIL) | Image loading and preprocessing |
| Matplotlib | Plot training curves and confusion matrix |

---

## 2. Project Structure

Organise your project files and image data exactly as shown below. The classifier reads labels dynamically, so simply add or remove sub-folders to change the number of classes.

```
image_classifier/
  ├── train.py               # Training script
  ├── evaluate.py            # Evaluation script
  ├── model/
  │   └── classifier.h5      # Saved model (after training)
  ├── data/
  │   ├── train/             # Training images
  │   │   ├── cats/          # ← label = 'cats'
  │   │   │   ├── cat001.jpg
  │   │   │   └── cat002.jpg
  │   │   ├── dogs/          # ← label = 'dogs'
  │   │   │   ├── dog001.jpg
  │   │   │   └── dog002.jpg
  │   │   └── cars/          # ← label = 'cars'  (add any class)
  │   └── test/              # Evaluation images (same structure)
  │       ├── cats/
  │       ├── dogs/
  │       └── cars/
  └── requirements.txt
```

> **Tip:** To add a new class, simply create a new sub-folder with the class name inside both `data/train/` and `data/test/` and populate them with images. No code changes are needed.

---

## 3. Installation

### 3.1 Requirements File

Create a `requirements.txt` in your project root:

```
tensorflow>=2.12.0
numpy>=1.23.0
Pillow>=9.0.0
matplotlib>=3.6.0
```

### 3.2 Create a Virtual Environment (Recommended)

```bash
# Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Training Script (`train.py`)

The training script dynamically scans the training directory, builds a CNN, trains it, and saves the resulting model to disk.

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────
TRAIN_DIR  = "data/train"
MODEL_PATH = "model/classifier.h5"
IMG_SIZE   = (128, 128)  # Width x Height
BATCH_SIZE = 32
EPOCHS     = 20

# ─── Discover labels dynamically ──────────────────────────────────────────────
class_names = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# ─── Build image dataset ───────────────────────────────────────────────────────
def load_dataset(directory, class_names, img_size):
    images, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(directory, cls)
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize(img_size)
                images.append(np.array(img))
                labels.append(idx)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
    return np.array(images, dtype="float32") / 255.0, np.array(labels)

X_train, y_train = load_dataset(TRAIN_DIR, class_names, IMG_SIZE)
print(f"Training samples: {len(X_train)}")

# ─── Build CNN model ───────────────────────────────────────────────────────────
def build_model(num_classes, img_size):
    inputs = keras.Input(shape=(*img_size, 3))
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = build_model(num_classes, IMG_SIZE)
model.summary()

# ─── Compile & train ──────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15
)

# ─── Save model and labels ────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)

with open("model/class_names.txt", "w") as f:
    f.write("\n".join(class_names))
print(f"Model saved to {MODEL_PATH}")

# ─── Plot training curves ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],    label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend()
axes[1].plot(history.history["loss"],    label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss"); axes[1].legend()
plt.tight_layout()
plt.savefig("model/training_curves.png")
plt.show()
```

### Configuration Options

| Parameter | Description |
|---|---|
| `TRAIN_DIR` | Path to the root training folder containing class sub-folders |
| `MODEL_PATH` | Where the trained model (`.h5` file) will be saved |
| `IMG_SIZE` | All images are resized to this `(width, height)` in pixels |
| `BATCH_SIZE` | Number of images processed per gradient update step |
| `EPOCHS` | How many full passes through the training data to run |

---

## 5. Evaluation Script (`evaluate.py`)

The evaluation script loads the saved model, runs it against the test directory, and reports overall accuracy plus a per-class confusion matrix.

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────
TEST_DIR   = "data/test"
MODEL_PATH = "model/classifier.h5"
IMG_SIZE   = (128, 128)

# ─── Load class names saved during training ───────────────────────────────────
with open("model/class_names.txt") as f:
    class_names = [l.strip() for l in f.readlines()]
num_classes = len(class_names)
print(f"Classes: {class_names}")

# ─── Load model ───────────────────────────────────────────────────────────────
model = keras.models.load_model(MODEL_PATH)

# ─── Load test data ───────────────────────────────────────────────────────────
def load_dataset(directory, class_names, img_size):
    images, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(directory, cls)
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize(img_size)
                images.append(np.array(img))
                labels.append(idx)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
    return np.array(images, dtype="float32") / 255.0, np.array(labels)

X_test, y_test = load_dataset(TEST_DIR, class_names, IMG_SIZE)
print(f"Test samples: {len(X_test)}")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss    : {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ─── Confusion matrix ─────────────────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test), axis=1)

conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(y_test, y_pred):
    conf_matrix[true][pred] += 1

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix  (Accuracy: {accuracy*100:.1f}%)")

thresh = conf_matrix.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, conf_matrix[i, j], ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

# ─── Per-class accuracy ───────────────────────────────────────────────────────
print("\nPer-Class Accuracy:")
for i, cls in enumerate(class_names):
    total   = conf_matrix[i].sum()
    correct = conf_matrix[i][i]
    pct     = (correct / total * 100) if total > 0 else 0
    print(f"  {cls:<20} {correct}/{total}  ({pct:.1f}%)")
```

---

## 6. CNN Architecture Explained

The model uses a classic Convolutional Neural Network pattern: alternating convolution and pooling layers for feature extraction, followed by dense layers for classification.

### Layer-by-Layer Breakdown

| Layer | Role |
|---|---|
| Input | Accepts RGB images resized to `IMG_SIZE × 3` channels |
| Conv2D (32 filters) | Detects low-level features: edges, textures, colours |
| MaxPooling2D | Halves spatial dimensions; reduces overfitting |
| Conv2D (64 filters) | Learns mid-level patterns from pooled feature maps |
| MaxPooling2D | Further spatial reduction |
| Conv2D (128 filters) | Captures high-level abstract features |
| MaxPooling2D | Final spatial reduction before flattening |
| Flatten | Converts 3D feature maps to a 1D vector |
| Dense (256) | Fully-connected layer for combination of features |
| Dropout (0.5) | Randomly zeroes 50% of activations to prevent overfitting |
| Dense (softmax) | Outputs a probability for each class (`num_classes` outputs) |

### Why These Choices?

- **Sparse Categorical Crossentropy** — used because labels are integers, not one-hot encoded
- **Adam Optimizer** — adaptive learning rate; works well out of the box
- **Dropout (0.5)** — strong regularisation for small-to-medium datasets
- **Softmax output** — produces a probability distribution that sums to 1.0

---

## 7. Running the Project

### Step-by-Step Workflow

1. **Prepare your data** — place images in `data/train/<label>/` and `data/test/<label>/`
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the model:**
   ```bash
   python train.py
   ```
4. **Evaluate the model:**
   ```bash
   python evaluate.py
   ```
5. **Check results** — accuracy and confusion matrix are printed to the terminal and saved in `model/`

### Expected Terminal Output

```
# Training
Found 3 classes: ['cars', 'cats', 'dogs']
Training samples: 2400
Epoch 1/20
64/64 ━━━━━━━━━━━━━━━━━━━━  12s  accuracy: 0.4821  val_accuracy: 0.5200
...
Epoch 20/20
64/64 ━━━━━━━━━━━━━━━━━━━━  11s  accuracy: 0.9112  val_accuracy: 0.8733
Model saved to model/classifier.h5

# Evaluation
Classes: ['cars', 'cats', 'dogs']
Test samples: 600

Test Loss    : 0.3451
Test Accuracy: 87.33%

Per-Class Accuracy:
  cars                 194/200  (97.0%)
  cats                 161/200  (80.5%)
  dogs                 169/200  (84.5%)
```

---

## 8. Tips & Best Practices

### 8.1 Data Quality

- Aim for at least **200 images per class** for reasonable results; 1000+ is ideal
- Ensure **balanced classes** — similar number of images per folder
- Use consistent image formats (JPEG or PNG). Pillow will skip unreadable files automatically
- More diverse training images (different angles, lighting) improve generalisation

### 8.2 Improving Accuracy

**Data Augmentation** — add random flips, rotations, and zoom during training:

```python
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

augmentation = keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.1),
    RandomZoom(0.1),
])
```

**Early Stopping** — halt training when validation loss stops improving:

```python
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
model.fit(..., callbacks=[callback])
```

**Transfer Learning** — use a pretrained backbone for much better results with small datasets:

```python
base = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
base.trainable = False  # Freeze pretrained weights

inputs  = keras.Input(shape=(*IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model   = keras.Model(inputs, outputs)
```

### 8.3 Adding a New Class

1. Create `data/train/<new_class>/` and `data/test/<new_class>/`
2. Populate both folders with images of the new class
3. Re-run `python train.py` — labels are detected automatically
4. Re-run `python evaluate.py` to see updated accuracy

### 8.4 Common Issues

| Issue | Solution |
|---|---|
| Low accuracy (<50%) | Add more training images; try data augmentation |
| Overfitting (val > train loss) | Increase Dropout; add more data or augmentation |
| Memory error | Reduce `BATCH_SIZE` or `IMG_SIZE` in configuration |
| `FileNotFoundError` | Verify `TRAIN_DIR` / `TEST_DIR` paths and folder structure |
| Class not detected | Ensure sub-folder exists in `train/` — hidden files are ignored |

---

## 9. Output Files

| File | Contents |
|---|---|
| `model/classifier.h5` | Saved Keras model (weights + architecture) |
| `model/class_names.txt` | One class name per line — used by `evaluate.py` |
| `model/training_curves.png` | Line charts of accuracy and loss per epoch |
| `model/confusion_matrix.png` | Grid showing predicted vs. actual labels |

---

> **Note:** This guide assumes Python 3.9+ and TensorFlow 2.12+. GPU acceleration is used automatically if CUDA is available; training will fall back to CPU otherwise.
