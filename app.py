import argparse
import json
import subprocess
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow import keras

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

_MODEL_CACHE: dict[str, keras.Model] = {}


def _count_images_in_dir(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def dataset_summary(data_dir: Path) -> str:
    lines = [f"Data root: {data_dir}"]
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        lines.append(f"\n[{split}] {split_dir}")
        if not split_dir.exists():
            lines.append("  (missing)")
            continue

        class_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        if not class_dirs:
            lines.append("  (no class directories)")
            continue

        total = 0
        for cls_dir in class_dirs:
            count = _count_images_in_dir(cls_dir)
            total += count
            lines.append(f"  - {cls_dir.name}: {count}")
        lines.append(f"  total: {total}")
    return "\n".join(lines)


def list_runs(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    runs = [p.name for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    return sorted(runs, reverse=True)


def latest_pointer_text(models_dir: Path) -> str:
    latest_path = models_dir / "latest.json"
    if not latest_path.exists():
        return "No latest run pointer found."

    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"Failed to parse latest.json: {exc}"

    run_name = payload.get("run_name", "unknown")
    model_path = payload.get("model_path", "unknown")
    best_val_acc = payload.get("best_val_accuracy")
    best_val_loss = payload.get("best_val_loss")
    return (
        f"Latest run: `{run_name}`\n\n"
        f"Model: `{model_path}`\n\n"
        f"Best val acc: `{best_val_acc}` | Best val loss: `{best_val_loss}`"
    )


def resolve_run_dir(models_dir: Path, run_name: str | None) -> Path:
    if run_name:
        run_dir = models_dir / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_dir}")
        return run_dir

    latest_path = models_dir / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError("No latest run pointer found. Train a model first.")
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    return Path(payload["run_dir"])


def run_command(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    header = f"$ {' '.join(cmd)}\nexit_code={proc.returncode}\n\n"
    return header + (proc.stdout or "")


def train_from_ui(
    data_dir: str,
    models_dir: str,
    run_name: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    img_h: int,
    img_w: int,
    learning_rate: float,
    lr_scheduler: str,
) -> tuple[str, str]:
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "train.py",
        "--train-dir",
        str(Path(data_dir) / "train"),
        "--val-dir",
        str(Path(data_dir) / "val"),
        "--models-dir",
        models_dir,
        "--backbone",
        backbone,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--img-size",
        str(img_h),
        str(img_w),
        "--learning-rate",
        str(learning_rate),
        "--lr-scheduler",
        lr_scheduler,
    ]
    if run_name.strip():
        cmd.extend(["--run-name", run_name.strip()])

    logs = run_command(cmd, root)
    latest_text = latest_pointer_text(Path(models_dir))
    return logs, latest_text


def evaluate_from_ui(
    data_dir: str,
    models_dir: str,
    run_name: str,
    batch_size: int,
    img_h: int,
    img_w: int,
) -> tuple[str, str]:
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "evaluate.py",
        "--test-dir",
        str(Path(data_dir) / "test"),
        "--models-dir",
        models_dir,
        "--batch-size",
        str(batch_size),
        "--img-size",
        str(img_h),
        str(img_w),
    ]
    run_name = run_name.strip()
    if run_name:
        cmd.extend(["--run-name", run_name])

    logs = run_command(cmd, root)

    report_md = "No evaluation report found."
    try:
        run_dir = resolve_run_dir(Path(models_dir), run_name if run_name else None)
        report_path = run_dir / "evaluation_report.json"
        if report_path.exists():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report_lines = [
                f"Run: `{payload.get('run_name')}`",
                f"Test Loss: `{payload.get('test_loss'):.4f}`",
                f"Test Accuracy: `{payload.get('test_accuracy') * 100:.2f}%`",
                "",
                "Per-class:",
            ]
            per_class = payload.get("per_class", {})
            for cls, stats in per_class.items():
                report_lines.append(
                    f"- {cls}: {stats['correct']}/{stats['total']} ({stats['accuracy_pct']:.1f}%)"
                )
            report_md = "\n".join(report_lines)
    except Exception as exc:  # noqa: BLE001
        report_md = f"Could not load report: {exc}"

    return logs, report_md


def _load_model_for_run(run_dir: Path) -> tuple[keras.Model, list[str]]:
    cache_key = str(run_dir)
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = keras.models.load_model(run_dir / "model.keras")
        _MODEL_CACHE[cache_key] = model
    class_names = [
        line.strip()
        for line in (run_dir / "class_names.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return model, class_names


def predict_image(
    image: Image.Image | None,
    models_dir: str,
    run_name: str,
    img_h: int,
    img_w: int,
) -> tuple[str, dict[str, float]]:
    if image is None:
        return "No image provided.", {}

    try:
        run_dir = resolve_run_dir(Path(models_dir), run_name.strip() or None)
    except Exception as exc:  # noqa: BLE001
        return f"Run resolution failed: {exc}", {}

    try:
        model, class_names = _load_model_for_run(run_dir)
    except Exception as exc:  # noqa: BLE001
        return f"Model load failed: {exc}", {}

    rgb = image.convert("RGB").resize((img_w, img_h))
    x = np.array(rgb, dtype=np.float32)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
    status = f"Run: {run_dir.name} | Predicted: {top_label} ({probs[top_idx] * 100:.2f}%)"

    score_map: dict[str, float] = {}
    for i, cls in enumerate(class_names):
        score_map[cls] = float(probs[i])
    return status, score_map


def refresh_ui(data_dir: str, models_dir: str):
    data_summary = dataset_summary(Path(data_dir))
    runs = list_runs(Path(models_dir))
    latest_md = latest_pointer_text(Path(models_dir))
    run_choices = [""] + runs
    return data_summary, latest_md, gr.update(choices=run_choices, value=""), gr.update(choices=run_choices, value="")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Image Classifier GUI") as demo:
        gr.Markdown("# Image Classifier GUI")
        gr.Markdown("Train, evaluate, and run predictions with versioned model runs.")

        with gr.Row():
            data_dir_in = gr.Textbox(label="Data Directory", value=str(DATA_DIR))
            models_dir_in = gr.Textbox(label="Models Directory", value=str(MODELS_DIR))
            refresh_btn = gr.Button("Refresh")

        dataset_box = gr.Textbox(label="Dataset Summary", lines=14)
        latest_md = gr.Markdown(label="Latest Run")

        with gr.Tab("Train"):
            with gr.Row():
                run_name_in = gr.Textbox(label="Run Name (optional)", placeholder="run-20260307-120000")
                backbone_in = gr.Dropdown(choices=["custom", "mobilenetv2", "efficientnetb0"], value="custom", label="Backbone")
                scheduler_in = gr.Dropdown(choices=["none", "plateau", "cosine"], value="plateau", label="LR Scheduler")
            with gr.Row():
                epochs_in = gr.Number(label="Epochs", value=20, precision=0)
                batch_in = gr.Number(label="Batch Size", value=32, precision=0)
                img_h_in = gr.Number(label="Image Height", value=128, precision=0)
                img_w_in = gr.Number(label="Image Width", value=128, precision=0)
                lr_in = gr.Number(label="Learning Rate", value=0.001)
            train_btn = gr.Button("Start Training")
            train_logs = gr.Textbox(label="Training Logs", lines=16)

        with gr.Tab("Evaluate"):
            eval_run_dd = gr.Dropdown(choices=[""], value="", label="Run Name (blank = latest)")
            with gr.Row():
                eval_batch = gr.Number(label="Batch Size", value=32, precision=0)
                eval_h = gr.Number(label="Image Height", value=128, precision=0)
                eval_w = gr.Number(label="Image Width", value=128, precision=0)
            eval_btn = gr.Button("Run Evaluation")
            eval_logs = gr.Textbox(label="Evaluation Logs", lines=14)
            eval_report = gr.Markdown(label="Evaluation Report")

        with gr.Tab("Predict"):
            pred_run_dd = gr.Dropdown(choices=[""], value="", label="Run Name (blank = latest)")
            with gr.Row():
                pred_h = gr.Number(label="Image Height", value=128, precision=0)
                pred_w = gr.Number(label="Image Width", value=128, precision=0)
            image_in = gr.Image(type="pil", label="Input Image")
            pred_btn = gr.Button("Predict")
            pred_status = gr.Textbox(label="Prediction")
            pred_scores = gr.Label(label="Class Probabilities")

        refresh_btn.click(
            fn=refresh_ui,
            inputs=[data_dir_in, models_dir_in],
            outputs=[dataset_box, latest_md, eval_run_dd, pred_run_dd],
        )

        train_btn.click(
            fn=train_from_ui,
            inputs=[
                data_dir_in,
                models_dir_in,
                run_name_in,
                backbone_in,
                epochs_in,
                batch_in,
                img_h_in,
                img_w_in,
                lr_in,
                scheduler_in,
            ],
            outputs=[train_logs, latest_md],
        ).then(
            fn=refresh_ui,
            inputs=[data_dir_in, models_dir_in],
            outputs=[dataset_box, latest_md, eval_run_dd, pred_run_dd],
        )

        eval_btn.click(
            fn=evaluate_from_ui,
            inputs=[data_dir_in, models_dir_in, eval_run_dd, eval_batch, eval_h, eval_w],
            outputs=[eval_logs, eval_report],
        )

        pred_btn.click(
            fn=predict_image,
            inputs=[image_in, models_dir_in, pred_run_dd, pred_h, pred_w],
            outputs=[pred_status, pred_scores],
        )

        demo.load(
            fn=refresh_ui,
            inputs=[data_dir_in, models_dir_in],
            outputs=[dataset_box, latest_md, eval_run_dd, pred_run_dd],
        )

    return demo


def launch(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch Gradio UI for image classifier")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    demo = build_app()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    launch()
