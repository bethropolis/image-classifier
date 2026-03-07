import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from utils import IMAGE_EXTENSIONS, load_leaderboard

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

_MODEL_CACHE: dict[str, Any] = {}
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKHF]|\x1b\].*?\x07|\r")


def _clean_console_text(raw: str) -> str:
    return _ANSI_ESCAPE.sub("", raw)


def _count_images_in_dir(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


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


def latest_run_badge(models_dir: Path) -> str:
    latest_path = models_dir / "latest.json"
    if not latest_path.exists():
        return "Latest run: _none_"
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        run_name = payload.get("run_name", "unknown")
        return f"Latest run: **`{run_name}`** (used when selection is blank)"
    except Exception as exc:  # noqa: BLE001
        return f"Latest run: _unavailable_ ({exc})"


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


def _run_command_stream(cmd: list[str], cwd: Path):
    header = f"$ {' '.join(cmd)}\n\n"
    logs = header
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    if proc.stdout is not None:
        for line in proc.stdout:
            logs += _clean_console_text(line)
            yield logs, None

    exit_code = proc.wait()
    logs += f"\n[exit_code={exit_code}]\n"
    yield logs, exit_code


def _format_eval_report(report_path: Path) -> str:
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
    return "\n".join(report_lines)


def train_from_ui(
    data_dir: str,
    models_dir: str,
    run_name: str,
    backbone: str,
    pretrained: bool,
    unfreeze_backbone: bool,
    disable_augmentation: bool,
    early_stopping_patience: int,
    epochs: int,
    batch_size: int,
    img_h: int,
    img_w: int,
    learning_rate: float,
    lr_scheduler: str,
):
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
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--img-size",
        str(int(img_h)),
        str(int(img_w)),
        "--learning-rate",
        str(float(learning_rate)),
        "--lr-scheduler",
        lr_scheduler,
        "--early-stopping-patience",
        str(int(early_stopping_patience)),
        "--pretrained",
        "imagenet" if pretrained else "none",
        "--no-plot",
    ]
    if run_name.strip():
        cmd.extend(["--run-name", run_name.strip()])
    if unfreeze_backbone:
        cmd.append("--unfreeze-backbone")
    if disable_augmentation:
        cmd.append("--disable-augmentation")

    latest_text = latest_pointer_text(Path(models_dir))
    yield "Starting training...", latest_text, None

    final_logs = ""
    exit_code = 1
    for logs, code in _run_command_stream(cmd, root):
        final_logs = logs
        latest_text = latest_pointer_text(Path(models_dir))
        yield logs, latest_text, None
        if code is not None:
            exit_code = code

    curves_img = None
    if exit_code == 0:
        try:
            run_dir = resolve_run_dir(Path(models_dir), run_name.strip() or None)
            curves_path = run_dir / "training_curves.png"
            if curves_path.exists():
                curves_img = str(curves_path)
        except Exception:  # noqa: BLE001
            pass

    yield final_logs, latest_pointer_text(Path(models_dir)), curves_img


def evaluate_from_ui(
    data_dir: str,
    models_dir: str,
    run_name: str,
    batch_size: int,
):
    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "evaluate.py",
        "--test-dir",
        str(Path(data_dir) / "test"),
        "--models-dir",
        models_dir,
        "--batch-size",
        str(int(batch_size)),
        "--no-plot",
    ]

    run_name = run_name.strip()
    if run_name:
        cmd.extend(["--run-name", run_name])

    yield "Starting evaluation...", "Waiting for evaluation report...", None

    final_logs = ""
    exit_code = 1
    for logs, code in _run_command_stream(cmd, root):
        final_logs = logs
        yield logs, "Running...", None
        if code is not None:
            exit_code = code

    report_md = "No evaluation report found."
    cm_img = None
    if exit_code == 0:
        try:
            run_dir = resolve_run_dir(Path(models_dir), run_name if run_name else None)
            report_path = run_dir / "evaluation_report.json"
            cm_path = run_dir / "confusion_matrix.png"
            if report_path.exists():
                report_md = _format_eval_report(report_path)
            if cm_path.exists():
                cm_img = str(cm_path)
        except Exception as exc:  # noqa: BLE001
            report_md = f"Could not load report: {exc}"

    yield final_logs, report_md, cm_img


def _load_model_for_run(run_dir: Path):
    from tensorflow import keras  # lazy import for faster app startup

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


def _get_run_img_size(run_dir: Path, model) -> tuple[int, int]:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        image_size = payload.get("image_size")
        if isinstance(image_size, list) and len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])

    shape = getattr(model, "input_shape", None)
    if isinstance(shape, tuple) and len(shape) >= 3 and shape[1] and shape[2]:
        return int(shape[1]), int(shape[2])

    return 128, 128


def predict_image(
    image: Image.Image | None,
    models_dir: str,
    run_name: str,
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

    img_h, img_w = _get_run_img_size(run_dir, model)
    rgb = image.convert("RGB").resize((img_w, img_h))
    x = np.array(rgb, dtype=np.float32)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
    status = f"Run: {run_dir.name} | Input resized to: {img_h}x{img_w} | Predicted: {top_label} ({probs[top_idx] * 100:.2f}%)"

    score_map: dict[str, float] = {}
    for i, cls in enumerate(class_names):
        score_map[cls] = float(probs[i])
    return status, score_map


def refresh_ui(data_dir: str, models_dir: str):
    data_summary = dataset_summary(Path(data_dir))
    runs = list_runs(Path(models_dir))
    latest_md = latest_pointer_text(Path(models_dir))
    badge = latest_run_badge(Path(models_dir))
    run_choices = [""] + runs
    return (
        data_summary,
        latest_md,
        gr.update(choices=run_choices, value=""),
        gr.update(choices=run_choices, value=""),
        badge,
        badge,
    )


LEADERBOARD_HEADERS = [
    "Rank",
    "Run",
    "Backbone",
    "Accuracy (%)",
    "Loss",
    "Score",
    "Strength Tags",
    "Evaluated At (UTC)",
]


def refresh_leaderboard(models_dir: str) -> tuple[str, list[list]]:
    def _relative_time(iso_ts: str) -> str:
        if not iso_ts or iso_ts == "n/a":
            return "n/a"
        try:
            dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            seconds = max(0, int((now - dt).total_seconds()))
        except Exception:  # noqa: BLE001
            return "n/a"

        if seconds < 60:
            return f"{seconds}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        if days < 7:
            return f"{days}d ago"
        weeks = days // 7
        if weeks < 52:
            return f"{weeks}w ago"
        years = weeks // 52
        return f"{years}y ago"

    entries = load_leaderboard(Path(models_dir))
    if not entries:
        return "No evaluated runs found yet.", []

    rows: list[list] = []
    for idx, entry in enumerate(entries, start=1):
        accuracy_pct = float(entry.get("test_accuracy", 0.0)) * 100.0
        loss = float(entry.get("test_loss", 0.0))
        score = accuracy_pct - (loss * 10.0)
        per_class = entry.get("per_class", {}) or {}
        strengths = [
            cls
            for cls, stats in per_class.items()
            if float((stats or {}).get("accuracy_pct", 0.0)) >= 80.0
        ]
        strength_tags = ", ".join(strengths) if strengths else "none"

        rows.append(
            [
                idx,
                entry.get("run_name", "unknown"),
                entry.get("backbone", "unknown"),
                round(accuracy_pct, 2),
                round(loss, 4),
                round(score, 2),
                strength_tags,
                _relative_time(entry.get("evaluated_at_utc", "n/a")),
            ]
        )

    top = rows[0]
    summary = (
        f"Top run: **`{top[1]}`** | "
        f"Accuracy: **{top[3]}%** | Loss: **{top[4]}** | Score: **{top[5]}**"
    )
    return summary, rows


def _start_refresh_ui():
    return (
        gr.update(interactive=False, value="Refreshing..."),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def _end_refresh_ui():
    return (
        gr.update(interactive=True, value="Refresh"),
        gr.update(interactive=True, value="Start Training"),
        gr.update(interactive=True, value="Run Evaluation"),
        gr.update(interactive=True, value="Predict"),
        gr.update(interactive=True, value="Refresh Leaderboard"),
    )


def _start_train_ui():
    return (
        gr.update(interactive=False, value="Training..."),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        "",
        None,
    )


def _end_train_ui():
    return (
        gr.update(interactive=True, value="Start Training"),
        gr.update(interactive=True, value="Run Evaluation"),
        gr.update(interactive=True, value="Predict"),
        gr.update(interactive=True, value="Refresh"),
        gr.update(interactive=True, value="Refresh Leaderboard"),
    )


def _start_eval_ui():
    return (
        gr.update(interactive=False, value="Evaluating..."),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        "",
        "",
        None,
    )


def _end_eval_ui():
    return (
        gr.update(interactive=True, value="Run Evaluation"),
        gr.update(interactive=True, value="Start Training"),
        gr.update(interactive=True, value="Predict"),
        gr.update(interactive=True, value="Refresh"),
        gr.update(interactive=True, value="Refresh Leaderboard"),
    )


def _start_predict_ui():
    return (
        gr.update(interactive=False, value="Predicting..."),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        "",
        None,
    )


def _end_predict_ui():
    return (
        gr.update(interactive=True, value="Predict"),
        gr.update(interactive=True, value="Start Training"),
        gr.update(interactive=True, value="Run Evaluation"),
        gr.update(interactive=True, value="Refresh"),
        gr.update(interactive=True, value="Refresh Leaderboard"),
    )


def _start_leaderboard_refresh_ui():
    return (
        gr.update(interactive=False, value="Refreshing Leaderboard..."),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def _end_leaderboard_refresh_ui():
    return (
        gr.update(interactive=True, value="Refresh Leaderboard"),
        gr.update(interactive=True, value="Refresh"),
        gr.update(interactive=True, value="Start Training"),
        gr.update(interactive=True, value="Run Evaluation"),
        gr.update(interactive=True, value="Predict"),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Image Classifier GUI") as demo:
        gr.Markdown("# Image Classifier GUI")
        gr.Markdown("Train, evaluate, and run predictions with versioned model runs.")

        with gr.Row():
            data_dir_in = gr.Textbox(label="Data Directory", value=str(DATA_DIR))
            models_dir_in = gr.Textbox(label="Models Directory", value=str(MODELS_DIR))
            refresh_btn = gr.Button("Refresh")

        with gr.Accordion("Dataset Summary", open=False):  # open=False = collapsed by default
            dataset_box = gr.Textbox(label="", lines=14, container=False)

        latest_md = gr.Markdown(label="Latest Run")

        with gr.Tab("Train"):
            with gr.Row():
                run_name_in = gr.Textbox(label="Run Name (optional)", placeholder="run-20260307-120000")
                backbone_in = gr.Dropdown(
                    choices=["custom", "mobilenetv2", "efficientnetb0"],
                    value="mobilenetv2",
                    label="Backbone",
                )
                scheduler_in = gr.Dropdown(choices=["none", "plateau", "cosine"], value="plateau", label="LR Scheduler")
            with gr.Row():
                pretrained_in = gr.Checkbox(label="Use ImageNet pretrained weights", value=True)
                unfreeze_in = gr.Checkbox(label="Unfreeze backbone (fine-tune)", value=False)
                disable_aug_in = gr.Checkbox(label="Disable augmentation", value=False)
                patience_in = gr.Number(label="Early stopping patience", value=5, precision=0)
            with gr.Row():
                epochs_in = gr.Number(label="Epochs", value=20, precision=0)
                batch_in = gr.Number(label="Batch Size", value=32, precision=0)
                img_h_in = gr.Number(label="Image Height", value=128, precision=0)
                img_w_in = gr.Number(label="Image Width", value=128, precision=0)
                lr_in = gr.Number(label="Learning Rate", value=0.001)
            train_btn = gr.Button("Start Training")
            train_logs = gr.Textbox(label="Training Logs", lines=16)
            train_curve_img = gr.Image(label="Training Curves", type="filepath")

        with gr.Tab("Evaluate"):
            eval_latest_badge = gr.Markdown("Latest run: _none_")
            eval_run_dd = gr.Dropdown(choices=[""], value="", label="Run Name (blank = latest)")
            eval_batch = gr.Number(label="Batch Size", value=32, precision=0)
            eval_btn = gr.Button("Run Evaluation")
            eval_logs = gr.Textbox(label="Evaluation Logs", lines=14)
            eval_report = gr.Markdown(label="Evaluation Report")
            eval_cm_img = gr.Image(label="Confusion Matrix", type="filepath")

        with gr.Tab("Predict"):
            pred_latest_badge = gr.Markdown("Latest run: _none_")
            pred_run_dd = gr.Dropdown(choices=[""], value="", label="Run Name (blank = latest)")
            image_in = gr.Image(type="pil", label="Input Image")
            pred_btn = gr.Button("Predict")
            pred_status = gr.Textbox(label="Prediction")
            pred_scores = gr.Label(label="Class Probabilities")

        with gr.Tab("Leaderboard"):
            leaderboard_summary = gr.Markdown("No evaluated runs found yet.")
            leaderboard_refresh_btn = gr.Button("Refresh Leaderboard")
            leaderboard_df = gr.Dataframe(
                headers=LEADERBOARD_HEADERS,
                datatype=["number", "str", "str", "number", "number", "number", "str", "str"],
                value=[],
                row_count=(1, "dynamic"),
                col_count=(len(LEADERBOARD_HEADERS), "fixed"),
                label="Model Ranking",
            )

        refresh_btn.click(
            fn=_start_refresh_ui,
            outputs=[refresh_btn, train_btn, eval_btn, pred_btn, leaderboard_refresh_btn],
        ).then(
            fn=refresh_ui,
            inputs=[data_dir_in, models_dir_in],
            outputs=[dataset_box, latest_md, eval_run_dd, pred_run_dd, eval_latest_badge, pred_latest_badge],
        ).then(
            fn=refresh_leaderboard,
            inputs=[models_dir_in],
            outputs=[leaderboard_summary, leaderboard_df],
        ).then(
            fn=_end_refresh_ui,
            outputs=[refresh_btn, train_btn, eval_btn, pred_btn, leaderboard_refresh_btn],
        )

        train_btn.click(
            fn=_start_train_ui,
            outputs=[train_btn, eval_btn, pred_btn, refresh_btn, leaderboard_refresh_btn, train_logs, train_curve_img],
        ).then(
            fn=train_from_ui,
            inputs=[
                data_dir_in,
                models_dir_in,
                run_name_in,
                backbone_in,
                pretrained_in,
                unfreeze_in,
                disable_aug_in,
                patience_in,
                epochs_in,
                batch_in,
                img_h_in,
                img_w_in,
                lr_in,
                scheduler_in,
            ],
            outputs=[train_logs, latest_md, train_curve_img],
        ).then(
            fn=refresh_ui,
            inputs=[data_dir_in, models_dir_in],
            outputs=[dataset_box, latest_md, eval_run_dd, pred_run_dd, eval_latest_badge, pred_latest_badge],
        ).then(
            fn=refresh_leaderboard,
            inputs=[models_dir_in],
            outputs=[leaderboard_summary, leaderboard_df],
        ).then(
            fn=_end_train_ui,
            outputs=[train_btn, eval_btn, pred_btn, refresh_btn, leaderboard_refresh_btn],
        )

        eval_btn.click(
            fn=_start_eval_ui,
            outputs=[eval_btn, train_btn, pred_btn, refresh_btn, leaderboard_refresh_btn, eval_logs, eval_report, eval_cm_img],
        ).then(
            fn=evaluate_from_ui,
            inputs=[data_dir_in, models_dir_in, eval_run_dd, eval_batch],
            outputs=[eval_logs, eval_report, eval_cm_img],
        ).then(
            fn=refresh_leaderboard,
            inputs=[models_dir_in],
            outputs=[leaderboard_summary, leaderboard_df],
        ).then(
            fn=_end_eval_ui,
            outputs=[eval_btn, train_btn, pred_btn, refresh_btn, leaderboard_refresh_btn],
        )

        leaderboard_refresh_btn.click(
            fn=_start_leaderboard_refresh_ui,
            outputs=[leaderboard_refresh_btn, refresh_btn, train_btn, eval_btn, pred_btn],
        ).then(
            fn=refresh_leaderboard,
            inputs=[models_dir_in],
            outputs=[leaderboard_summary, leaderboard_df],
        ).then(
            fn=_end_leaderboard_refresh_ui,
            outputs=[leaderboard_refresh_btn, refresh_btn, train_btn, eval_btn, pred_btn],
        )

        pred_btn.click(
            fn=_start_predict_ui,
            outputs=[pred_btn, train_btn, eval_btn, refresh_btn, leaderboard_refresh_btn, pred_status, pred_scores],
        ).then(
            fn=predict_image,
            inputs=[image_in, models_dir_in, pred_run_dd],
            outputs=[pred_status, pred_scores],
        ).then(
            fn=_end_predict_ui,
            outputs=[pred_btn, train_btn, eval_btn, refresh_btn, leaderboard_refresh_btn],
        )

        demo.load(
            fn=refresh_ui,
            inputs=[data_dir_in, models_dir_in],
            outputs=[dataset_box, latest_md, eval_run_dd, pred_run_dd, eval_latest_badge, pred_latest_badge],
        ).then(
            fn=refresh_leaderboard,
            inputs=[models_dir_in],
            outputs=[leaderboard_summary, leaderboard_df],
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
