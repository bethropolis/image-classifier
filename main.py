import argparse

import app
import evaluate
import train


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Image classifier command runner",
        usage="python main.py {train|evaluate|gui} [options]",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Run model training")
    subparsers.add_parser("evaluate", help="Run model evaluation")
    subparsers.add_parser("gui", help="Launch Gradio GUI")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, passthrough = parse_args(argv)
    if args.command == "train":
        train.main(passthrough)
        return
    if args.command == "evaluate":
        evaluate.main(passthrough)
        return
    if args.command == "gui":
        app.launch(passthrough)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
