#!/usr/bin/env python3
"""Download cat images from cataas.com into train/val/test class folders.

Uses a throttled request loop to avoid overloading the API.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CAT_API_URL = "https://cataas.com/cat"
USER_AGENT = "image-classifier-cat-downloader/1.0"

LOGGER = logging.getLogger("download_cats")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download cat images into data splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def download_random_cat_image(destination: Path, timeout_seconds: float, max_retries: int) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            req = Request(CAT_API_URL, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout_seconds) as response:
                data = response.read()
                content_type = response.headers.get("Content-Type", "")

            if not content_type.startswith("image/"):
                raise ValueError(f"Unexpected content type: {content_type}")

            destination.write_bytes(data)
            return
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Failed to download cat image after {max_retries} attempts") from exc
            backoff = 0.5 * attempt + random.uniform(0.0, 0.25)
            LOGGER.warning(
                "Cat download failed (attempt %d/%d): %s; retrying in %.2fs",
                attempt,
                max_retries,
                exc,
                backoff,
            )
            time.sleep(backoff)


def make_destination(split_dir: Path, index: int, nonce: str) -> Path:
    digest = hashlib.sha1(f"{index}-{nonce}".encode("utf-8")).hexdigest()[:12]
    filename = f"cat_{index:05d}_{digest}.jpg"
    return split_dir / filename


def populate_split(
    split_dir: Path,
    count: int,
    *,
    delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    existing = len([p for p in split_dir.iterdir() if p.is_file()])

    LOGGER.info("Preparing %s (existing files: %d, target total: %d)", split_dir, existing, count)

    downloaded = 0
    i = existing
    while i < count:
        destination = make_destination(split_dir, i + 1, f"{time.time_ns()}")
        if destination.exists():
            i += 1
            continue

        try:
            download_random_cat_image(
                destination=destination,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            downloaded += 1
            i += 1
            LOGGER.info("Saved %s", destination)
        except RuntimeError as exc:
            LOGGER.warning("Skipping failed image download: %s", exc)

        time.sleep(delay_seconds)

    LOGGER.info("Completed %s: downloaded %d new files", split_dir, downloaded)
    return downloaded


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must be >= 0")

    cat_dirs = {
        "train": args.data_dir / "train" / "cats",
        "val": args.data_dir / "val" / "cats",
        "test": args.data_dir / "test" / "cats",
    }
    targets = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }

    total_new = 0
    for split in ("train", "val", "test"):
        target = targets[split]
        if target < 0:
            raise ValueError(f"{split} count must be >= 0")
        total_new += populate_split(
            cat_dirs[split],
            target,
            delay_seconds=args.delay_seconds,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )

    LOGGER.info("Done. Downloaded %d new images in total.", total_new)


if __name__ == "__main__":
    main()
