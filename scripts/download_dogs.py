#!/usr/bin/env python3
"""Download dog images from dog.ceo into train/val/test class folders.

Uses a throttled request loop to avoid overloading the API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

API_URL = "https://dog.ceo/api/breeds/image/random"
USER_AGENT = "image-classifier-dog-downloader/1.0"

LOGGER = logging.getLogger("download_dogs")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dog images into data splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def request_json(url: str, timeout_seconds: float) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def fetch_random_image_url(timeout_seconds: float, max_retries: int) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            payload = request_json(API_URL, timeout_seconds=timeout_seconds)
            if payload.get("status") != "success":
                raise ValueError(f"Unexpected API response status: {payload!r}")
            image_url = payload.get("message")
            if not image_url or not isinstance(image_url, str):
                raise ValueError(f"Missing image URL in API response: {payload!r}")
            return image_url
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Failed to fetch image URL after {max_retries} attempts") from exc
            backoff = 0.5 * attempt + random.uniform(0.0, 0.25)
            LOGGER.warning("API request failed (attempt %d/%d): %s; retrying in %.2fs", attempt, max_retries, exc, backoff)
            time.sleep(backoff)

    raise RuntimeError("Unreachable retry logic")


def extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        return suffix
    return ".jpg"


def download_image(url: str, destination: Path, timeout_seconds: float) -> None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout_seconds) as response:
        data = response.read()
    destination.write_bytes(data)


def make_destination(split_dir: Path, image_url: str, index: int) -> Path:
    image_hash = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    ext = extension_from_url(image_url)
    filename = f"dog_{index:05d}_{image_hash}{ext}"
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
    seen_urls: set[str] = set()
    i = existing

    while i < count:
        image_url = fetch_random_image_url(timeout_seconds=timeout_seconds, max_retries=max_retries)
        if image_url in seen_urls:
            LOGGER.debug("Skipping duplicate URL in this run: %s", image_url)
            time.sleep(delay_seconds)
            continue
        seen_urls.add(image_url)

        destination = make_destination(split_dir, image_url, i + 1)
        if destination.exists():
            i += 1
            continue

        try:
            download_image(image_url, destination, timeout_seconds=timeout_seconds)
            downloaded += 1
            i += 1
            LOGGER.info("Saved %s", destination)
        except (HTTPError, URLError, TimeoutError) as exc:
            LOGGER.warning("Failed to download %s: %s", image_url, exc)

        time.sleep(delay_seconds)

    LOGGER.info("Completed %s: downloaded %d new files", split_dir, downloaded)
    return downloaded


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must be >= 0")

    dog_dirs = {
        "train": args.data_dir / "train" / "dogs",
        "val": args.data_dir / "val" / "dogs",
        "test": args.data_dir / "test" / "dogs",
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
            dog_dirs[split],
            target,
            delay_seconds=args.delay_seconds,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )

    LOGGER.info("Done. Downloaded %d new images in total.", total_new)


if __name__ == "__main__":
    main()
