#!/usr/bin/env python3
"""Download fox images from randomfox.ca into train/val/test class folders.

API: https://randomfox.ca/floof/
Response: {"image": "https://randomfox.ca/images/...", "link": "..."}
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

API_URL = "https://randomfox.ca/floof/"
USER_AGENT = "image-classifier-fox-downloader/1.0"

LOGGER = logging.getLogger("download_foxes")


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[handler])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fox images into data splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _request_json(url: str, timeout: float) -> dict:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_random_image_url(timeout: float, max_retries: int) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            payload = _request_json(API_URL, timeout)
            url = payload.get("image")
            if not url or not isinstance(url, str):
                raise ValueError(f"Unexpected response: {payload!r}")
            return url
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Failed to fetch fox URL after {max_retries} attempts") from exc
            backoff = 0.5 * attempt + random.uniform(0.0, 0.25)
            LOGGER.warning("API request failed (attempt %d/%d): %s — retrying in %.2fs", attempt, max_retries, exc, backoff)
            time.sleep(backoff)
    raise RuntimeError("Unreachable")


def download_image(url: str, destination: Path, timeout: float) -> None:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        destination.write_bytes(resp.read())


def make_destination(split_dir: Path, image_url: str, index: int) -> Path:
    ext = Path(urlparse(image_url).path).suffix.lower() or ".jpg"
    digest = hashlib.sha1(image_url.encode()).hexdigest()[:12]
    return split_dir / f"fox_{index:05d}_{digest}{ext}"


def populate_split(
    split_dir: Path,
    count: int,
    *,
    delay_seconds: float,
    timeout_seconds: float,
    max_retries: int,
) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for p in split_dir.iterdir() if p.is_file())
    LOGGER.info("Preparing %s  (existing: %d, target: %d)", split_dir, existing, count)

    downloaded = 0
    seen_urls: set[str] = set()
    i = existing

    while i < count:
        try:
            image_url = fetch_random_image_url(timeout_seconds, max_retries)
        except RuntimeError as exc:
            LOGGER.warning("Skipping — could not get URL: %s", exc)
            time.sleep(delay_seconds)
            continue

        if image_url in seen_urls:
            time.sleep(delay_seconds)
            continue
        seen_urls.add(image_url)

        destination = make_destination(split_dir, image_url, i + 1)
        if destination.exists():
            i += 1
            continue

        try:
            download_image(image_url, destination, timeout_seconds)
            LOGGER.info("Saved %s", destination)
            downloaded += 1
            i += 1
        except (HTTPError, URLError, TimeoutError) as exc:
            LOGGER.warning("Download failed for %s: %s", image_url, exc)

        time.sleep(delay_seconds)

    LOGGER.info("Done %s — %d new files downloaded.", split_dir, downloaded)
    return downloaded


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds must be >= 0")

    dirs = {
        "train": args.data_dir / "train" / "foxes",
        "val":   args.data_dir / "val"   / "foxes",
        "test":  args.data_dir / "test"  / "foxes",
    }
    targets = {"train": args.train_count, "val": args.val_count, "test": args.test_count}

    total = 0
    for split in ("train", "val", "test"):
        if targets[split] < 0:
            raise ValueError(f"--{split}-count must be >= 0")
        total += populate_split(
            dirs[split], targets[split],
            delay_seconds=args.delay_seconds,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
    LOGGER.info("Finished. %d new fox images downloaded in total.", total)


if __name__ == "__main__":
    main()
