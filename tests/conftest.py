import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def make_dummy_image():
    def _write(path: Path, size: tuple[int, int] = (32, 32)) -> None:
        array = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")
        image.save(path)

    return _write
