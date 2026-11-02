"""Streaming file hashes without workflow dependencies."""
import hashlib
from functools import lru_cache
from pathlib import Path
from PIL import Image

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=100000)
def pixel_sha256(path: Path) -> str:
    """Hash RGB dimensions and pixels using the native cached comparison rule."""
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()

