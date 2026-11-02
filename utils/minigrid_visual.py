"""Leaf deterministic image transforms for the optional Task C probe profile."""
import hashlib

import numpy as np
from PIL import Image


def matched_probe_visual(image: Image.Image, visual: str, seed_material: str) -> Image.Image:
    """Apply deterministic matched-probe noise or edge-preserving style filters."""
    if visual == 'clean':
        return image
    if visual == 'noisy':
        seed = int.from_bytes(hashlib.sha256(seed_material.encode('utf-8')).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        array = np.asarray(image, dtype=np.float64)
        noise = rng.normal(0.0, 20.0, array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    if visual == 'style':
        import cv2
        return Image.fromarray(cv2.edgePreservingFilter(
            np.asarray(image), flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4))
    raise ValueError('Unsupported matched-probe visual condition: ' + visual)
