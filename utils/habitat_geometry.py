"""Habitat world-coordinate heading calculations, in degrees."""
from __future__ import annotations
import math
import numpy as np


def yaw_to_target(position: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    return math.degrees(math.atan2(-float(delta[0]), -float(delta[2])))

