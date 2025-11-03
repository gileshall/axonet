"""Camera sampling utilities for dataset generation."""

import math
import random
from typing import Optional

import numpy as np


def fibonacci_sphere(n: int, *, jitter: float = 0.0, rng: Optional[random.Random] = None) -> np.ndarray:
    """Return (n,3) unit vectors approximately uniformly distributed on S^2.
    
    Args:
        n: Number of points to generate
        jitter: Optional jitter amount for randomization
        rng: Optional random number generator
        
    Returns:
        (n, 3) array of unit vectors on sphere
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if rng is None:
        rng = random
    
    ga = math.pi * (3.0 - math.sqrt(5.0))
    z = np.linspace(1 - 1/n, -1 + 1/n, n, dtype=np.float32)
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    theta = np.arange(n, dtype=np.float32) * ga
    
    if jitter > 0:
        theta = theta + np.array([rng.uniform(-jitter, jitter) for _ in range(n)], dtype=np.float32)
        z = np.clip(z + np.array([rng.uniform(-jitter, jitter) for _ in range(n)], dtype=np.float32) * (1.0 / n), -1.0, 1.0)
        r = np.sqrt(np.maximum(0.0, 1 - z * z))
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return dirs.astype(np.float32)


def random_sphere(n: int, rng: Optional[random.Random] = None) -> np.ndarray:
    """Random unit vectors on S^2 using Gaussian sampling.
    
    Args:
        n: Number of points to generate
        rng: Optional random number generator
        
    Returns:
        (n, 3) array of random unit vectors
    """
    if rng is None:
        rng = random
    v = np.array([[rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(n)], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v

