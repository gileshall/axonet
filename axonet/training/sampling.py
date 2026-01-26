"""Camera sampling utilities for dataset generation."""

import math
import random
from typing import List, Optional, Tuple

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


def compute_neuron_pca(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA on node positions to find the neuron's natural axes.

    Args:
        positions: (N, 3) array of node positions

    Returns:
        eigenvectors: (3, 3) columns are principal components, sorted by descending eigenvalue
        eigenvalues: (3,) sorted descending
        center: (3,) mean position
    """
    center = positions.mean(axis=0)
    centered = positions - center
    cov = np.dot(centered.T, centered) / max(len(positions) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvectors.astype(np.float32), eigenvalues.astype(np.float32), center.astype(np.float32)


def pca_guided_sampling(
    positions: np.ndarray,
    n_canonical: int = 6,
    n_biased: int = 12,
    n_random: int = 6,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Sample camera directions guided by PCA of neuron geometry.

    Three tiers:
    - canonical: +/-PC1, +/-PC2, +/-PC3 (guaranteed broadside views)
    - biased: concentrated near the PC1-PC2 plane (largest projected area)
    - random: uniform on S^2 for diversity

    Falls back to fibonacci_sphere if the neuron is near-spherical or has
    too few positions.

    Args:
        positions: (N, 3) array of node positions
        n_canonical: number of canonical views (should be even, max 6)
        n_biased: number of biased views
        n_random: number of random views
        rng: optional random number generator

    Returns:
        directions: (n_total, 3) unit vectors
        tiers: list of tier labels per direction
    """
    total = n_canonical + n_biased + n_random

    if rng is None:
        rng = random

    # Fallback for too few positions
    if len(positions) < 10:
        return fibonacci_sphere(total, rng=rng), ["fibonacci"] * total

    eigenvectors, eigenvalues, _ = compute_neuron_pca(positions)

    # Fallback for near-spherical neurons
    if eigenvalues[2] > 0 and eigenvalues[0] / max(eigenvalues[2], 1e-12) < 1.1:
        return fibonacci_sphere(total, rng=rng), ["fibonacci"] * total

    dirs: List[np.ndarray] = []
    tiers: List[str] = []

    # Tier A: canonical views along principal axes
    canonical_axes = []
    for i in range(min(3, n_canonical // 2 + n_canonical % 2)):
        canonical_axes.append(eigenvectors[:, i])
        if len(canonical_axes) < n_canonical:
            canonical_axes.append(-eigenvectors[:, i])
    canonical_axes = canonical_axes[:n_canonical]
    for d in canonical_axes:
        d = d / (np.linalg.norm(d) + 1e-12)
        dirs.append(d.astype(np.float32))
        tiers.append("canonical")

    # Tier B: biased toward PC1-PC2 plane (low dot with PC3)
    pc3 = eigenvectors[:, 2]
    # Generate candidate pool from fibonacci sphere
    n_candidates = max(4 * n_biased, 48)
    candidates = fibonacci_sphere(n_candidates, rng=rng)
    # Score by |dot(d, pc3)| — lower means more broadside
    scores = np.abs(candidates @ pc3)
    order = np.argsort(scores)

    # Angular threshold to avoid duplicating canonical views (~15 degrees)
    cos_threshold = math.cos(math.radians(15.0))
    canonical_arr = np.array(dirs, dtype=np.float32)  # current canonical dirs

    biased_dirs = []
    for idx in order:
        if len(biased_dirs) >= n_biased:
            break
        c = candidates[idx]
        # Check angular distance to all canonical directions
        if len(canonical_arr) > 0:
            dots = np.abs(canonical_arr @ c)
            if np.any(dots > cos_threshold):
                continue
        biased_dirs.append(c)

    # If we couldn't find enough (unlikely), fill from remaining candidates
    if len(biased_dirs) < n_biased:
        for idx in order:
            if len(biased_dirs) >= n_biased:
                break
            c = candidates[idx]
            already = any(np.allclose(c, bd) for bd in biased_dirs)
            if not already and not any(np.allclose(c, d) for d in dirs):
                biased_dirs.append(c)

    for d in biased_dirs:
        dirs.append(d.astype(np.float32))
        tiers.append("biased")

    # Tier C: random for diversity
    random_dirs = random_sphere(n_random, rng=rng)
    for d in random_dirs:
        dirs.append(d.astype(np.float32))
        tiers.append("random")

    directions = np.array(dirs, dtype=np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
    return directions, tiers


def compute_projected_extent(positions: np.ndarray, direction: np.ndarray) -> float:
    """Compute the projected extent of positions onto the plane perpendicular to direction.

    Args:
        positions: (N, 3) array of positions
        direction: (3,) unit vector (camera look direction)

    Returns:
        Diagonal of the 2D bounding box of projected points
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction) + 1e-12

    # Build an orthonormal basis for the image plane
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(direction, up)) > 0.95:
        up = np.array([0.0, 0.0, 1.0])
    u = np.cross(direction, up)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(u, direction)
    v /= np.linalg.norm(v) + 1e-12

    # Project positions onto u, v
    centered = positions - positions.mean(axis=0)
    proj_u = centered @ u
    proj_v = centered @ v

    extent_u = proj_u.max() - proj_u.min()
    extent_v = proj_v.max() - proj_v.min()

    return float(math.sqrt(extent_u ** 2 + extent_v ** 2))

