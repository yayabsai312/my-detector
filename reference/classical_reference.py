"""Module 1 — classical computer vision operations, as small documented functions.

Every function takes and returns NumPy arrays. Colour images are RGB (we convert from OpenCV's BGR once, on
loading). Greyscale images are 2-D uint8 arrays with values 0 (black) to 255 (white).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------------ images and colour
def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB array (height, width, 3). OpenCV reads BGR, so we convert once here."""
    bgr = cv2.imread(str(path))
    if bgr is None:                                   # imread does not raise: it returns None on a bad path
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_grey(rgb: np.ndarray) -> np.ndarray:
    """Weighted sum of the channels: 0.299 R + 0.587 G + 0.114 B (the eye is most sensitive to green)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Hue (which colour, 0-179 in OpenCV), saturation (how pure), value (how bright)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def brightness_contrast(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """new = alpha * pixel + beta, clipped to 0-255. alpha scales contrast, beta shifts brightness."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ------------------------------------------------------------------ filters and kernels
def gaussian_blur(grey: np.ndarray, size: int = 5) -> np.ndarray:
    """Weighted average of each neighbourhood; closer pixels weigh more. size must be odd."""
    if size % 2 == 0:
        raise ValueError("kernel size must be odd, so the window has a centre pixel")
    return cv2.GaussianBlur(grey, (size, size), 0)


def apply_kernel(grey: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide a hand-written kernel over the image (OpenCV filter2D). The result is clipped to 0-255."""
    return cv2.filter2D(grey, -1, np.asarray(kernel, dtype=np.float32))


def convolve_by_hand(grey: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """The same operation written out: at every pixel, multiply the window by the kernel and add.
    Returns float32 (no clipping), with reflected borders like OpenCV's default."""
    k = np.asarray(kernel, dtype=np.float32)
    r = k.shape[0] // 2
    padded = np.pad(grey.astype(np.float32), r, mode="reflect")
    out = np.zeros(grey.shape, np.float32)
    for i in range(grey.shape[0]):
        for j in range(grey.shape[1]):
            out[i, j] = (padded[i:i + 2 * r + 1, j:j + 2 * r + 1] * k).sum()
    return out


KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32),
    "average": np.ones((3, 3), np.float32) / 9,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    "horizontal_change": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),
}


# ------------------------------------------------------------------ thresholding
def threshold_fixed(grey: np.ndarray, t: int) -> np.ndarray:
    """Pixels brighter than t become 255 (white), the rest 0 (black)."""
    return cv2.threshold(grey, t, 255, cv2.THRESH_BINARY)[1]


def threshold_otsu(grey: np.ndarray) -> tuple[int, np.ndarray]:
    """Otsu chooses t automatically: the value that best splits the histogram into two groups."""
    t, out = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t), out


def threshold_adaptive(grey: np.ndarray, block: int = 21, c: int = 10) -> np.ndarray:
    """A different threshold for every block × block neighbourhood (its weighted mean minus c)."""
    return cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)


# ------------------------------------------------------------------ edges
def sobel(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First derivative in x and y, and the gradient magnitude scaled to 0-255."""
    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag8 = np.uint8(255 * mag / mag.max()) if mag.max() > 0 else np.zeros_like(grey)
    return gx, gy, mag8


def laplacian(grey: np.ndarray) -> np.ndarray:
    """Absolute second derivative (all directions at once). Very sensitive to noise: blur first."""
    return cv2.convertScaleAbs(cv2.Laplacian(grey, cv2.CV_64F, ksize=3))


def canny(grey: np.ndarray, low: int, high: int) -> np.ndarray:
    """Thin, connected edges: kept above `high`, discarded below `low`, kept in between only if connected."""
    if low >= high:
        raise ValueError("low threshold must be smaller than high threshold")
    return cv2.Canny(grey, low, high)


def edge_pixels(edges: np.ndarray) -> int:
    return int((edges > 0).sum())
