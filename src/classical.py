"""Module 1: classical computer-vision operations. See docs/classical_spec.md."""
import cv2
import numpy as np


def load_rgb(path):
    """Read an image file as RGB; raise FileNotFoundError if it cannot be read."""
    img = cv2.imread(str(path))
    if img is None:  # OpenCV returns None silently on a bad path
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_grey(rgb):
    """Greyscale as 0.299 R + 0.587 G + 0.114 B."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def to_hsv(rgb):
    """Convert RGB to HSV (OpenCV hue runs 0-179)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def brightness_contrast(img, alpha=1.0, beta=0.0):
    """alpha * pixel + beta, clipped to 0-255 (no uint8 wrap-around)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gaussian_blur(grey, size=5):
    """Gaussian blur with a size x size window; size must be odd."""
    if size % 2 == 0:
        raise ValueError(f"size must be odd, got {size}")
    return cv2.GaussianBlur(grey, (size, size), 0)


def apply_kernel(grey, kernel):
    """Convolve with a hand-written kernel; result clipped to 0-255."""
    return cv2.filter2D(grey, -1, kernel)


def convolve_by_hand(grey, kernel):
    """Same as filter2D (no kernel flip, reflected borders), written with two loops."""
    kernel = np.asarray(kernel, np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(grey.astype(np.float32), ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros(grey.shape, np.float32)
    for y in range(grey.shape[0]):
        for x in range(grey.shape[1]):
            out[y, x] = np.sum(padded[y:y + kh, x:x + kw] * kernel)
    return out


KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32),
    "average": np.full((3, 3), 1 / 9, np.float32),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    "horizontal_change": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),
}


def threshold_fixed(grey, t):
    """Pixels brighter than t become 255, the rest 0."""
    _, binary = cv2.threshold(grey, t, 255, cv2.THRESH_BINARY)
    return binary


def threshold_otsu(grey):
    """Return (t, binary) with Otsu choosing t."""
    t, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t), binary


def threshold_adaptive(grey, block=21, c=10):
    """Threshold per neighbourhood: Gaussian-weighted mean minus c."""
    return cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def sobel(grey):
    """Return (gx, gy, magnitude scaled to 0-255 by its maximum)."""
    gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    peak = mag.max()
    mag = mag / peak * 255 if peak > 0 else mag  # a flat image has no edges
    return gx, gy, mag.astype(np.uint8)


def laplacian(grey):
    """Absolute second derivative as uint8."""
    return cv2.convertScaleAbs(cv2.Laplacian(grey, cv2.CV_64F))


def canny(grey, low, high):
    """Canny edges; low must be smaller than high."""
    if low >= high:
        raise ValueError(f"low ({low}) must be smaller than high ({high})")
    return cv2.Canny(grey, low, high)


def edge_pixels(edges):
    """Number of non-zero pixels."""
    return int(np.count_nonzero(edges))
