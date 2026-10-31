"""Checks that each Module 1 operation does what its explanation says. Uses small synthetic images."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import classical as cl

rng = np.random.default_rng(0)
IMG = rng.integers(0, 256, (40, 50), dtype=np.uint8)


def test_load_rgb_raises_on_bad_path():
    with pytest.raises(FileNotFoundError):
        cl.load_rgb("does/not/exist.png")


def test_grey_uses_weighted_sum():
    pure_green = np.zeros((2, 2, 3), np.uint8); pure_green[..., 1] = 255
    pure_blue = np.zeros((2, 2, 3), np.uint8); pure_blue[..., 2] = 255
    assert cl.to_grey(pure_green)[0, 0] == round(0.587 * 255)        # 150
    assert cl.to_grey(pure_blue)[0, 0] == round(0.114 * 255)         # 29: green looks brighter than blue


def test_brightness_clips_at_255():
    out = cl.brightness_contrast(np.full((2, 2), 220, np.uint8), beta=60)
    assert out.max() == 255                                          # no wrap-around, unlike raw uint8 addition
    assert (np.full((2, 2), 220, np.uint8) + np.uint8(60))[0, 0] == 24   # what raw uint8 arithmetic would do


def test_identity_kernel_changes_nothing():
    assert np.array_equal(cl.apply_kernel(IMG, cl.KERNELS["identity"]), IMG)


def test_hand_convolution_matches_opencv():
    k = cl.KERNELS["sharpen"]
    import cv2
    ref = cv2.filter2D(IMG.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(cl.convolve_by_hand(IMG, k) - ref).max() < 1e-3


def test_blur_requires_odd_size_and_reduces_variation():
    with pytest.raises(ValueError):
        cl.gaussian_blur(IMG, 4)
    assert cl.gaussian_blur(IMG, 21).std() < cl.gaussian_blur(IMG, 5).std() < IMG.std()


def test_otsu_splits_two_groups():
    two = np.concatenate([np.full(500, 50), np.full(500, 200)]).astype(np.uint8).reshape(20, 50)
    t, out = cl.threshold_otsu(two)
    assert 50 <= t < 200
    assert set(np.unique(out)) == {0, 255}


def test_edges_respond_to_a_step_and_not_to_a_flat_image():
    flat = np.full((30, 30), 120, np.uint8)
    step = flat.copy(); step[:, 15:] = 220
    assert cl.edge_pixels(cl.canny(flat, 100, 200)) == 0
    assert cl.edge_pixels(cl.canny(step, 100, 200)) > 0
    _, _, mag = cl.sobel(step)
    assert mag[:, 14:16].mean() > mag[:, :10].mean()


def test_higher_canny_thresholds_keep_fewer_edges():
    noisy = cl.gaussian_blur(IMG, 3)
    counts = [cl.edge_pixels(cl.canny(noisy, lo, hi)) for lo, hi in [(30, 90), (100, 200), (200, 400)]]
    assert counts[0] >= counts[1] >= counts[2]
    with pytest.raises(ValueError):
        cl.canny(noisy, 200, 100)


def test_hsv_hue_of_pure_red_and_blue():
    red = np.zeros((2, 2, 3), np.uint8); red[..., 0] = 255
    blue = np.zeros((2, 2, 3), np.uint8); blue[..., 2] = 255
    assert cl.to_hsv(red)[0, 0, 0] == 0                  # hue 0 = red
    assert cl.to_hsv(blue)[0, 0, 0] == 120               # OpenCV hue runs 0-179: blue is 120


def test_fixed_threshold_is_strictly_greater():
    img = np.array([[99, 100, 101]], np.uint8)
    assert cl.threshold_fixed(img, 100).tolist() == [[0, 0, 255]]


def test_adaptive_threshold_handles_uneven_light():
    grad = np.tile(np.linspace(40, 220, 60, dtype=np.uint8), (60, 1))     # light grows left to right
    img = grad.copy(); img[25:35, ::6] = np.clip(grad[25:35, ::6].astype(int) - 60, 0, 255)   # dark marks everywhere
    out = cl.threshold_adaptive(img, 21, 10)
    assert set(np.unique(out)) <= {0, 255}
    assert (out[25:35, ::6] == 0).mean() > 0.9            # marks found on both the dark and the bright side


def test_laplacian_zero_on_flat_and_ramp():
    flat = np.full((20, 20), 90, np.uint8)
    ramp = np.tile(np.arange(20, dtype=np.uint8) * 5, (20, 1))            # constant slope: second derivative 0
    assert cl.laplacian(flat).max() == 0
    assert cl.laplacian(ramp)[:, 2:-2].max() == 0
