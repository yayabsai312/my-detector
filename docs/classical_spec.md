# Specification of `src/classical.py` (Module 1)

Give this file to your AI assistant and ask it to implement `src/classical.py`. Every function takes and returns NumPy
arrays. Colour images are RGB; greyscale images are 2-D `uint8` arrays (0 = black, 255 = white).
`python -m pytest -q tests` checks your implementation (all 13 tests must pass).

| Function | What it does |
|---|---|
| `load_rgb(path)` | Read an image file as RGB (OpenCV reads BGR). Raise `FileNotFoundError` on a bad path (OpenCV returns `None` silently). |
| `to_grey(rgb)` | Weighted sum 0.299 R + 0.587 G + 0.114 B. |
| `to_hsv(rgb)` | Convert to HSV. |
| `brightness_contrast(img, alpha=1.0, beta=0.0)` | `alpha * pixel + beta`, **clipped** to 0–255 (no uint8 wrap-around). |
| `gaussian_blur(grey, size=5)` | Gaussian blur; raise `ValueError` if `size` is even. |
| `apply_kernel(grey, kernel)` | Convolve with a hand-written kernel (OpenCV `filter2D`), result clipped to 0–255. |
| `convolve_by_hand(grey, kernel)` | The same operation written with two loops; float32 output, reflected borders like OpenCV. |
| `KERNELS` | Dict with `identity`, `average`, `sharpen`, `horizontal_change` (3 × 3). |
| `threshold_fixed(grey, t)` | Brighter than `t` → 255, else 0. |
| `threshold_otsu(grey)` | Returns `(t, binary)`, with Otsu choosing `t`. |
| `threshold_adaptive(grey, block=21, c=10)` | A threshold per neighbourhood (Gaussian-weighted mean minus `c`). |
| `sobel(grey)` | Returns `(gx, gy, magnitude_0_255)`. |
| `laplacian(grey)` | Absolute second derivative as `uint8`. |
| `canny(grey, low, high)` | Canny edges; raise `ValueError` if `low >= high`. |
| `edge_pixels(edges)` | Number of non-zero pixels. |

## Details that matter (the tests check them)
- Use OpenCV functions where they exist (`cv2.cvtColor`, `cv2.convertScaleAbs`, `cv2.GaussianBlur`, `cv2.filter2D`, `cv2.threshold`,
  `cv2.adaptiveThreshold`, `cv2.Sobel`, `cv2.Laplacian`, `cv2.Canny`): they round and clip the way the tests expect.
- `convolve_by_hand` must give the same numbers as `cv2.filter2D` with `borderType=cv2.BORDER_REFLECT_101`, which is NumPy's
  `np.pad(..., mode="reflect")`. Like `filter2D`, do **not** flip the kernel (strictly speaking this is correlation).
- `KERNELS` values: identity `[[0,0,0],[0,1,0],[0,0,0]]`; average: nine times `1/9`; sharpen `[[0,-1,0],[-1,5,-1],[0,-1,0]]`;
  horizontal_change `[[-1,0,1],[-1,0,1],[-1,0,1]]`. Use `float32`.
- `threshold_otsu` returns `t` as an `int`. `sobel` uses `ksize=3` and scales the magnitude to 0–255 by its maximum.
