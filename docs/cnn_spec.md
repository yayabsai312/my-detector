# Specification of `src/cnn.py` (Module 2, Assignment 2.1)

Give this file to your AI assistant and ask it to implement `src/cnn.py`.
`python -m pytest -q tests/test_cnn.py` checks your work: **10 tests must pass**.
The tests use ResNet-18 without its trained weights, so they download nothing and run in a few seconds.

## What the module does
It turns the Module 2 class into code you can run on **your own photos**:
the size formula, the small CNN you sketched by hand, and three ways to look inside a real network
(what it answers, what its layers see, and where it looked).

## Functions of `src/cnn.py`

| Function | What it does |
|---|---|
| `output_size(n, kernel, padding=0, stride=1)` | Side of the map after a convolution or a pooling: `(n + 2·padding − kernel) // stride + 1`. `output_size(224, 7, 3, 2)` → 112. |
| `small_cnn(n_classes=2)` | The small CNN from the class activity, as an `nn.Sequential` in this exact order: `Conv2d(3, 8, 3, padding=1)`, `ReLU`, `MaxPool2d(2)`, `Conv2d(8, 16, 3, padding=1)`, `ReLU`, `MaxPool2d(2)`, `AdaptiveAvgPool2d(1)`, `Flatten`, `Linear(16, n_classes)`. |
| `trace(model, input_shape=(3, 64, 64))` | Pass a zero image (batch of 1) through each layer of an `nn.Sequential`. Return a list of dicts `{"layer", "shape", "weights"}`: the first row is `{"layer": "input", "shape": (64, 64, 3), "weights": 0}`; then one row per layer with the layer's class name (`"Conv2d"`), its output size as **(height, width, channels)** — or `(n,)` once it is flat — and the number of numbers in its `weight` (biases not counted; 0 if it has none). For `small_cnn()` the weights add up to **1,400**. |
| `count_parameters(model)` | Every number the model learns, weights **and** biases. ResNet-18 → 11,689,512. |
| `load_resnet(pretrained=True)` | Return `(model, preprocess, categories)`: `torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1` (or `weights=None` if `pretrained=False`) **in eval mode**; a function `preprocess(rgb)` that takes an RGB `uint8` numpy array of any size and returns a float tensor `3 × 224 × 224` (resize to exactly 224 × 224 — **no crop**, so the heat map lines up with the whole photo — then `ToTensor` and `Normalize` with the ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`); and the list of the 1,000 class names (`ResNet18_Weights.IMAGENET1K_V1.meta["categories"]`, available without downloading). |
| `classify(model, preprocess, categories, rgb, top=5)` | The `top` most probable classes as a list of `(name, probability)`, highest first. Probabilities come from a softmax. No gradients. |
| `feature_maps(model, preprocess, rgb, layer="layer1")` | The maps one stage of ResNet-18 produces for this image, as a numpy `float32` array `(channels, height, width)`: `layer1` → (64, 56, 56), `layer4` → (512, 7, 7). `layer` must be one of `layer1`–`layer4`, otherwise `ValueError`. Use a forward hook and remove it afterwards. |
| `gradcam(model, preprocess, rgb, layer="layer4", class_index=None)` | Grad-CAM: where the network looked. Forward pass with hooks on `layer` (activations and gradients); choose the most probable class (or `class_index`); back-propagate its score; weight each map by the mean of its gradient, add them up, apply ReLU, **resize to the height and width of `rgb`**, and scale to 0–1. Return `(heat, class_index, probability)`. Remove the hooks afterwards. Never change `rgb`. |
| `overlay(rgb, heat, alpha=0.45)` | The photo with the heat map on top (colour map `jet`: red = looked the most). Return a **new** `uint8` array of the same shape; never change the input. |

## Details that matter (the tests check them)
- No new packages: PyTorch, torchvision, NumPy, Pillow and Matplotlib are already installed with Ultralytics.
- The shapes in `trace` are (height, width, channels), the way we write them in class — PyTorch stores (channels, height, width).
- `gradcam` and `classify` must agree: with `class_index=None`, the class Grad-CAM explains is the first one `classify` returns.

## Then use it: `notebooks/02_my_field.ipynb`
Put **three photos from your own field** in `data/my_field/` and run the notebook. It uses your `src/cnn.py` to show, for each photo,
ResNet-18's top-3 answer, what `layer1` and `layer4` see, and the Grad-CAM heat map, and saves the figure in
`results/my_field_cnn.png`. It also prints `trace(small_cnn())` so you can check your paper sketch from the activity.
Those results go into Assignment 2.1.
