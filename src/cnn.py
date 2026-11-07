"""Module 2: a small CNN, and three ways to look inside ResNet-18.

Follows docs/cnn_spec.md. Shapes printed by `trace` are (height, width, channels),
the way we write them in class; PyTorch stores (channels, height, width).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
STAGES = ("layer1", "layer2", "layer3", "layer4")


def output_size(n, kernel, padding=0, stride=1):
    """Side of the map after a convolution or a pooling."""
    return (n + 2 * padding - kernel) // stride + 1


def small_cnn(n_classes=2):
    """The small CNN of the class activity."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, n_classes),
    )


def trace(model, input_shape=(3, 64, 64)):
    """Pass a zero image through each layer; one row per layer."""
    c, h, w = input_shape
    rows = [{"layer": "input", "shape": (h, w, c), "weights": 0}]
    x = torch.zeros(1, c, h, w)
    model.eval()
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if x.dim() == 4:
                shape = (x.shape[2], x.shape[3], x.shape[1])
            else:
                shape = tuple(x.shape[1:])
            weight = getattr(layer, "weight", None)
            rows.append({
                "layer": layer.__class__.__name__,
                "shape": shape,
                "weights": int(weight.numel()) if weight is not None else 0,
            })
    return rows


def count_parameters(model):
    """Every number the model learns, weights and biases."""
    return sum(p.numel() for p in model.parameters())


def load_resnet(pretrained=True):
    """Return (model, preprocess, categories) for ResNet-18 in eval mode."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights if pretrained else None)
    model.eval()
    categories = list(weights.meta["categories"])
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def preprocess(rgb):
        # exactly 224 x 224, no crop: the heat map must line up with the whole photo
        small = cv2.resize(np.ascontiguousarray(rgb), (224, 224), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
        return normalize(x)

    return model, preprocess, categories


def classify(model, preprocess, categories, rgb, top=5):
    """The `top` most probable classes as (name, probability), highest first."""
    with torch.no_grad():
        probs = F.softmax(model(preprocess(rgb).unsqueeze(0)), dim=1)[0]
    p, idx = probs.topk(top)
    return [(categories[int(i)], float(v)) for v, i in zip(p, idx)]


def feature_maps(model, preprocess, rgb, layer="layer1"):
    """Maps produced by one stage of ResNet-18: (channels, height, width)."""
    if layer not in STAGES:
        raise ValueError(f"layer must be one of {STAGES}, got {layer!r}")
    store = {}
    handle = getattr(model, layer).register_forward_hook(
        lambda module, inputs, output: store.update(maps=output.detach()))
    try:
        with torch.no_grad():
            model(preprocess(rgb).unsqueeze(0))
    finally:
        handle.remove()
    return store["maps"][0].cpu().numpy().astype(np.float32)


def gradcam(model, preprocess, rgb, layer="layer4", class_index=None):
    """Grad-CAM: where the network looked. Returns (heat, class_index, probability)."""
    store = {}
    target = getattr(model, layer)

    def forward_hook(module, inputs, output):
        # hook 1 keeps the activations; hook 2 (on the tensor) keeps their gradient
        store["act"] = output
        store["grad_handle"] = output.register_hook(lambda g: store.update(grad=g))

    h1 = target.register_forward_hook(forward_hook)
    try:
        x = preprocess(rgb).unsqueeze(0).requires_grad_(True)
        model.zero_grad()
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        k = int(logits[0].argmax()) if class_index is None else int(class_index)
        logits[0, k].backward()
        act = store["act"].detach()[0]            # (C, h, w)
        grad = store["grad"].detach()[0]          # (C, h, w)
        weights = grad.mean(dim=(1, 2))           # one weight per map
        cam = F.relu((weights[:, None, None] * act).sum(0)).numpy()
        p = float(probs[k].detach())
    finally:
        h1.remove()
        if "grad_handle" in store:
            store["grad_handle"].remove()
        model.zero_grad()
    height, width = rgb.shape[:2]
    cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    cam = cam - cam.min()
    peak = cam.max()
    heat = (cam / peak if peak > 0 else cam).astype(np.float32)
    return heat, k, p


def overlay(rgb, heat, alpha=0.45):
    """The photo with the heat map on top (jet: red = looked the most). Returns a new array."""
    colors = (colormaps["jet"](np.clip(heat, 0, 1))[..., :3] * 255).astype(np.float32)
    mixed = (1 - alpha) * rgb.astype(np.float32) + alpha * colors
    return np.clip(mixed, 0, 255).astype(np.uint8)
