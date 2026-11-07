"""Checks that src/cnn.py does what docs/cnn_spec.md says. Uses ResNet-18 without its trained weights: nothing is downloaded."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cnn


@pytest.fixture(scope="module")
def resnet():
    torch.manual_seed(0)
    return cnn.load_resnet(pretrained=False)


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)      # not square on purpose


def test_output_size_formula():
    assert cnn.output_size(64, 3, padding=1, stride=1) == 64        # padding keeps the size
    assert cnn.output_size(5, 3) == 3                               # no padding: the window does not fit at the edges
    assert cnn.output_size(5, 3, padding=1) == 5
    assert cnn.output_size(5, 3, stride=2) == 2
    assert cnn.output_size(224, 7, padding=3, stride=2) == 112      # first layer of ResNet-18
    assert cnn.output_size(64, 2, stride=2) == 32                   # max pooling 2 x 2


def test_trace_small_cnn_sizes():
    rows = cnn.trace(cnn.small_cnn(), (3, 64, 64))
    assert rows[0] == {"layer": "input", "shape": (64, 64, 3), "weights": 0}
    assert [r["shape"] for r in rows] == [(64, 64, 3), (64, 64, 8), (64, 64, 8), (32, 32, 8), (32, 32, 16), (32, 32, 16),
                                          (16, 16, 16), (1, 1, 16), (16,), (2,)]
    assert [r["layer"] for r in rows[1:4]] == ["Conv2d", "ReLU", "MaxPool2d"]


def test_trace_small_cnn_weights():
    rows = cnn.trace(cnn.small_cnn(), (3, 64, 64))
    assert [r["weights"] for r in rows if r["weights"]] == [216, 1152, 32]
    assert sum(r["weights"] for r in rows) == 1400
    assert cnn.trace(cnn.small_cnn(n_classes=4))[-1]["shape"] == (4,)


def test_count_parameters():
    assert cnn.count_parameters(cnn.small_cnn()) == 1400 + 8 + 16 + 2     # weights + biases
    model, _, _ = cnn.load_resnet(pretrained=False)
    assert cnn.count_parameters(model) == 11_689_512


def test_load_resnet(resnet, photo):
    model, preprocess, categories = resnet
    assert not model.training
    assert len(categories) == 1000
    x = preprocess(photo)
    assert tuple(x.shape) == (3, 224, 224) and x.dtype == torch.float32


def test_classify(resnet, photo):
    model, preprocess, categories = resnet
    out = cnn.classify(model, preprocess, categories, photo, top=5)
    assert len(out) == 5
    probs = [p for _, p in out]
    assert probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs) and sum(probs) <= 1 + 1e-6
    assert all(label in categories for label, _ in out)
    assert len(cnn.classify(model, preprocess, categories, photo, top=3)) == 3


def test_feature_maps_shapes(resnet, photo):
    model, preprocess, _ = resnet
    assert cnn.feature_maps(model, preprocess, photo, "layer1").shape == (64, 56, 56)
    assert cnn.feature_maps(model, preprocess, photo, "layer4").shape == (512, 7, 7)
    with pytest.raises(ValueError):
        cnn.feature_maps(model, preprocess, photo, "layer9")


def test_gradcam(resnet, photo):
    model, preprocess, categories = resnet
    before = photo.copy()
    heat, k, p = cnn.gradcam(model, preprocess, photo)
    assert heat.shape == photo.shape[:2]                            # same size as the photo, not 224 x 224
    assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
    assert 0 <= k < len(categories) and 0 <= p <= 1
    assert k == categories.index(cnn.classify(model, preprocess, categories, photo, top=1)[0][0])
    assert np.array_equal(photo, before)


def test_gradcam_chosen_class(resnet, photo):
    model, preprocess, _ = resnet
    heat, k, _ = cnn.gradcam(model, preprocess, photo, class_index=7)
    assert k == 7 and heat.shape == (120, 200)


def test_overlay(photo):
    before = photo.copy()
    heat = np.zeros(photo.shape[:2], np.float32); heat[:, 100:] = 1.0
    out = cnn.overlay(photo, heat)
    assert out.shape == photo.shape and out.dtype == np.uint8
    assert np.array_equal(photo, before)
    assert not np.array_equal(out, photo)
