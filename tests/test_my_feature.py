"""My own improvement: find_duplicate_pairs() reports identical images that sit in different splits.

The fake dataset is built so that every right answer is known in advance.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as dt


def _img(root, split, name, value):
    folder = root / "images" / split
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / name), np.full((50, 100, 3), value, np.uint8))


def test_duplicate_pairs_across_splits(tmp_path):
    root = tmp_path / "ds"
    _img(root, "train", "a.png", 10)
    _img(root, "val", "a_copy.png", 10)            # same pixels as train/a.png, different file name
    _img(root, "test", "b.png", 200)               # different image: must not be flagged
    _img(root, "train", "c.png", 90)
    _img(root, "train", "c_copy.png", 90)          # twin inside ONE split: not a leak, must not be flagged
    for split, name in (("train", "d.png"), ("val", "d_val.png"), ("test", "d_test.png")):
        _img(root, split, name, 140)               # one image in all three splits: three pairs

    pairs = dt.find_duplicate_pairs(root)

    assert pairs == [
        {"split_a": "train", "file_a": "a.png", "split_b": "val", "file_b": "a_copy.png"},
        {"split_a": "train", "file_a": "d.png", "split_b": "val", "file_b": "d_val.png"},
        {"split_a": "train", "file_a": "d.png", "split_b": "test", "file_b": "d_test.png"},
        {"split_a": "val", "file_a": "d_val.png", "split_b": "test", "file_b": "d_test.png"},
    ]
    flagged = {p["file_a"] for p in pairs} | {p["file_b"] for p in pairs}
    assert "b.png" not in flagged                  # the different image
    assert "c.png" not in flagged and "c_copy.png" not in flagged      # same-split twins
