"""Checks that src/train.py does what docs/train_spec.md says. Uses a tiny fake dataset and fake results: nothing is trained."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train

CLASSES = ["cat", "dog"]
HEADER = "epoch, time, train/box_loss, val/box_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"


@pytest.fixture
def root(tmp_path):
    """Six training images: two classes, some with small boxes and some with large ones."""
    r = tmp_path / "pets_yolo"
    rows = {"a": ["0 .5 .5 .5 .5"], "b": ["0 .5 .5 .05 .05"], "c": ["1 .5 .5 .5 .5", "0 .2 .2 .4 .4"],
            "d": ["1 .5 .5 .05 .05"], "e": ["0 .5 .5 .6 .6"], "f": ["1 .5 .5 .6 .6"]}
    for s in ("train", "val", "test"):
        (r / "images" / s).mkdir(parents=True); (r / "labels" / s).mkdir(parents=True)
    for n, lines in rows.items():
        cv2.imwrite(str(r / "images" / "train" / f"{n}.png"), np.zeros((20, 20, 3), np.uint8))
        (r / "labels" / "train" / f"{n}.txt").write_text("\n".join(lines) + "\n")
    cv2.imwrite(str(r / "images" / "val" / "v.png"), np.zeros((20, 20, 3), np.uint8))
    (r / "labels" / "val" / "v.txt").write_text("0 .5 .5 .5 .5\n1 .5 .5 .5 .5\n")
    return r


def _train_images(y):
    cfg = yaml.safe_load(Path(y).read_text())
    base = Path(cfg["path"]) if cfg.get("path") else Path(y).parent
    t = Path(cfg["train"]); t = t if t.is_absolute() else base / t
    return cfg, [l for l in t.read_text().splitlines() if l.strip()]


def _results(run, rows):
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.csv").write_text(HEADER + "\n".join(", ".join(str(x) for x in r) for r in rows) + "\n")


def test_all_and_half(root):
    y, n = train.make_subset(root, "all", CLASSES)
    assert n == 6 and len(_train_images(y)[1]) == 6
    y2, n2 = train.make_subset(root, "half", CLASSES)
    assert n2 == 3 and len(_train_images(y2)[1]) == 3
    assert train.make_subset(root, "half", CLASSES)[1] == 3
    assert _train_images(train.make_subset(root, "half", CLASSES)[0])[1] == _train_images(y2)[1]    # same seed, same photos


def test_one_class(root):
    y, n = train.make_subset(root, "one_class", CLASSES, class_name="dog")
    assert n == 3                                                     # c, d and f have a dog
    cfg, imgs = _train_images(y)
    names = cfg["names"]
    assert (list(names.values()) if isinstance(names, dict) else names) == ["dog"]
    for p in imgs:
        lab = Path(p).parents[2] / "labels" / "train" / (Path(p).stem + ".txt")
        lines = [l for l in lab.read_text().splitlines() if l.strip()]
        assert lines and all(l.split()[0] == "0" for l in lines)      # renumbered to 0, only dogs
    assert (root / "labels" / "train" / "c.txt").read_text().startswith("1 ")    # the original is not changed
    val = Path(cfg["path"]) / cfg["val"] / "v.png" if cfg.get("path") else Path(y).parent / cfg["val"] / "v.png"
    assert val.exists()
    lab = val.parents[2] / "labels" / "val" / (val.stem + ".txt")
    assert lab.read_text().strip() == "0 .5 .5 .5 .5"                 # validation keeps only the dog, renumbered


def test_large(root):
    assert train.make_subset(root, "large", CLASSES)[1] == 4          # a, c, e and f


def test_bad_choice(root):
    with pytest.raises(ValueError):
        train.make_subset(root, "everything", CLASSES)


def test_train_command(tmp_path):
    cmd = train.train_command(tmp_path / "d.yaml", 7, 320, "run1", runs=tmp_path / "runs")
    assert isinstance(cmd, list) and all(isinstance(x, str) for x in cmd)
    assert cmd[0] == sys.executable
    text = " ".join(cmd)
    for piece in ("7", "320", "run1", "d.yaml"):
        assert piece in text


def test_read_progress(tmp_path):
    _results(tmp_path, [[1, 60, 2.0, 2.1, .1, .2, .3, .1], [2, 121, 1.5, 1.9, .2, .3, .4, .2]])
    rows = train.read_progress(tmp_path)
    assert len(rows) == 2 and rows[1]["epoch"] == 2
    assert rows[0]["train_loss"] == 2.0 and rows[1]["val_loss"] == 1.9 and rows[1]["mAP50"] == .4 and rows[1]["mAP50_95"] == .2
    assert train.read_progress(tmp_path / "nothing") == []


def test_summarize():
    rows = [{"epoch": i + 1, "seconds": (i + 1) * 60.0, "train_loss": 3 - i * .1, "val_loss": 2 - i * .1, "mAP50": .5 + i * .01, "mAP50_95": .2, "precision": .5, "recall": .5} for i in range(9)]
    s = train.summarize(rows)
    assert s["epochs"] == 9 and s["best_epoch"] == 9 and s["best_mAP50"] == pytest.approx(.58)
    assert s["minutes"] == pytest.approx(9.0) and s["overfitting"] is False
    assert train.summarize([]) == {}


def test_overfitting_flag():
    rows = [{"epoch": i + 1, "seconds": float(i), "train_loss": 3 - i * .1, "val_loss": 2 if i < 6 else 2 + (i - 5) * .2, "mAP50": .5, "mAP50_95": .2, "precision": .5, "recall": .5} for i in range(9)]
    assert train.summarize(rows)["overfitting"] is True
    assert train.summarize(rows[:5])["overfitting"] is False           # too few epochs to tell


def test_estimate_minutes():
    assert train.estimate_minutes(150, 30) == 75.0
    assert train.estimate_minutes(10, 1) == 0.2


def test_best_weights(tmp_path):
    assert train.best_weights(tmp_path) is None
    (tmp_path / "weights").mkdir(); (tmp_path / "weights" / "best.pt").write_text("x")
    assert Path(train.best_weights(tmp_path)).name == "best.pt"
