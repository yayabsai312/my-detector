"""Training for my detector (Module 3): choose the data, launch a run, read its progress.

Follows docs/train_spec.md. The training itself never runs inside the app process: `launch`
starts a separate Python, so the page keeps responding while it trains.
"""
import csv
import os
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

from src import data as dt

ROOT = Path(__file__).resolve().parents[1]
CHOICES = ("all", "half", "one_class", "large")


# --------------------------------------------------------------------------- choosing the data
def _label_file(root, split, image):
    return Path(root) / "labels" / split / f"{Path(image).stem}.txt"


def _median_area(label_path):
    """Median box area (fraction of the image) of one label file; 0.0 if there are no boxes."""
    if not Path(label_path).exists():
        return 0.0
    areas = [w * h for _, _, _, w, h in dt.read_labels(label_path)]
    return statistics.median(areas) if areas else 0.0


def _class_index(classes, class_name):
    if class_name not in classes:
        raise ValueError(f"class_name must be one of {list(classes)}, got {class_name!r}")
    return list(classes).index(class_name)


def _chosen_images(root, choice, classes, seed, class_name, min_area):
    """The training images a choice keeps (a sorted list of paths). Writes nothing."""
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}, got {choice!r}")
    images = dt._images(root, "train")
    if choice == "all":
        return images
    if choice == "half":
        return sorted(random.Random(seed).sample(images, len(images) // 2))
    if choice == "large":
        return [p for p in images if _median_area(_label_file(root, "train", p)) >= min_area]
    idx = _class_index(classes, class_name)
    return [p for p in images
            if _label_file(root, "train", p).exists()
            and any(c == idx for c, *_ in dt.read_labels(_label_file(root, "train", p)))]


def count_images(root, choice, classes, seed=42, class_name=None, min_area=0.02):
    """How many training images a choice gives, without writing anything (for the page)."""
    return len(_chosen_images(Path(root), choice, classes, seed, class_name, min_area))


def _write_one_class(root, out, chosen, idx):
    """Copies for `one_class`: kept train images, all val/test images, labels filtered and renumbered to 0."""
    keep_train = {p.name for p in chosen}
    for split in ("train", "val", "test"):
        img_dir, lab_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for img in dt._images(root, split):
            if split == "train" and img.name not in keep_train:
                continue
            shutil.copy2(img, img_dir / img.name)
            kept = []
            src = _label_file(root, split, img)
            if src.exists():
                for line in src.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and int(float(parts[0])) == idx:
                        kept.append(" ".join(["0", *parts[1:]]))
            (lab_dir / f"{img.stem}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))


def make_subset(root, choice, classes, seed=42, class_name=None, min_area=0.02):
    """Write root/subsets/<choice>/{train.txt, data.yaml}; return (yaml_path, number of training images).

    The original images and labels are never changed."""
    root = Path(root).resolve()
    chosen = _chosen_images(root, choice, classes, seed, class_name, min_area)
    out = root / "subsets" / choice
    cfg = {"path": str(root), "train": str(out / "train.txt"), "val": "images/val", "test": "images/test",
           "names": {i: n for i, n in enumerate(classes)}}
    listed = chosen
    if choice == "one_class":
        shutil.rmtree(out, ignore_errors=True)          # a folder this function made earlier
        out.mkdir(parents=True)
        _write_one_class(root, out, chosen, _class_index(classes, class_name))
        cfg.update(path=str(out), train=str(out / "train.txt"), names={0: class_name})
        listed = [out / "images" / "train" / p.name for p in chosen]
    else:
        out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(str(p) for p in listed) + "\n")
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return yaml_path, len(chosen)


# --------------------------------------------------------------------------- launching a run
def _runs_dir(runs):
    runs = Path(runs)
    return runs if runs.is_absolute() else ROOT / runs


def _weights_path(weights):
    w = Path(weights)
    if not w.is_absolute() and (ROOT / w).exists():
        return str(ROOT / w)
    return str(weights)


def train_command(yaml_path, epochs, imgsz, name, runs="runs", weights="yolo26n.pt", batch=8):
    """The command (a list of strings) that trains in a separate Python. Nothing is run here."""
    code = ("from ultralytics import YOLO; "
            f"YOLO({_weights_path(weights)!r}).train(data={str(Path(yaml_path).resolve())!r}, "
            f"epochs={int(epochs)}, imgsz={int(imgsz)}, batch={int(batch)}, seed=42, "
            f"project={str(_runs_dir(runs).resolve())!r}, name={str(name)!r}, exist_ok=True)")
    return [sys.executable, "-c", code]


def launch(yaml_path, epochs, imgsz, name, runs="runs", weights="yolo26n.pt"):
    """Start the training in the background, log to runs/<name>.log, return the process at once."""
    runs_dir = _runs_dir(runs)
    runs_dir.mkdir(parents=True, exist_ok=True)
    cmd = train_command(yaml_path, epochs, imgsz, name, runs=runs_dir, weights=weights)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")   # progress bars print non-ASCII
    with open(runs_dir / f"{name}.log", "w", encoding="utf-8") as log:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)


# --------------------------------------------------------------------------- reading a run
_COLUMNS = {"epoch": "epoch", "time": "seconds", "train/box_loss": "train_loss", "val/box_loss": "val_loss",
            "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
            "metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50_95"}


def read_progress(run_dir):
    """One dict per finished epoch from results.csv; [] if the file is not there yet."""
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            table = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]
    except OSError:
        return []
    rows = []
    for raw in table:
        try:
            row = {new: float(raw[old]) for old, new in _COLUMNS.items()}
        except (KeyError, ValueError):                  # a line still being written
            continue
        row["epoch"] = int(row["epoch"])
        rows.append(row)
    return rows


def summarize(rows):
    """{"epochs", "minutes", "best_epoch", "best_mAP50", "overfitting"}; {} when there are no rows."""
    if not rows:
        return {}
    best = max(rows, key=lambda r: r["mAP50"])          # the first epoch if there is a tie
    last_third = rows[-(len(rows) // 3):]
    overfitting = (len(rows) >= 6
                   and last_third[-1]["val_loss"] > last_third[0]["val_loss"]
                   and last_third[-1]["train_loss"] < last_third[0]["train_loss"])
    return {"epochs": len(rows), "minutes": round(rows[-1]["seconds"] / 60, 1), "best_epoch": best["epoch"],
            "best_mAP50": best["mAP50"], "overfitting": overfitting}


def estimate_minutes(seconds_per_epoch, epochs):
    """Expected minutes for a run, rounded to 1 decimal."""
    return round(seconds_per_epoch * epochs / 60, 1)


def best_weights(run_dir):
    """weights/best.pt of a run if it exists, otherwise None."""
    path = Path(run_dir) / "weights" / "best.pt"
    return path if path.exists() else None
