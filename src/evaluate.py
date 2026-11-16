"""Evaluation for my detector (Module 3, Assignment 3.2): IoU, matching, precision/recall at a threshold.

Follows docs/eval_spec.md. Boxes are [x1, y1, x2, y2] in pixels. A prediction is
{"box": [...], "score": 0.9, "label": "bottle"}; a ground-truth box is {"box": [...], "label": "bottle"}.
"""
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- boxes and matching
def iou(a, b):
    """Intersection over union of two boxes, from 0 to 1. Boxes that do not touch give 0."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, truths, iou_threshold=0.5):
    """Compare one image. Predictions go from the highest score to the lowest; each takes the free
    ground-truth box of the same label with the highest IoU, if that IoU reaches the threshold."""
    taken = set()
    tp, fp = [], []
    for p in sorted(preds, key=lambda q: q["score"], reverse=True):
        best, best_iou = None, 0.0
        for i, t in enumerate(truths):
            if i in taken or t["label"] != p["label"]:
                continue
            v = iou(p["box"], t["box"])
            if v > best_iou:
                best, best_iou = i, v
        if best is not None and best_iou >= iou_threshold:
            taken.add(best)
            tp.append(p)
        else:
            fp.append(p)
    fn = [t for i, t in enumerate(truths) if i not in taken]
    return {"tp": tp, "fp": fp, "fn": fn}


def precision_recall(tp, fp, fn):
    """(precision, recall) from three counts; dividing by zero gives 0."""
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return precision, recall


# --------------------------------------------------------------------------- thresholds
def at_threshold(images, conf, iou_threshold=0.5):
    """Keep predictions with score >= conf and add up all images."""
    tp = fp = fn = 0
    for d in images.values():
        kept = [p for p in d["preds"] if p["score"] >= conf]
        r = match(kept, d["truths"], iou_threshold)
        tp += len(r["tp"])
        fp += len(r["fp"])
        fn += len(r["fn"])
    precision, recall = precision_recall(tp, fp, fn)
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def threshold_table(images, confs=(0.1, 0.25, 0.5, 0.75)):
    """One at_threshold row per value, in that order."""
    return [at_threshold(images, c) for c in confs]


def what_changed(images, conf_low, conf_high, iou_threshold=0.5):
    """Going from the low to the high threshold: how many correct and how many wrong detections disappeared."""
    low = at_threshold(images, conf_low, iou_threshold)
    high = at_threshold(images, conf_high, iou_threshold)
    return {"removed_correct": low["tp"] - high["tp"], "removed_wrong": low["fp"] - high["fp"]}


# --------------------------------------------------------------------------- the model
def read_label_boxes(label_file):
    """Read one YOLO label file as normalized boxes [(class, x1, y1, x2, y2)].
    A 5-number row is a box; a longer row is a polygon and becomes its bounding box (like Ultralytics).
    Returns (boxes, mixed): mixed is True if the file has both kinds, which Ultralytics rejects."""
    boxes, kinds = [], set()
    path = Path(label_file)
    if not path.exists():
        return boxes, False
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, v = int(float(parts[0])), [float(x) for x in parts[1:]]
        if len(parts) == 5:
            kinds.add("box")
            cx, cy, bw, bh = v
            boxes.append((c, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        else:
            kinds.add("polygon")
            xs, ys = v[0::2], v[1::2]
            boxes.append((c, min(xs), min(ys), max(xs), max(ys)))
    return boxes, len(kinds) > 1


def predict_images(weights, image_dir, label_dir, classes, imgsz=416):
    """Run the model with conf=0.001 over every image of a folder; ground truth comes from the YOLO labels.
    Polygon rows become boxes; images whose label file mixes boxes and polygons are skipped and printed."""
    from ultralytics import YOLO  # imported here so the tests need no model

    model = YOLO(str(weights))
    names = list(classes.values()) if isinstance(classes, dict) else list(classes)
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    images, skipped = {}, []
    for path in paths:
        boxes, mixed = read_label_boxes(Path(label_dir) / f"{path.stem}.txt")
        if mixed:
            skipped.append(path.name)  # boxes and polygons in one file: Ultralytics ignores this image too
            continue
        res = model.predict(str(path), conf=0.001, imgsz=imgsz, verbose=False)[0]
        h, w = res.orig_shape
        preds = []
        for box, score, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            preds.append({"box": [float(v) for v in box], "score": float(score), "label": names[int(cls)]})
        truths = [{"box": [x1 * w, y1 * h, x2 * w, y2 * h], "label": names[c]} for c, x1, y1, x2, y2 in boxes]
        images[path.name] = {"preds": preds, "truths": truths}
    if skipped:
        print(f"skipped {len(skipped)} image(s) whose labels mix boxes and polygons: {skipped}")
    print(f"{len(images)} images evaluated")
    return images


def yolo_metrics(weights, yaml_path, imgsz=416, split="test"):
    """Ultralytics val on one split: mAP50, mAP50-95 and the mAP50 of each class."""
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(yaml_path), imgsz=imgsz, split=split, verbose=False, plots=False)
    names = m.names
    per_class = {names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.ap50)}
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "per_class": per_class}


# --------------------------------------------------------------------------- looking at results
def pick_examples(images, conf=0.25, n=3):
    """Names of up to n correct images (TP and no FP or FN, most TP first) and n wrong ones (most FP + FN)."""
    correct, wrong = [], []
    for name, d in images.items():
        r = match([p for p in d["preds"] if p["score"] >= conf], d["truths"])
        t, f, m = len(r["tp"]), len(r["fp"]), len(r["fn"])
        if t > 0 and f == 0 and m == 0:
            correct.append((t, name))
        if f + m > 0:
            wrong.append((f + m, name))
    correct.sort(key=lambda x: (-x[0], x[1]))
    wrong.sort(key=lambda x: (-x[0], x[1]))
    return {"correct": [n_ for _, n_ in correct[:n]], "wrong": [n_ for _, n_ in wrong[:n]]}


def draw_result(rgb, result):
    """A new image: TP green, FP red, FN yellow (RGB), with the score on each prediction. The input is not changed."""
    out = np.ascontiguousarray(rgb).copy()
    green, red, yellow = (0, 255, 0), (255, 0, 0), (255, 255, 0)

    def rect(box, color):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        return x1, y1

    for t in result["fn"]:
        rect(t["box"], yellow)
    for group, color in ((result["tp"], green), (result["fp"], red)):
        for p in group:
            x, y = rect(p["box"], color)
            cv2.putText(out, f"{p['score']:.2f}", (x, max(y - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
