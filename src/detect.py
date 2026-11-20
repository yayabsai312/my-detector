"""Detection for the third page of my system (Module 4). Follows docs/detect_spec.md.

A prediction is {"box": [x1, y1, x2, y2], "score": 0.87, "label": "bottle"} with the box in pixels.
The model runs once per photo (predict); moving the confidence slider only calls keep.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src import train as tr


def list_runs(runs="runs"):
    """One dict per finished run (runs/<name>/weights/best.pt exists), sorted by name."""
    runs = Path(runs)
    if not runs.is_dir():
        return []
    found = []
    for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        weights = run_dir / "weights" / "best.pt"
        if not weights.exists():
            continue
        summary = tr.summarize(tr.read_progress(run_dir))
        why_file = runs / f"{run_dir.name}_why.txt"
        found.append({
            "name": run_dir.name,
            "weights": str(weights),
            "mAP50": summary.get("best_mAP50") if summary else None,
            "why": why_file.read_text(encoding="utf-8").strip() if why_file.exists() else "",
        })
    return found


@lru_cache(maxsize=4)
def load_model(weights):
    """The Ultralytics model of these weights, loaded once per path."""
    from ultralytics import YOLO
    return YOLO(str(weights))


def to_preds(result, names):
    """One Ultralytics result -> list of predictions, highest score first."""
    boxes = result.boxes
    preds = [{"box": [round(float(v), 1) for v in box], "score": round(float(score), 3),
              "label": names[int(cls)]}
             for box, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
    return sorted(preds, key=lambda p: p["score"], reverse=True)


def predict(weights, rgb, imgsz=416):
    """Run the model once on an RGB image and return all predictions (conf 0.001)."""
    model = load_model(weights)
    bgr = np.ascontiguousarray(rgb[..., ::-1])          # Ultralytics reads NumPy images as BGR
    result = model(bgr, imgsz=imgsz, conf=0.001, verbose=False)[0]
    return to_preds(result, model.names)


def keep(preds, conf):
    """The predictions with score >= conf, in the same order."""
    return [p for p in preds if p["score"] >= conf]


def count_by_label(preds, classes):
    """{class: number of predictions}, every class present, in the order of classes."""
    counts = {c: 0 for c in classes}
    for p in preds:
        if p["label"] in counts:
            counts[p["label"]] += 1
    return counts


def describe(preds, conf, classes):
    """One sentence for the page."""
    if not preds:
        return f"Nothing found at confidence {conf:.2f}."
    counts = count_by_label(preds, classes)
    parts = [f"{n} {label}" for label, n in counts.items() if n > 0]
    return f"At confidence {conf:.2f}: {', '.join(parts)}."


_COLORS = [(230, 60, 60), (40, 140, 240), (50, 180, 90), (240, 170, 30), (160, 80, 200)]


def draw(rgb, preds):
    """A new RGB image with each box, its label and its score. The input is not changed.

    Lines and text scale with the size of the image, so the labels stay readable on large photos.
    """
    out = np.ascontiguousarray(rgb).copy()
    scale = max(out.shape[0], out.shape[1]) / 640            # 1.0 for a 640 px photo
    font = max(0.4, 0.5 * scale)
    line = max(2, round(2 * scale))
    text_thickness = max(1, round(scale))
    labels = sorted({p["label"] for p in preds})
    for p in preds:
        color = _COLORS[labels.index(p["label"]) % len(_COLORS)]
        x1, y1, x2, y2 = (int(round(v)) for v in p["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line)
        text = f"{p['label']} {p['score']:.2f}"
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, text_thickness)
        top = max(y1 - h - base, 0)
        cv2.rectangle(out, (x1, top), (x1 + w, top + h + base), color, -1)
        cv2.putText(out, text, (x1, top + h), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255),
                    text_thickness, cv2.LINE_AA)
    return out


def default_conf(project, fallback=0.25):
    """The threshold chosen in Assignment 3.2 (project.yaml `threshold:`), as a float."""
    value = project.get("threshold")
    return float(fallback if value is None else value)


def system_check(project, runs="runs", results="results"):
    """Which phases of the system are ready."""
    return {
        "data": (Path("data") / f"{project['name']}_yolo" / "images").exists(),
        "training": len(list_runs(runs)) > 0,
        "results": (Path(results) / "thresholds.png").exists(),
    }
