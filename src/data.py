"""Module 2: turn any object-detection dataset into one YOLO folder with a fixed split, then measure it.

Follows docs/data_spec.md. A YOLO label line is `class_id cx cy w h`, with the box centre and size
as fractions (0-1) of the image.
"""
import hashlib
import json
import random
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

SPLITS = ("train", "val", "test")
FORMATS = ("voc", "coco", "yolo")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# folder names a downloaded YOLO dataset may use for each split
_SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
_PALETTE = [(255, 215, 0), (0, 200, 255), (255, 80, 80), (80, 220, 100), (200, 120, 255), (255, 150, 0)]


# --------------------------------------------------------------------------- project and download
def load_project(path="project.yaml"):
    """Read project.yaml and check it."""
    with open(path, encoding="utf-8") as f:
        project = yaml.safe_load(f) or {}
    if not project.get("name"):
        raise ValueError("project.yaml needs a name")
    if not project.get("classes"):
        raise ValueError("project.yaml needs at least one class")
    if project.get("format") not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {project.get('format')!r}")
    source = project.get("source") or {}
    if not (source.get("kaggle") or source.get("folder")):
        raise ValueError("project.yaml needs source.kaggle or source.folder")
    return project


def download(project):
    """Return the folder of the raw dataset (Kaggle download, or the local folder)."""
    source = project["source"]
    if source.get("kaggle"):
        import kagglehub
        return Path(kagglehub.dataset_download(source["kaggle"]))
    folder = Path(source["folder"])
    if not folder.exists():
        raise FileNotFoundError(f"dataset folder not found: {folder}")
    return folder


# --------------------------------------------------------------------------- box conversion
def pixels_to_yolo(box, width, height):
    """[x1, y1, x2, y2] in pixels -> (cx, cy, w, h) as fractions. The box is clipped to the image first."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(float(x1), width)), max(0.0, min(float(x2), width))
    y1, y2 = max(0.0, min(float(y1), height)), max(0.0, min(float(y2), height))
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height)


def yolo_to_pixels(cx, cy, w, h, width, height):
    """(cx, cy, w, h) as fractions -> [x1, y1, x2, y2] in pixels."""
    return [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]


# --------------------------------------------------------------------------- reading annotations
def read_voc(xml_path):
    """One PASCAL VOC .xml -> {"filename", "width", "height", "objects": [(class_name, [x1, y1, x2, y2]), ...]}."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    objects = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        box = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        objects.append((obj.find("name").text.strip(), box))
    return {"filename": (root.findtext("filename") or Path(xml_path).stem).strip(),
            "width": int(float(size.findtext("width"))) if size is not None else 0,
            "height": int(float(size.findtext("height"))) if size is not None else 0,
            "objects": objects}


def read_coco(json_path):
    """A COCO .json -> a list of records with the same shape as read_voc."""
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)
    names = {c["id"]: c["name"] for c in coco.get("categories", [])}
    records = {}
    for im in coco.get("images", []):
        records[im["id"]] = {"filename": Path(im["file_name"]).name, "width": im["width"],
                             "height": im["height"], "objects": []}
    for ann in coco.get("annotations", []):
        x, y, w, h = ann["bbox"]
        if ann["image_id"] in records:
            records[ann["image_id"]]["objects"].append((names[ann["category_id"]], [x, y, x + w, y + h]))
    return list(records.values())


def read_labels(label_path):
    """A YOLO .txt -> list of (class_id, cx, cy, w, h). An empty file gives []."""
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(float(parts[0])), *[float(v) for v in parts[1:5]]))
    return boxes


# --------------------------------------------------------------------------- building the dataset
def split_names(names, seed=42, ratios=(0.70, 0.15, 0.15)):
    """Sort, shuffle with random.Random(seed), then cut at int(0.70 n) and int(0.85 n)."""
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n = len(names)
    a = int(ratios[0] * n)
    b = int(round(ratios[0] + ratios[1], 10) * n)
    return {"train": names[:a], "val": names[a:b], "test": names[b:]}


def write_data_yaml(root, classes):
    """Write root/data.yaml (paths + class names), the file YOLO reads in Module 3."""
    root = Path(root)
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: c for i, c in enumerate(classes)}}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _image_size(path):
    img = _read_bgr(path)
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def convert_to_yolo(records, images_dir, out, classes, seed=42):
    """Split the records, copy each image and write its label file. Returns a report dict."""
    out, images_dir = Path(out), Path(images_dir)
    index = {}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            index.setdefault(p.name, p)
    by_name = {r["filename"]: r for r in records}
    splits = split_names(list(by_name), seed=seed)
    report = {"images": {s: 0 for s in SPLITS}, "skipped_objects": Counter(), "missing_images": 0}
    for split, names in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for name in names:
            src = index.get(name)
            if src is None:
                report["missing_images"] += 1
                continue
            rec = by_name[name]
            width, height = rec["width"], rec["height"]
            if not width or not height:
                width, height = _image_size(src)
            lines = []
            for cls, box in rec["objects"]:
                if cls not in classes:
                    report["skipped_objects"][cls] += 1
                    continue
                cx, cy, w, h = pixels_to_yolo(box, width, height)
                lines.append(f"{classes.index(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            shutil.copy2(src, out / "images" / split / name)
            (out / "labels" / split / f"{Path(name).stem}.txt").write_text("".join(l + "\n" for l in lines))
            report["images"][split] += 1
    write_data_yaml(out, classes)
    return report


def _find_split_dir(src, split):
    for alias in _SPLIT_ALIASES[split]:
        found = sorted(p for p in Path(src).rglob(alias) if p.is_dir())
        if found:
            return found[0]
    return None


def _copy_yolo(src, out):
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        sd = _find_split_dir(src, split)
        if sd is None:
            continue
        imgs = sd / "images" if (sd / "images").is_dir() else sd
        lbls = sd / "labels" if (sd / "labels").is_dir() else sd
        for p in sorted(imgs.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT:
                shutil.copy2(p, out / "images" / split / p.name)
        for p in sorted(lbls.iterdir()):
            if p.is_file() and p.suffix.lower() == ".txt":
                shutil.copy2(p, out / "labels" / split / p.name)


def prepare(project, src, out=None):
    """Raw folder -> data/<name>_yolo (or `out`). Rewrites the generated folders each time."""
    src = Path(src)
    out = Path(out) if out else Path("data") / f"{project['name']}_yolo"
    for sub in ("images", "labels"):                  # start clean: only our own generated folders
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    out.mkdir(parents=True, exist_ok=True)
    classes = project["classes"]
    fmt = project["format"]
    if fmt == "voc":
        records = [read_voc(x) for x in sorted(src.rglob("*.xml"))]
        convert_to_yolo(records, src, out, classes)
    elif fmt == "coco":
        if not project.get("coco_json"):
            raise ValueError("format coco needs coco_json in project.yaml")
        convert_to_yolo(read_coco(src / project["coco_json"]), src, out, classes)
    elif fmt == "yolo":
        _copy_yolo(src, out)
        write_data_yaml(out, classes)
    else:
        raise ValueError(f"format must be one of {FORMATS}")
    return out


# --------------------------------------------------------------------------- measuring
def _images(root, split):
    folder = Path(root) / "images" / split
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if folder.is_dir() else []


def _labels(root, split):
    folder = Path(root) / "labels" / split
    return sorted(folder.glob("*.txt")) if folder.is_dir() else []


def class_counts(root, classes):
    """Boxes per class per split, listing every class (0 if none)."""
    counts = {s: {c: 0 for c in classes} for s in SPLITS}
    for split in SPLITS:
        for lp in _labels(root, split):
            for cid, *_ in read_labels(lp):
                if 0 <= cid < len(classes):
                    counts[split][classes[cid]] += 1
    return counts


def split_stats(root):
    """Per split: images, boxes, boxes_per_image, median_box_area_pct, small_boxes_lt_1pct."""
    stats = {}
    for split in SPLITS:
        n_images = len(_images(root, split))
        areas = [w * h for lp in _labels(root, split) for _, _, _, w, h in read_labels(lp)]
        stats[split] = {
            "images": n_images,
            "boxes": len(areas),
            "boxes_per_image": round(len(areas) / n_images, 2) if n_images else 0.0,
            "median_box_area_pct": round(statistics.median(areas) * 100, 2) if areas else 0.0,
            "small_boxes_lt_1pct": sum(a < 0.01 for a in areas),
        }
    return stats


def review(root, n_classes):
    """Quality checks. Returns {"counts": {...}, "examples": {...}} with the seven checks of the spec."""
    keys = ["images_without_label", "labels_without_image", "empty_labels", "out_of_range_boxes",
            "unknown_class_ids", "tiny_boxes_lt_0.1pct", "duplicates_across_splits"]
    examples = {k: [] for k in keys}
    hashes = {}
    for split in SPLITS:
        images, labels = _images(root, split), _labels(root, split)
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        for p in images:
            if p.stem not in label_stems:
                examples["images_without_label"].append(f"{split}/{p.stem}")
            hashes.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), []).append((split, p.stem))
        for lp in labels:
            tag = f"{split}/{lp.stem}"
            if lp.stem not in image_stems:
                examples["labels_without_image"].append(tag)
            boxes = read_labels(lp)
            if not boxes:
                examples["empty_labels"].append(tag)
            for cid, cx, cy, w, h in boxes:
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    examples["out_of_range_boxes"].append(tag)
                if cid < 0 or cid >= n_classes:
                    examples["unknown_class_ids"].append(tag)
                if w * h < 0.001:
                    examples["tiny_boxes_lt_0.1pct"].append(tag)
    for group in hashes.values():
        if len({s for s, _ in group}) > 1:
            examples["duplicates_across_splits"].append(" = ".join(f"{s}/{n}" for s, n in group))
    counts = {}
    for k in keys:
        counts[k] = len(examples[k])          # the count is per offending box / file; examples list each file once
        examples[k] = sorted(dict.fromkeys(examples[k]))
    return {"counts": counts, "examples": examples}


# --------------------------------------------------------------------------- looking at images
def _read_bgr(path):
    data = np.fromfile(str(path), dtype=np.uint8)            # works with any characters in the path
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def _read_rgb(path):
    bgr = _read_bgr(path)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_boxes(rgb, boxes, labels=None, color=(255, 215, 0), thickness=2):
    """Draw pixel boxes on an RGB image. Returns a copy. `color` is one colour or one per box."""
    out = np.ascontiguousarray(rgb).copy()
    for i, box in enumerate(boxes):
        c = color[i] if isinstance(color[0], (tuple, list)) else color
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), tuple(int(v) for v in c), thickness)
        if labels is not None and i < len(labels):
            cv2.putText(out, str(labels[i]), (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        tuple(int(v) for v in c), 1, cv2.LINE_AA)
    return out


def boxes_for(image_path):
    """Ground truth of one image of the YOLO folder: [(class_id, [x1, y1, x2, y2]), ...] in pixels."""
    image_path = Path(image_path)
    split_dir = image_path.parent
    label = split_dir.parent.parent / "labels" / split_dir.name / f"{image_path.stem}.txt"
    if not label.exists():
        return []
    height, width = _read_rgb(image_path).shape[:2]
    return [(cid, yolo_to_pixels(cx, cy, w, h, width, height)) for cid, cx, cy, w, h in read_labels(label)]


def show(image_path, classes=None):
    """The image as RGB with its boxes and class names, one colour per class."""
    rgb = _read_rgb(image_path)
    found = boxes_for(image_path)
    colors = [_PALETTE[cid % len(_PALETTE)] for cid, _ in found]
    names = [classes[cid] if classes and 0 <= cid < len(classes) else str(cid) for cid, _ in found]
    if not found:
        return rgb.copy()
    return draw_boxes(rgb, [b for _, b in found], names, colors)


def sample_grid(root, split="train", n=20, seed=0, classes=None, tile=(256, 192)):
    """n random images with their boxes, in a grid of 5 columns; file names in the top-left corners."""
    images = _images(root, split)
    chosen = random.Random(seed).sample(images, min(n, len(images)))
    tw, th = tile
    rows = max(1, -(-len(chosen) // 5))
    grid = np.zeros((rows * th, 5 * tw, 3), np.uint8)
    for i, path in enumerate(chosen):
        img = cv2.resize(show(path, classes), (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (min(tw, 9 * len(path.name) + 6), 16), (0, 0, 0), -1)
        cv2.putText(img, path.name, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, 5)
        grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = img
    return grid


def write_report(root, classes, out="results/dataset_review.md"):
    """Markdown with boxes per class, split statistics and quality checks; saves the grid next to it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = class_counts(root, classes)
    stats = split_stats(root)
    rv = review(root, len(classes))
    lines = [f"# Dataset review: {Path(root).name}", "", "## Boxes per class", "",
             "| name | train | val | test | total |", "|---|---|---|---|---|"]
    for c in classes:
        row = [counts[s][c] for s in SPLITS]
        lines.append(f"| {c} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    lines += ["", "## Split statistics", "",
              "| split | images | boxes | boxes per image | median box area % | boxes < 1% of image |",
              "|---|---|---|---|---|---|"]
    for s in SPLITS:
        st = stats[s]
        lines.append(f"| {s} | {st['images']} | {st['boxes']} | {st['boxes_per_image']} | "
                     f"{st['median_box_area_pct']} | {st['small_boxes_lt_1pct']} |")
    lines += ["", "## Quality checks", "", "| check | count | first examples |", "|---|---|---|"]
    for k, v in rv["counts"].items():
        lines.append(f"| {k} | {v} | {', '.join(rv['examples'][k][:3])} |")
    if _images(root, "train"):
        grid = sample_grid(root, "train", n=20, seed=0, classes=classes)
        grid_path = out.parent / "dataset_sample_grid.jpg"
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        buf.tofile(str(grid_path))
        lines += ["", "## 20-image grid (train, seed 0)", "", f"![grid]({grid_path.name})"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
