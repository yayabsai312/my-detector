# Specification of `src/data.py` and the data page `app/app.py` (Module 2)

Give this file to your AI assistant and ask it to implement `src/data.py` first, then `app/app.py`.
`python -m pytest -q tests/test_data.py` checks your work: **14 tests must pass**. The whole suite,
`python -m pytest -q tests`, must give **37 passed** (13 from Module 1, 10 for `src/cnn.py`, 14 for `src/data.py`) plus your own test.

## What the module does
It turns any object-detection dataset into **one YOLO folder with a fixed split**, then measures it before we trust it.

```
data/<name>_yolo/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/     one .txt per image
  data.yaml                                     paths + class names (used by YOLO in Module 3)
```

A YOLO label file has one line per box: `class_id cx cy w h`, where `class_id` is the position of the class in
`project.yaml` (0, 1, 2...) and `cx cy w h` are the box centre, width and height **as fractions of the image (0–1)**.

## `project.yaml` (you fill it in; it describes YOUR project)
```yaml
name: potholes                      # short, no spaces: the folder becomes data/potholes_yolo
classes: [pothole]                  # the objects you detect, in a fixed order
format: voc                         # voc | coco | yolo  (how the downloaded annotations are written)
source:
  kaggle: andrewmvd/pothole-detection   # OR   folder: path/to/my/dataset
# coco_json: annotations.json       # only for format: coco (path inside the dataset folder)
```

## Functions of `src/data.py`

| Function | What it does |
|---|---|
| `load_project(path="project.yaml")` | Read the YAML and return a dict. `ValueError` if there is no `name`, no classes, a format outside `voc / coco / yolo`, or no `source.kaggle` / `source.folder`. |
| `download(project)` | Return the folder of the raw dataset: `kagglehub.dataset_download(...)` for Kaggle, or the local folder (`FileNotFoundError` if it does not exist). |
| `pixels_to_yolo(box, width, height)` | `[x1, y1, x2, y2]` in pixels → `(cx, cy, w, h)` fractions. **Clip the box to the image first.** |
| `yolo_to_pixels(cx, cy, w, h, width, height)` | The inverse: back to `[x1, y1, x2, y2]` in pixels. |
| `read_voc(xml_path)` | One PASCAL VOC `.xml` → `{"filename", "width", "height", "objects": [(class_name, [x1, y1, x2, y2]), ...]}` (floats). |
| `read_coco(json_path)` | A COCO `.json` → a list of records with the same shape. COCO boxes are `[x, y, width, height]`: convert them to corners. `filename` is the file name only. |
| `split_names(names, seed=42, ratios=(0.70, 0.15, 0.15))` | **Sort**, shuffle with `random.Random(seed)`, then cut at `int(0.70 * n)` and `int(0.85 * n)` into `{"train", "val", "test"}` (665 names → 465 / 100 / 100). The same set of names always gives the same split. |
| `convert_to_yolo(records, images_dir, out, classes, seed=42)` | Split the records, copy each image (found anywhere under `images_dir`) and write its label file; write `data.yaml`. Objects whose class is not in `classes` are **skipped and counted**. Returns `{"images": {split: n}, "skipped_objects": Counter, "missing_images": n}`. |
| `write_data_yaml(root, classes)` | Write `root/data.yaml` with `path` (absolute), `train: images/train`, `val: images/val`, `test: images/test`, `names: {0: ..., 1: ...}`. |
| `prepare(project, src, out=None)` | Raw folder → `data/<name>_yolo` (or `out`). `voc`: read every `.xml` under `src`. `coco`: read `src/<coco_json>`. `yolo`: the dataset is already split; copy it and write `data.yaml`. Returns the root. |
| `read_labels(label_path)` | A YOLO `.txt` → list of `(class_id, cx, cy, w, h)`. An empty file gives `[]`. |
| `class_counts(root, classes)` | Boxes per class per split: `{"train": {"pothole": 1213}, "val": {...}, "test": {...}}`. |
| `split_stats(root)` | Per split: `images`, `boxes`, `boxes_per_image` (2 decimals), `median_box_area_pct` (box area as % of the image, 2 decimals), `small_boxes_lt_1pct`. |
| `review(root, n_classes)` | Quality checks → `{"counts": {...}, "examples": {...}}` with exactly these keys: `images_without_label`, `labels_without_image`, `empty_labels`, `out_of_range_boxes` (cx, cy outside 0–1 or w, h outside (0, 1]), `unknown_class_ids`, `tiny_boxes_lt_0.1pct` (w·h < 0.001), `duplicates_across_splits` (same file bytes, md5, in two splits). |
| `draw_boxes(rgb, boxes, labels=None, color=(255, 215, 0), thickness=2)` | Draw pixel boxes on an RGB image. **Return a copy**; never change the input. |
| `boxes_for(image_path)` | Ground truth of one image of the YOLO folder: `[(class_id, [x1, y1, x2, y2]), ...]` in pixels. |
| `show(image_path, classes=None)` | The image as RGB with its boxes and class names, one colour per class. |
| `sample_grid(root, split="train", n=20, seed=0, classes=None, tile=(256, 192))` | `n` random images (`random.Random(seed).sample`) with their boxes, resized to `tile` (width, height), in a grid of **5 columns** (fill the last row with black tiles). Write each image's file name in the top-left corner of its tile, so the grid can be checked image by image. |
| `write_report(root, classes, out="results/dataset_review.md")` | Markdown with boxes per class (a row per class, `| name | train | val | test | total |`), split statistics and quality checks; save the grid as `dataset_sample_grid.jpg` next to it. |

Also define `SPLITS = ("train", "val", "test")`, `FORMATS = ("voc", "coco", "yolo")` and
`IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")` (compare extensions in lower case).

## The data page: `app/app.py`
A Gradio app. `build_app(project_path="project.yaml")` returns a `gr.Blocks`; `python app/app.py` launches it.
It reads `project.yaml`, finds `data/<name>_yolo` and shows, in a tab called **"1 · Data"**:
1. the class balance (a bar chart of training boxes per class),
2. the split statistics and the quality checks as tables,
3. an image browser: choose a split and an image number, see the image with its boxes and the list of classes,
4. a button that shows the 20-image grid, with a seed you can change.

If the dataset folder does not exist yet, the tab says so instead of crashing. Modules 3 and 4 add more tabs to this file.

## Details that matter (the tests check them)
- Use `yaml.safe_load` / `yaml.safe_dump` (PyYAML is already installed with Ultralytics). No new packages.
- Write label lines with 6 decimals: `0 0.200000 0.500000 0.200000 0.600000`.
- An image whose objects are all skipped still gets an empty label file (it is a valid image with no objects).
- `class_counts` and the report must list **every** class of the project, even with 0 boxes.
