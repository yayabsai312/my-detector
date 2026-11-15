# Specification of `src/evaluate.py` (Module 3, Assignment 3.2)

Give this file to your AI assistant and ask it for a plan. `python -m pytest -q tests/test_evaluate.py` checks your work: **10 tests must pass**.
The tests use small invented boxes, so they need no model.

Boxes are always `[x1, y1, x2, y2]` in pixels. A prediction is `{"box": [...], "score": 0.9, "label": "pothole"}`. A ground-truth box is `{"box": [...], "label": "pothole"}`.

## Functions of `src/evaluate.py`

| Function | What it does |
|---|---|
| `iou(a, b)` | Intersection over union of two boxes, from 0 to 1. Boxes that do not touch give 0. |
| `match(preds, truths, iou_threshold=0.5)` | Compares one image. Goes through the predictions from the highest score to the lowest. Each one takes the free ground-truth box of the same label with the highest IoU, if that IoU reaches the threshold. Returns `{"tp": [...], "fp": [...], "fn": [...]}`: matched predictions, predictions left over, and ground-truth boxes nobody found. |
| `precision_recall(tp, fp, fn)` | Takes three counts and returns `(precision, recall)`. Dividing by zero gives 0. |
| `at_threshold(images, conf, iou_threshold=0.5)` | `images` is a dict `{name: {"preds": [...], "truths": [...]}}`. Keeps predictions with `score >= conf` and adds up all images. Returns `{"conf", "tp", "fp", "fn", "precision", "recall"}`. |
| `threshold_table(images, confs=(0.1, 0.25, 0.5, 0.75))` | One `at_threshold` row per value, in that order. |
| `what_changed(images, conf_low, conf_high, iou_threshold=0.5)` | Going from the low to the high threshold, returns `{"removed_correct": n, "removed_wrong": n}`: how many correct and how many wrong detections disappeared. |
| `predict_images(weights, image_dir, label_dir, classes, imgsz=416)` | Runs the model with `conf=0.001` over every image of a folder and returns the `images` dict. Ground truth comes from the YOLO labels, converted to pixels. |
| `yolo_metrics(weights, yaml_path, imgsz=416, split="test")` | Runs Ultralytics `val` on that split and returns `{"mAP50", "mAP50_95", "per_class": {name: mAP50}}`. |
| `pick_examples(images, conf=0.25, n=3)` | Returns `{"correct": [...], "wrong": [...]}`, each with up to `n` image names. `correct` are the images with TP and no FP or FN, most TP first. `wrong` are the images with the most FP + FN. |
| `draw_result(rgb, result)` | A new image from `match`'s result: TP in green, FP in red, FN in yellow, with the score on each prediction. Never changes the input. |

## Then use it: `notebooks/03_my_results.ipynb`
It uses your `src/evaluate.py` with the model you trained. It prints mAP50, mAP50-95 and the table at four thresholds, and saves `results/thresholds.png` and `results/examples.png` (three correct, three wrong). Those results go into Assignment 3.2.
