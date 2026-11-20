# Specification of `src/detect.py` and the page "3 · Detection & results" (Module 4, Assignment 4.1)

Give this file to your AI assistant and ask it for a plan. `python -m pytest -q tests/test_detect.py` checks your work: **10 tests must pass**.
The tests use a fake model and fake runs, so they need no training and run in seconds.

A prediction has the same shape as in Module 3: `{"box": [x1, y1, x2, y2], "score": 0.87, "label": "pothole"}`, with the box in pixels.

## What the module does
It takes the model you trained on the page "2 · Training", runs it on a photo, and keeps only the boxes above the confidence you choose.
It also tells the app which phases of your system are ready, so that each page uses what the previous one produced.

## Functions of `src/detect.py`

| Function | What it does |
|---|---|
| `list_runs(runs="runs")` | One dict per finished run, sorted by name: `{"name", "weights", "mAP50", "why"}`. A run is finished when `runs/<name>/weights/best.pt` exists; skip the others. `weights` is that path as a string. `mAP50` is the highest validation mAP50 in its `results.csv` (use `read_progress` and `summarize` from `src/train.py`), or `None` if there is no `results.csv`. `why` is the text of `runs/<name>_why.txt`, or `""`. |
| `load_model(weights)` | Returns an Ultralytics `YOLO` model. Keep it in a cache by path, so moving a slider never loads the same model twice. |
| `to_preds(result, names)` | Converts one Ultralytics result into a list of predictions, from the highest score to the lowest. `result.boxes.xyxy`, `result.boxes.conf` and `result.boxes.cls` are arrays (call `.tolist()` on them); `names` maps a class number to its name. Boxes are rounded to 1 decimal, scores to 3. |
| `predict(weights, rgb, imgsz=416)` | Runs the model **once** on an RGB image, with `conf=0.001` and `verbose=False`, and returns `to_preds(...)`. Ultralytics reads NumPy images as BGR: convert the RGB image to BGR before calling the model. |
| `keep(preds, conf)` | The predictions with `score >= conf`, in the same order. Moving the confidence slider only calls this: the model is not run again. |
| `count_by_label(preds, classes)` | A dict with every class in `classes`, in that order, and how many predictions have that label (0 if none). |
| `describe(preds, conf, classes)` | One sentence for the page. With predictions: `"At confidence 0.25: 3 pothole."` (for several classes, separated by commas: `"At confidence 0.40: 2 stop, 1 speedlimit."`, only classes with at least one box). With none: `"Nothing found at confidence 0.50."`. The confidence always has 2 decimals. |
| `draw(rgb, preds)` | A new image with each box and its label and score (for example `pothole 0.87`). Never changes the input image. |
| `default_conf(project, fallback=0.25)` | The confidence threshold you chose in Assignment 3.2, read from `project["threshold"]`; if `project.yaml` has no `threshold`, `fallback`. Always a float. |
| `system_check(project, runs="runs", results="results")` | Which phases of the system are ready: `{"data": ..., "training": ..., "results": ...}`. `data` is True when `data/<name>_yolo/images` exists; `training` when `list_runs(runs)` is not empty; `results` when `results/thresholds.png` exists. |

## The page `3 · Detection & results` in `app/app.py`
Add a third tab to your app. It must let the user:
1. See at the top of the app which phases are ready (`system_check`), with what to do when one is missing.
2. Choose a finished run (`list_runs`), and see its mAP50 and **why** it was trained.
3. Upload a photo, or choose one from the test split, and press **Detect**: the model runs once (`predict`).
4. Move a confidence slider that starts at `default_conf(project)`: the boxes, the sentence of `describe` and a table of `count_by_label` change without running the model again (`keep`).
5. See the results of Module 3: `results/thresholds.png`, `results/examples.png` and your `docs/experiment.md`, if they exist.

## Details that matter
- No new packages: Ultralytics, Gradio, OpenCV and Matplotlib are already installed.
- Write the threshold you chose in Assignment 3.2 in `project.yaml`, for example `threshold: 0.40`. You decide it; the code only reads it.
- If there is no finished run, the page says so and explains how to train one. It must not crash.
- Do not train to check your code: use the tests, and then try the page with the run you trained in Module 3.
