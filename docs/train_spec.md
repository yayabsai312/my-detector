# Specification of `src/train.py` and the page "2 · Training" (Module 3, Assignment 3.1)

Give this file to your AI assistant and ask it for a plan. `python -m pytest -q tests/test_train.py` checks your work: **10 tests must pass**.
The tests build a tiny fake dataset and fake training results, so they train nothing and run in seconds.

## What the module does
It chooses the data a detector trains on, launches the training without freezing the app, and reads its progress.

## Functions of `src/train.py`

| Function | What it does |
|---|---|
| `make_subset(root, choice, classes, seed=42, class_name=None, min_area=0.02)` | Writes a training list and a YAML file, and returns `(yaml_path, n_images)`. `root` is the YOLO folder of the project. `choice` is one of: `"all"` (every training image), `"half"` (a random half, using `seed`), `"one_class"` (only images with `class_name`; their labels keep only that class, renumbered to 0, and the YAML has one class), `"large"` (only images whose median box covers at least `min_area` of the image). Any other value raises `ValueError`. Files go to `root/subsets/<choice>/`: a list `train.txt` with absolute image paths, and `data.yaml`. For `one_class`, write copies inside that folder with filtered images and labels for train, and with the same images and filtered labels for val and test, so that the metrics stay right and the original data is never changed. For the other choices, validation and test are the ones in `root`. |
| `train_command(yaml_path, epochs, imgsz, name, runs="runs", weights="yolo26n.pt", batch=8)` | Returns the command as a list of strings to run the training in a separate process with the current Python (`sys.executable`). It must train with Ultralytics using exactly these values, with `seed=42`, `project=` the absolute path of `runs`, `name=name`, `exist_ok=True`. Nothing is run here. |
| `launch(yaml_path, epochs, imgsz, name, runs="runs", weights="yolo26n.pt")` | Starts `train_command(...)` with `subprocess.Popen`, sends its output to `runs/<name>.log`, and returns the process right away. The app must keep responding while it trains. |
| `read_progress(run_dir)` | Reads `results.csv` from `run_dir` and returns one dict per finished epoch: `epoch`, `seconds` (the total time so far, as Ultralytics writes it), `train_loss`, `val_loss` (`train/box_loss` and `val/box_loss`), `precision`, `recall`, `mAP50`, `mAP50_95`. Column names may have spaces around them. If the file does not exist yet, return `[]`. |
| `summarize(rows)` | From those rows, returns `{"epochs", "minutes", "best_epoch", "best_mAP50", "overfitting"}`. `minutes` is the total time. `best_epoch` is the one with the highest `mAP50`. `overfitting` is `True` when, over the last third of the epochs, `val_loss` went up while `train_loss` went down. With fewer than 6 epochs it is always `False`: that is too little to tell. With no rows, return `{}`. |
| `estimate_minutes(seconds_per_epoch, epochs)` | Expected minutes for a run, rounded to 1 decimal. |
| `best_weights(run_dir)` | Path of `weights/best.pt` inside `run_dir` if it exists; otherwise `None`. |

## The page `2 · Training` in `app/app.py`
Add a second tab to the app you built in Module 2. It must let the user:
1. Choose the training data (`all`, `half`, `one_class`, `large`) and see how many images that means.
2. Write **why** they chose it (a text box). Save it with the run in `runs/<name>_why.txt` before the training starts.
3. Choose the epochs (30 by default) and the image size (416 by default), and press **Start training**.
4. See the progress while it trains: a table and a chart of the losses and mAP50, refreshed every few seconds, without blocking the page.

## Details that matter
- No new packages: Ultralytics, Gradio and Matplotlib are already installed.
- Never change the files in `data/<name>_yolo/images` or `labels`.
- Do not run a real training to check your code: use the tests. The real training comes after they pass.
