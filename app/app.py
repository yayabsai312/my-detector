"""The data page of my system (Module 2). Run: python app/app.py

Tab "1 · Data": class balance, split statistics, quality checks, an image browser and the 20-image grid.
Tab "2 · Training" (Module 3): choose the data, say why, start a training and watch it.
Module 4 adds more tabs to this file.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from matplotlib.figure import Figure

from src import data as dt
from src import train as tr


def _balance_figure(counts, classes):
    fig = Figure(figsize=(6, max(1.8, 0.5 * len(classes) + 1.2)))
    ax = fig.subplots()
    ax.barh(classes, [counts["train"][c] for c in classes], color="#0C1422")
    ax.set_xlabel("boxes in the training split")
    ax.set_title("Class balance")
    fig.tight_layout()
    return fig


PROGRESS_HEADERS = ["epoch", "minutes", "train_loss", "val_loss", "precision", "recall", "mAP50", "mAP50_95"]


def _progress_figure(rows):
    fig = Figure(figsize=(7, 3))
    left, right = fig.subplots(1, 2)
    if rows:
        epochs = [r["epoch"] for r in rows]
        left.plot(epochs, [r["train_loss"] for r in rows], label="train")
        left.plot(epochs, [r["val_loss"] for r in rows], label="val")
        left.legend()
        right.plot(epochs, [r["mAP50"] for r in rows], color="#0C1422")
    left.set_title("box loss")
    right.set_title("mAP50")
    for ax in (left, right):
        ax.set_xlabel("epoch")
    fig.tight_layout()
    return fig


def _log_tail(runs, name, lines=15):
    log = runs / f"{name}.log"
    if not log.exists():
        return ""
    return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _training_tab(root, classes, base):
    """The widgets of the tab '2 · Training'. Call it inside a gr.Tab."""
    runs = base / "runs"
    procs = {}                                   # run name -> the process started from this page

    def how_many(choice, class_name):
        try:
            n = tr.count_images(root, choice, classes, class_name=class_name)
        except ValueError as error:
            return f"**{error}**"
        total = tr.count_images(root, "all", classes)
        return f"**{n}** of {total} training images"

    gr.Markdown("Choose the data, say why, then start. A real training can take an hour or more: "
                "try 2 epochs first to see how long one epoch takes.")
    with gr.Row():
        choice = gr.Radio(list(tr.CHOICES), value="all", label="training data")
        class_name = gr.Dropdown(classes, value=classes[0], label="class (only used by one_class)")
    count = gr.Markdown(how_many("all", classes[0]))
    why = gr.Textbox(label="Why I chose this data", lines=3)
    with gr.Row():
        run_name = gr.Textbox(value="run1", label="run name (letters, digits, - and _)")
        epochs = gr.Number(value=30, precision=0, label="epochs")
        imgsz = gr.Number(value=416, precision=0, label="image size")
    start_button = gr.Button("Start training", variant="primary")
    message = gr.Markdown()
    status = gr.Markdown()
    table = gr.Dataframe(headers=PROGRESS_HEADERS, value=[], interactive=False, label="finished epochs")
    plot = gr.Plot(label="progress")

    for control in (choice, class_name):
        control.change(how_many, [choice, class_name], count)

    def start(choice_value, class_value, why_text, name, n_epochs, size):
        name = (name or "").strip()
        if not (why_text or "").strip():
            return "**Write why you chose this data first.**"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            return "**Run name: use letters, digits, - and _ only.**"
        if (runs / name).exists() or (name in procs and procs[name].poll() is None):
            return f"**`runs/{name}` already exists.** Choose another run name."
        try:
            yaml_path, n_images = tr.make_subset(root, choice_value, classes, class_name=class_value)
        except ValueError as error:
            return f"**{error}**"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{name}_why.txt").write_text(
            f"data: {choice_value}\nimages: {n_images}\n\n{why_text.strip()}\n", encoding="utf-8")
        procs[name] = tr.launch(yaml_path, int(n_epochs), int(size), name, runs=runs)
        return f"Started `{name}` on {n_images} images. The table fills in after the first epoch."

    def refresh(name, n_epochs):
        name = (name or "").strip()
        rows = tr.read_progress(runs / name) if name else []
        table_rows = [[r["epoch"], round(r["seconds"] / 60, 1), round(r["train_loss"], 3), round(r["val_loss"], 3),
                       round(r["precision"], 3), round(r["recall"], 3), round(r["mAP50"], 3),
                       round(r["mAP50_95"], 3)] for r in rows]
        proc = procs.get(name)
        if proc is None:
            text = "No training started from this page for this run name." if not rows else \
                f"Results found for `{name}`: {len(rows)} epochs."
        elif proc.poll() is None:
            text = f"Training `{name}`: {len(rows)} of {int(n_epochs)} epochs done."
            if rows:
                per_epoch = rows[-1]["seconds"] / len(rows)
                text += f" About {tr.estimate_minutes(per_epoch, int(n_epochs))} minutes in total."
        elif proc.returncode == 0:
            s = tr.summarize(rows)
            text = (f"Finished `{name}`: {s.get('epochs', 0)} epochs, {s.get('minutes', 0)} minutes, best mAP50 "
                    f"{s.get('best_mAP50', 0):.3f} at epoch {s.get('best_epoch', '-')}."
                    + (" The validation loss went up at the end: possible overfitting." if s.get("overfitting") else ""))
        else:
            text = f"Training `{name}` stopped with an error (code {proc.returncode}). End of the log:\n\n```\n{_log_tail(runs, name)}\n```"
        return text, table_rows, _progress_figure(rows)

    start_button.click(start, [choice, class_name, why, run_name, epochs, imgsz], message)
    timer = gr.Timer(5)
    timer.tick(refresh, [run_name, epochs], [status, table, plot])


def build_app(project_path="project.yaml"):
    """Return the Gradio Blocks of the data page."""
    project_path = Path(project_path)
    project = dt.load_project(project_path)
    classes = project["classes"]
    root = project_path.resolve().parent / "data" / f"{project['name']}_yolo"

    with gr.Blocks(title=project["name"]) as demo:
        gr.Markdown(f"# {project['name']}")
        with gr.Tab("1 · Data"):
            if not root.exists():
                gr.Markdown("**No dataset yet.** Run `notebooks/02_my_dataset.ipynb` first, then reload this page.")
                return demo

            counts = dt.class_counts(root, classes)
            stats = dt.split_stats(root)
            rv = dt.review(root, len(classes))

            gr.Markdown(f"Classes: {', '.join(classes)}  ·  folder: `{root.name}`")
            gr.Plot(_balance_figure(counts, classes), label="Class balance")

            gr.Markdown("### Boxes per class")
            gr.Dataframe([[c] + [counts[s][c] for s in dt.SPLITS] + [sum(counts[s][c] for s in dt.SPLITS)]
                          for c in classes],
                         headers=["class", *dt.SPLITS, "total"], interactive=False)
            gr.Markdown("### Split statistics")
            gr.Dataframe([[s, *stats[s].values()] for s in dt.SPLITS],
                         headers=["split", *next(iter(stats.values())).keys()], interactive=False)
            gr.Markdown("### Quality checks")
            gr.Dataframe([[k, v, ", ".join(rv["examples"][k][:3])] for k, v in rv["counts"].items()],
                         headers=["check", "count", "first examples"], interactive=False)

            gr.Markdown("### Duplicate images across splits")
            pairs = dt.find_duplicate_pairs(root)
            if pairs:
                gr.Markdown(f"{len(pairs)} pairs of identical images sit in different splits.")
                gr.Dataframe([[p["split_a"], p["file_a"], p["split_b"], p["file_b"]] for p in pairs],
                             headers=["split", "file", "split (copy)", "file (copy)"], interactive=False)
            else:
                gr.Markdown("No duplicates found.")

            gr.Markdown("### Image browser")
            biggest = max([len(dt._images(root, s)) for s in dt.SPLITS] + [1])
            with gr.Row():
                split = gr.Radio(list(dt.SPLITS), value="train", label="split")
                number = gr.Slider(1, biggest, value=1, step=1, label="image number")
            browser_img = gr.Image(label="image with its boxes", type="numpy")
            browser_info = gr.Markdown()

            def browse(split_name, i):
                images = dt._images(root, split_name)
                if not images:
                    return None, f"No images in {split_name}."
                path = images[min(int(i), len(images)) - 1]
                found = dt.boxes_for(path)
                listed = ", ".join(classes[c] if 0 <= c < len(classes) else str(c) for c, _ in found) or "none"
                return dt.show(path, classes), f"`{path.name}` · {len(found)} boxes · classes: {listed}"

            for control in (split, number):
                control.change(browse, [split, number], [browser_img, browser_info])
            demo.load(browse, [split, number], [browser_img, browser_info])

            gr.Markdown("### 20 random training images")
            with gr.Row():
                seed = gr.Number(value=0, precision=0, label="seed")
                button = gr.Button("Show 20 images")
            grid_img = gr.Image(label="grid", type="numpy")
            button.click(lambda s: dt.sample_grid(root, "train", n=20, seed=int(s), classes=classes),
                         seed, grid_img)

        with gr.Tab("2 · Training"):
            if not root.exists():
                gr.Markdown("**No dataset yet.** Run `notebooks/02_my_dataset.ipynb` first, then reload this page.")
            else:
                _training_tab(root, classes, project_path.resolve().parent)
    return demo


if __name__ == "__main__":
    build_app(ROOT / "project.yaml").launch()
