"""The data page of my system (Module 2). Run: python app/app.py

Tab "1 · Data": class balance, split statistics, quality checks, an image browser and the 20-image grid.
Modules 3 and 4 add more tabs to this file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from matplotlib.figure import Figure

from src import data as dt


def _balance_figure(counts, classes):
    fig = Figure(figsize=(6, max(1.8, 0.5 * len(classes) + 1.2)))
    ax = fig.subplots()
    ax.barh(classes, [counts["train"][c] for c in classes], color="#0C1422")
    ax.set_xlabel("boxes in the training split")
    ax.set_title("Class balance")
    fig.tight_layout()
    return fig


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
    return demo


if __name__ == "__main__":
    build_app(ROOT / "project.yaml").launch()
