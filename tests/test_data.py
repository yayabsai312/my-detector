"""Checks that src/data.py does what docs/data_spec.md says. Builds tiny fake datasets in a temporary folder."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as dt

CLASSES = ["cat", "dog"]


def _img(path: Path, w=100, h=50, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), value, np.uint8))


def _voc(path: Path, filename, objects, w=100, h=50):
    objs = "".join(f"<object><name>{n}</name><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin><xmax>{b[2]}</xmax>"
                   f"<ymax>{b[3]}</ymax></bndbox></object>" for n, b in objects)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<annotation><filename>{filename}</filename><size><width>{w}</width><height>{h}</height>"
                    f"<depth>3</depth></size>{objs}</annotation>")


@pytest.fixture
def voc_src(tmp_path):
    src = tmp_path / "raw"
    for i in range(20):
        _img(src / "images" / f"img{i:02d}.png", value=10 * i)
        objs = [("cat", [10, 10, 30, 40])] + ([("dog", [50, 5, 90, 45])] if i % 2 else []) + ([("bird", [0, 0, 5, 5])] if i == 3 else [])
        _voc(src / "annotations" / f"img{i:02d}.xml", f"img{i:02d}.png", objs)
    return src


def test_load_project_checks_fields(tmp_path):
    good = tmp_path / "p.yaml"
    good.write_text("name: pets\nclasses: [cat, dog]\nformat: voc\nsource:\n  folder: raw\n")
    assert dt.load_project(good)["classes"] == CLASSES
    bad = tmp_path / "b.yaml"
    bad.write_text("name: pets\nclasses: []\nformat: voc\nsource:\n  folder: raw\n")
    with pytest.raises(ValueError):
        dt.load_project(bad)
    bad.write_text("name: pets\nclasses: [cat]\nformat: jpg\nsource:\n  folder: raw\n")
    with pytest.raises(ValueError):
        dt.load_project(bad)


def test_box_conversion_both_ways():
    cx, cy, w, h = dt.pixels_to_yolo([10, 10, 30, 40], 100, 50)
    assert (round(cx, 4), round(cy, 4), round(w, 4), round(h, 4)) == (0.2, 0.5, 0.2, 0.6)
    assert np.allclose(dt.yolo_to_pixels(cx, cy, w, h, 100, 50), [10, 10, 30, 40])


def test_boxes_outside_the_image_are_clipped():
    cx, cy, w, h = dt.pixels_to_yolo([-10, 0, 50, 60], 100, 50)        # sticks out left and bottom
    assert 0 <= cx - w / 2 and cy + h / 2 <= 1.0 + 1e-9


def test_read_voc(voc_src):
    rec = dt.read_voc(voc_src / "annotations" / "img01.xml")
    assert rec["filename"] == "img01.png" and rec["width"] == 100 and rec["height"] == 50
    assert rec["objects"] == [("cat", [10.0, 10.0, 30.0, 40.0]), ("dog", [50.0, 5.0, 90.0, 45.0])]


def test_read_coco_turns_xywh_into_corners(tmp_path):
    j = tmp_path / "ann.json"
    j.write_text(json.dumps({"images": [{"id": 7, "file_name": "a.png", "width": 100, "height": 50}],
                             "annotations": [{"id": 1, "image_id": 7, "category_id": 3, "bbox": [10, 10, 20, 30]}],
                             "categories": [{"id": 3, "name": "dog"}]}))
    assert dt.read_coco(j) == [{"filename": "a.png", "width": 100, "height": 50, "objects": [("dog", [10, 10, 30, 40])]}]


def test_split_is_deterministic_disjoint_and_complete():
    names = [f"n{i}" for i in range(100)]
    a, b = dt.split_names(names, seed=42), dt.split_names(list(reversed(names)), seed=42)
    assert a == b                                                     # same input set -> same split, whatever the order
    assert [len(a[s]) for s in ("train", "val", "test")] == [70, 15, 15]
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set(names)
    assert not set(a["train"]) & set(a["test"])
    assert [len(v) for v in dt.split_names([f"p{i}" for i in range(665)]).values()] == [465, 100, 100]   # cut points: int(0.70 n), int(0.85 n)


def test_convert_voc_to_yolo(voc_src, tmp_path):
    out = tmp_path / "yolo"
    rep = dt.convert_to_yolo([dt.read_voc(x) for x in sorted((voc_src / "annotations").glob("*.xml"))],
                             voc_src, out, CLASSES)
    assert sum(rep["images"].values()) == 20
    assert rep["skipped_objects"]["bird"] == 1                        # not one of our classes: skipped and counted
    lbl = next((out / "labels").rglob("img01.txt")).read_text().split()
    assert lbl[:5] == ["0", "0.200000", "0.500000", "0.200000", "0.600000"]
    assert (out / "data.yaml").exists()


def test_prepare_voc_from_project(voc_src, tmp_path):
    root = dt.prepare({"name": "pets", "classes": CLASSES, "format": "voc", "source": {"folder": str(voc_src)}},
                      voc_src, tmp_path / "pets_yolo")
    assert len(list((root / "images").rglob("*.png"))) == 20


def test_class_counts_and_stats(voc_src, tmp_path):
    root = dt.prepare({"name": "pets", "classes": CLASSES, "format": "voc", "source": {"folder": str(voc_src)}},
                      voc_src, tmp_path / "pets_yolo")
    cc = dt.class_counts(root, CLASSES)
    assert sum(cc[s]["cat"] for s in cc) == 20 and sum(cc[s]["dog"] for s in cc) == 10
    st = dt.split_stats(root)
    assert sum(st[s]["images"] for s in st) == 20 and sum(st[s]["boxes"] for s in st) == 30


def test_review_finds_problems(tmp_path):
    root = tmp_path / "ds"
    _img(root / "images" / "train" / "a.png"); _img(root / "images" / "train" / "b.png", value=7)
    _img(root / "images" / "test" / "a_copy.png")                      # same pixels as train/a: a leak
    (root / "labels" / "train").mkdir(parents=True); (root / "labels" / "test").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 1.4 0.2\n5 0.5 0.5 0.01 0.01\n")
    (root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "labels" / "test" / "a_copy.txt").write_text("")
    c = dt.review(root, n_classes=2)["counts"]
    assert c["images_without_label"] == 1 and c["labels_without_image"] == 1 and c["empty_labels"] == 1
    assert c["out_of_range_boxes"] == 1 and c["unknown_class_ids"] == 1 and c["tiny_boxes_lt_0.1pct"] == 1
    assert c["duplicates_across_splits"] == 1


def test_draw_boxes_returns_a_copy():
    img = np.zeros((50, 100, 3), np.uint8)
    out = dt.draw_boxes(img, [[10, 10, 30, 40]], color=(255, 0, 0))
    assert img.sum() == 0 and out[10, 20].tolist() == [255, 0, 0]


def test_sample_grid_has_five_columns(voc_src, tmp_path):
    root = dt.prepare({"name": "pets", "classes": CLASSES, "format": "voc", "source": {"folder": str(voc_src)}},
                      voc_src, tmp_path / "pets_yolo")
    g = dt.sample_grid(root, "train", n=10, tile=(64, 48), classes=CLASSES)
    assert g.shape == (2 * 48, 5 * 64, 3)


def test_report_is_written(voc_src, tmp_path):
    root = dt.prepare({"name": "pets", "classes": CLASSES, "format": "voc", "source": {"folder": str(voc_src)}},
                      voc_src, tmp_path / "pets_yolo")
    md = dt.write_report(root, CLASSES, tmp_path / "results" / "dataset_review.md")
    text = md.read_text()
    assert "| cat |" in text and "| dog |" in text and (tmp_path / "results" / "dataset_sample_grid.jpg").exists()


def test_app_builds(voc_src, tmp_path, monkeypatch):
    root = dt.prepare({"name": "pets", "classes": CLASSES, "format": "voc", "source": {"folder": str(voc_src)}},
                      voc_src, tmp_path / "data" / "pets_yolo")
    (tmp_path / "project.yaml").write_text(f"name: pets\nclasses: [cat, dog]\nformat: voc\nsource:\n  folder: {voc_src}\n")
    monkeypatch.chdir(tmp_path)
    import gradio as gr
    from app import app as myapp
    assert isinstance(myapp.build_app("project.yaml"), gr.Blocks)
