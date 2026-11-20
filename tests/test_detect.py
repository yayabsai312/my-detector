"""Checks that src/detect.py does what docs/detect_spec.md says. Uses a fake model and fake runs: nothing is trained."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detect as dc

CLASSES = ["stop", "speedlimit"]
CSV = ("epoch, time, train/box_loss, val/box_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
       "1, 10, 2.0, 2.1, 0.3, 0.2, 0.40, 0.20\n"
       "2, 20, 1.5, 1.8, 0.5, 0.4, 0.61, 0.33\n"
       "3, 30, 1.2, 1.7, 0.6, 0.5, 0.55, 0.30\n")


def P(box, score, label="stop"):
    return {"box": box, "score": score, "label": label}


PREDS = [P([10, 10, 50, 50], .9), P([60, 10, 90, 40], .5, "speedlimit"), P([5, 60, 30, 90], .2)]


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.array(xyxy, float), np.array(conf, float), np.array(cls, float)


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes([[1.04, 2, 3, 4], [10, 20, 30, 40.06]], [0.30001, 0.9], [1, 0])


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [FakeResult()]

    predict = __call__
    names = {0: "stop", 1: "speedlimit"}


def make_run(runs, name, finished=True, csv=True, why=None):
    d = runs / name
    (d / "weights").mkdir(parents=True)
    if finished:
        (d / "weights" / "best.pt").write_bytes(b"fake")
    if csv:
        (d / "results.csv").write_text(CSV)
    if why is not None:
        (runs / f"{name}_why.txt").write_text(why)


def test_list_runs(tmp_path):
    make_run(tmp_path, "b_all", why="all photos, to compare")
    make_run(tmp_path, "a_half", csv=False)
    make_run(tmp_path, "c_unfinished", finished=False)
    runs = dc.list_runs(tmp_path)
    assert [r["name"] for r in runs] == ["a_half", "b_all"]          # sorted, unfinished skipped
    assert runs[1]["mAP50"] == pytest.approx(0.61)
    assert runs[1]["why"] == "all photos, to compare"
    assert runs[0]["mAP50"] is None and runs[0]["why"] == ""
    assert runs[1]["weights"].endswith(str(Path("b_all") / "weights" / "best.pt"))


def test_list_runs_empty(tmp_path):
    assert dc.list_runs(tmp_path / "missing") == []


def test_to_preds():
    preds = dc.to_preds(FakeResult(), {0: "stop", 1: "speedlimit"})
    assert [p["label"] for p in preds] == ["stop", "speedlimit"]       # highest score first
    assert preds[0]["score"] == 0.9 and preds[1]["score"] == 0.3
    assert preds[0]["box"] == [10.0, 20.0, 30.0, 40.1] and preds[1]["box"][0] == 1.0


def test_predict_runs_once_on_bgr(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(dc, "load_model", lambda weights: model)
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[..., 0] = 255                                               # pure red in RGB
    preds = dc.predict("x.pt", rgb)
    assert len(model.calls) == 1 and len(preds) == 2
    image, kwargs = model.calls[0]
    assert image[0, 0].tolist() == [0, 0, 255]                      # the model received BGR
    assert kwargs.get("conf") == 0.001
    assert rgb[0, 0].tolist() == [255, 0, 0]                        # the input was not changed


def test_keep():
    assert [p["score"] for p in dc.keep(PREDS, 0.5)] == [.9, .5]
    assert dc.keep(PREDS, 0.95) == []


def test_count_by_label():
    assert dc.count_by_label(PREDS, CLASSES) == {"stop": 2, "speedlimit": 1}
    assert list(dc.count_by_label([], CLASSES)) == CLASSES


def test_describe():
    assert dc.describe(dc.keep(PREDS, .4), .4, CLASSES) == "At confidence 0.40: 1 stop, 1 speedlimit."
    assert dc.describe([P([0, 0, 1, 1], .9)], .25, CLASSES) == "At confidence 0.25: 1 stop."
    assert dc.describe([], .5, CLASSES) == "Nothing found at confidence 0.50."


def test_draw_new_image():
    rgb = np.zeros((100, 100, 3), np.uint8)
    out = dc.draw(rgb, PREDS)
    assert out.shape == rgb.shape and out.sum() > 0
    assert rgb.sum() == 0                                          # the input image is untouched


def test_default_conf():
    assert dc.default_conf({"name": "x", "threshold": 0.4}) == pytest.approx(0.4)
    assert dc.default_conf({"name": "x"}) == pytest.approx(0.25)
    assert isinstance(dc.default_conf({"name": "x", "threshold": "0.6"}), float)


def test_system_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = {"name": "signs", "classes": CLASSES}
    assert dc.system_check(project) == {"data": False, "training": False, "results": False}
    (tmp_path / "data" / "signs_yolo" / "images").mkdir(parents=True)
    make_run(tmp_path / "runs", "all_30")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "thresholds.png").write_bytes(b"png")
    assert dc.system_check(project) == {"data": True, "training": True, "results": True}
