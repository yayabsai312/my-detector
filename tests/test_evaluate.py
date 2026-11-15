"""Checks that src/evaluate.py does what docs/eval_spec.md says. Uses small invented boxes: no model is needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev


def P(box, score, label="pothole"):
    return {"box": box, "score": score, "label": label}


def G(box, label="pothole"):
    return {"box": box, "label": label}


IMAGES = {
    "one": {"preds": [P([0, 0, 10, 10], .9), P([50, 50, 60, 60], .3)], "truths": [G([0, 0, 10, 10]), G([20, 20, 30, 30])]},
    "two": {"preds": [P([0, 0, 10, 10], .8), P([30, 30, 40, 40], .6)], "truths": [G([0, 0, 10, 10])]},
    "three": {"preds": [], "truths": [G([0, 0, 10, 10])]},
}


def test_iou():
    assert ev.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1)
    assert ev.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert ev.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_match():
    r = ev.match(IMAGES["one"]["preds"], IMAGES["one"]["truths"])
    assert (len(r["tp"]), len(r["fp"]), len(r["fn"])) == (1, 1, 1)
    assert r["tp"][0]["score"] == .9


def test_match_one_truth_one_prediction():
    r = ev.match([P([0, 0, 10, 10], .9), P([0, 0, 10, 11], .8)], [G([0, 0, 10, 10])])
    assert (len(r["tp"]), len(r["fp"]), len(r["fn"])) == (1, 1, 0)      # the second box on the same object is a false alarm
    assert r["tp"][0]["score"] == .9


def test_match_needs_same_label_and_enough_iou():
    assert len(ev.match([P([0, 0, 10, 10], .9, "dog")], [G([0, 0, 10, 10], "cat")])["tp"]) == 0
    assert len(ev.match([P([0, 0, 10, 10], .9)], [G([5, 0, 15, 10])])["tp"]) == 0
    assert len(ev.match([P([0, 0, 10, 10], .9)], [G([5, 0, 15, 10])], iou_threshold=.3)["tp"]) == 1


def test_precision_recall():
    assert ev.precision_recall(3, 1, 2) == pytest.approx((.75, .6))
    assert ev.precision_recall(0, 0, 0) == (0, 0)


def test_at_threshold():
    r = ev.at_threshold(IMAGES, .25)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 2, 2) and r["conf"] == .25
    assert r["precision"] == pytest.approx(.5) and r["recall"] == pytest.approx(.5)
    r = ev.at_threshold(IMAGES, .7)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 2)


def test_threshold_table():
    t = ev.threshold_table(IMAGES, confs=(.25, .7))
    assert [r["conf"] for r in t] == [.25, .7]
    assert t[0]["fp"] > t[1]["fp"]


def test_what_changed():
    assert ev.what_changed(IMAGES, .25, .7) == {"removed_correct": 0, "removed_wrong": 2}
    assert ev.what_changed(IMAGES, .7, .85) == {"removed_correct": 1, "removed_wrong": 0}


def test_pick_examples():
    r = ev.pick_examples(IMAGES, conf=.25, n=1)
    assert r["correct"] == [] and r["wrong"] and len(r["wrong"]) == 1
    ok = {"good": {"preds": [P([0, 0, 10, 10], .9)], "truths": [G([0, 0, 10, 10])]}, "bad": IMAGES["one"]}
    r = ev.pick_examples(ok, conf=.25, n=3)
    assert r["correct"] == ["good"] and r["wrong"] == ["bad"]


def test_draw_result():
    rgb = np.zeros((70, 70, 3), np.uint8); before = rgb.copy()
    out = ev.draw_result(rgb, ev.match(IMAGES["one"]["preds"], IMAGES["one"]["truths"]))
    assert out.shape == rgb.shape and not np.array_equal(out, rgb)
    assert np.array_equal(rgb, before)
