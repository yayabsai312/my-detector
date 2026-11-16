"""My own check of how src/evaluate.py reads YOLO label files (invented files, no model)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev


def test_box_polygon_and_mixed(tmp_path):
    box = tmp_path / "box.txt"; box.write_text("0 0.5 0.5 0.2 0.4\n")
    boxes, mixed = ev.read_label_boxes(box)
    assert not mixed and boxes[0] == pytest.approx((0, 0.4, 0.3, 0.6, 0.7))

    poly = tmp_path / "poly.txt"; poly.write_text("0 0.1 0.2 0.5 0.2 0.3 0.8\n")
    boxes, mixed = ev.read_label_boxes(poly)
    assert not mixed and boxes[0] == pytest.approx((0, 0.1, 0.2, 0.5, 0.8))

    both = tmp_path / "both.txt"; both.write_text("0 0.5 0.5 0.2 0.4\n0 0.1 0.2 0.5 0.2 0.3 0.8\n")
    assert ev.read_label_boxes(both)[1] is True
    assert ev.read_label_boxes(tmp_path / "missing.txt") == ([], False)
