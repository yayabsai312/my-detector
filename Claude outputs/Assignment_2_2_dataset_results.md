# Assignment 2.2 · Dataset results

MAI 630 Computer Vision · Atlantis University · Yaya Sow

## Project

My system detects reusable and disposable water bottles in photos of gym water-station areas, so that a facility manager can decide whether to install a water vending machine or refill stations.

## Dataset (stand-in for Assignment 2.2)

| Item | Value |
|---|---|
| Source | Kaggle: `siddharthkumarsah/plastic-bottles-image-dataset` (https://www.kaggle.com/datasets/siddharthkumarsah/plastic-bottles-image-dataset) |
| Licence | Attribution 4.0 International (CC BY 4.0) |
| Original annotation format | YOLO bounding boxes, already split train / valid / test |
| Size on Kaggle | 1.71 GB |
| Classes | 1: `bottle` |
| Converted to | `data/bottles_yolo` (YOLO folder, validation folder renamed `valid` to `val`) |
| Note | One-class stand-in. It has no reusable / disposable split and is not gym photos. Own gym photos with two classes come in Modules 3 and 4. |

Code: `src/data.py` and `app/app.py` (built with Claude Code from `docs/data_spec.md`). Test suite: 37 passed (13 Module 1, 10 `src/cnn.py`, 14 `src/data.py`).

## Boxes per class

| name | train | val | test | total |
|---|---|---|---|---|
| bottle | 3186 | 1550 | 792 | 5528 |

## Split statistics

| split | images | boxes | boxes per image | median box area % | boxes < 1% of image |
|---|---|---|---|---|---|
| train | 2177 | 3186 | 1.46 | 16.43 | 280 |
| val | 1174 | 1550 | 1.32 | 15.35 | 123 |
| test | 648 | 792 | 1.22 | 14.18 | 64 |

Every image has a label file (2,177 + 1,174 + 648 = 3,999 images).

## Quality checks

| check | count | first examples |
|---|---|---|
| images_without_label | 0 | |
| labels_without_image | 0 | |
| empty_labels | 0 | |
| out_of_range_boxes | 25 | test/bottle-208-_JPG.rf.735985d6..., test/bottle-280-_JPG.rf.95e4c5fa..., test/bottle-334-_JPG.rf.e2925e5d... |
| unknown_class_ids | 0 | |
| tiny_boxes_lt_0.1pct | 116 | test/25_jpg.rf.442c800d..., test/56_jpg.rf.edae0a1d..., test/bottle-208-_JPG.rf.735985d6... |
| duplicates_across_splits | 16 | train/empty-plastic-bottle-lying-around-park-ground-...146271701 = val/waterbottle211_..., train/empty-plastic-water-bottle-500x500_... = val/waterbottle229_..., train/plastic234_... = val/plasticbottle-440-_... |

## 20-image grid

File: `results/dataset_sample_grid.jpg` (20 random training images, seed 0, boxes drawn, file name in the corner of each tile). Full report: `results/dataset_review.md`.

## My observations (to write myself)

- **Boxes that do not match what I see** (image name, what is wrong):
- **Class balance:**
- **The two things most likely to make this dataset hard:**
  1.
  2.
