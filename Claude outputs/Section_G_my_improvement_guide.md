# Section G · My own improvement (Assignment 2.2, steps 21 to 23)

This file gives you everything you need to design and request your improvement. **You write `docs/my_feature.md` yourself, before asking Claude for anything.** Designing the improvement is your part of the work.

## 1. What the task is

Choose **one** thing your data page or your data module should do that it does not do yet. Claude builds it, and a test in `tests/test_my_feature.py` proves it works. The suite goes from 37 passed to **38 passed or more** (37 plus your own test or tests).

Fill in the four parts of `docs/my_feature.md`:

| Part | What to write |
|---|---|
| **What it does** | Two or three sentences. |
| **Why my dataset needs it** | What you saw in the data that made you want it. |
| **Where it appears** | A new function in `src/data.py`, a new part of the data page, or both. |
| **How I will know it works** | The test Claude must write in `tests/test_my_feature.py`, using a fake dataset where you know the right answer. |

Guide example (road signs): *What it does:* a histogram of box sizes per class on the data page. *Why:* many stop signs are tiny and far away. *Where:* a new function `box_areas()` in `src/data.py` and a chart in tab 1 · Data. *How I will know:* a test with a fake dataset where I know every box area.

## 2. Facts about my project and data (for the "Why" part)

Project: reusable and disposable water bottles in gym water-station photos, so that a facility manager can decide whether to install a water vending machine or refill stations.

Stand-in dataset: Kaggle `siddharthkumarsah/plastic-bottles-image-dataset`, CC BY 4.0, one class (`bottle`), converted to `data/bottles_yolo`.

| Split | Images | Boxes | Boxes per image | Median box area % | Boxes < 1% of image |
|---|---|---|---|---|---|
| train | 2177 | 3186 | 1.46 | 16.43 | 280 |
| val | 1174 | 1550 | 1.32 | 15.35 | 123 |
| test | 648 | 792 | 1.22 | 14.18 | 64 |

Quality checks above zero:

| Check | Count | Meaning |
|---|---|---|
| out_of_range_boxes | 25 | cx, cy outside 0 to 1, or w, h outside (0, 1] |
| tiny_boxes_lt_0.1pct | 116 | w times h under 0.001 of the image |
| duplicates_across_splits | 16 | the same image bytes (md5) in two splits |

The checks that were zero: images_without_label, labels_without_image, empty_labels, unknown_class_ids.

What you wrote in your own observations (reuse or ignore): three boxes that do not match (`00000023_jpg`, `ScreenShot2019-11-07`, `JPEG-image-488`), only one class, incorrect boxes, and very different photos.

What `/review-dataset` asked about (questions only, you decide if they are real): a possible non-bottle object (`plastic45_jpg`), a box away from the bottle (`800455_jpg`), a white bottle that may not be a water bottle (`plastic-157-_jpg`), near-copies of the same "elma" bottle on different backgrounds, and stock-photo watermark or transparent-background images that do not look like gym photos.

## 3. What already exists (so you do not ask for it again)

`src/data.py` functions: `load_project`, `download`, `pixels_to_yolo`, `yolo_to_pixels`, `read_voc`, `read_coco`, `read_labels`, `split_names`, `write_data_yaml`, `convert_to_yolo`, `prepare`, `class_counts`, `split_stats`, `review`, `draw_boxes`, `boxes_for`, `show`, `sample_grid`, `write_report`. Constants: `SPLITS`, `FORMATS`, `IMAGE_EXT`.

Data page (`app/app.py`, tab "1 · Data"): class balance chart, boxes-per-class table, split statistics table, quality-checks table, image browser (split plus image number), and the 20-image grid with a seed.

Helper functions in `src/data.py` that start with an underscore (for example `_images`, `_labels`) are internal. Your feature can use them, but say so in the plan.

## 4. Rules to keep in mind

- Never edit files in `data/bottles_yolo` by hand. Anything that changes the data must go through `prepare()` and be tested.
- Claude proposes a plan and waits for your approval before writing code.
- Do not ask Claude to write your observations or conclusions for the assignment.
- A good feature is small: one new function, or one new element of the page, and one clear test.

## 5. Ideas (your own is better)

Ideas from the guide, and ones that match what your data showed:

1. A histogram of box sizes per class on the data page (many small boxes: 280 in train).
2. A check that flags very thin or very elongated boxes.
3. A button that writes the files with problems to `results/problems.csv` (25 out of range, 116 tiny, 16 duplicates).
4. Showing only the images that contain a chosen class, or only the images that have a flagged problem, in the image browser.
5. A function that lists the duplicate pairs across splits so you can view both images side by side.
6. A count of images with more than N boxes, to look at crowded scenes (the grid showed images with several bottles in one box).

## 6. How to ask Claude (steps 22 and 23)

After you have written `docs/my_feature.md`, start Claude Code (`claude`) and send this request:

```
Read docs/my_feature.md and docs/data_spec.md. Plan this improvement, including a test
in tests/test_my_feature.py that would fail if it were wrong. Wait for my approval.
```

Check the plan:
- The test must check what you wrote under "How I will know it works", on a fake dataset where you know every number.
- Ask for changes until it does. Ask about anything you do not understand.
- Approve by typing: `Approved. Write the code and the test, then run: python -m pytest -q tests`

Then run the suite yourself (type the command only, not the prompt):

```
python -m pytest -q tests
```

You should see 38 passed or more. Run `python app/app.py` and look at the improvement with your own eyes. Write in your Assignment 2.2 description what it showed about your data (that sentence is yours).

## 7. After the improvement (steps 24 to 26)

1. **LOG.md:** a few lines: what you asked, what you checked, what you changed by hand. Entries for steps 13 to 15 (building `src/data.py` and `app/app.py`) and for the dataset notebook still need to be added.
2. **Commit and push.** Run `git status` first and check that no images or model files are listed (`data/` is in `.gitignore`):
```
git add -A
git commit -m "Assignment 2.2: data module, data page and my feature"
git push
```
3. **Submit in Moodle (Assignment 2.2):** online text with the repository link and the dataset link, and one file named `A2_2_Sow_Yaya_MAI630` (.pdf or .docx), about one page, with the grid image (`results/dataset_sample_grid.jpg`).
   - Repository link: https://github.com/yayabsai312/my-detector
   - Dataset link: https://www.kaggle.com/datasets/siddharthkumarsah/plastic-bottles-image-dataset

**Deadline:** the guide says Sunday 8 November 2026, 11:59 PM, and the class slide says Monday 9 November 2026, 11:59 PM. Use the earlier date to be safe.

## 8. Checklist

- [ ] `docs/my_feature.md` written by me (all four parts)
- [ ] Claude's plan reviewed and approved, with a test that matches "How I will know it works"
- [ ] `python -m pytest -q tests` shows 38 passed or more
- [ ] Data page opened, improvement checked with my own eyes
- [ ] `LOG.md` updated
- [ ] `git status` clean of data, then commit and push
- [ ] Assignment 2.2 file and links submitted in Moodle
