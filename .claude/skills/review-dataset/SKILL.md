---
name: review-dataset
description: Gather the evidence to review this project's object-detection dataset before trusting it — class balance, box sizes, quality checks and the 20-image grid. Use it when the user asks to check, inspect, validate or describe the dataset, or before the first training run.
---

# Review the dataset

Goal: collect the evidence the student needs for Assignment 2.2. You never change the data and you never write
the student's conclusions.

## Steps
1. Read `project.yaml`. The dataset must be in `data/<name>_yolo/`. If that folder does not exist, tell the
   student to run `notebooks/02_my_dataset.ipynb` first, and stop.
2. Write the report by running, from the project root with `.venv` active:
   `python -c "from src import data as dt; p = dt.load_project(); print(dt.write_report('data/' + p['name'] + '_yolo', p['classes']))"`
3. Read `results/dataset_review.md` and open the image `results/dataset_sample_grid.jpg`.
4. Report, as a short checklist in plain language:
   - images and boxes per split;
   - boxes per class in the training split, and how many times larger the biggest class is than the smallest;
   - how many boxes cover less than 1% of the image;
   - every quality check with a count above zero, with the file names it lists;
   - in the grid: the image names where a box looks questionable (an object without a box, a box too big or too
     small, a box on the wrong thing), written as **questions for the student to verify**, not as conclusions.
5. End with exactly this line: "Your turn: open the grid, check my questions, and write the two things most likely
   to make this dataset hard (Assignment 2.2)."

## Rules
- Never edit, move or delete anything inside `data/`.
- Never write the student's observations, the two hardest things, or any text for their assignment.
- If a command fails, show the error and stop; do not try to fix the data.
