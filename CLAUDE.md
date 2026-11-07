# My detector — project instructions

## Goal
Over four weeks this repository becomes my own object-detection system: classical computer vision (Module 1),
my dataset (Module 2), a trained detector (Module 3) and an app that someone else can run (Module 4).
The object I plan to detect: disposable water bottles

## How we work
- Before writing code, propose a short plan (files, steps, how we will verify it) and wait for my approval.
- Run the code and show me the real output. Do not claim something works without running it.
- I write the observations, interpretations and conclusions myself. Do not write them for me.

## Environment
- Python 3.12 in `.venv`. Install only from `requirements.txt`. Do not add or upgrade packages without asking.
- Notebook kernel: "Python (mai630)".
- Tests: `python -m pytest -q tests`
- Run a notebook end to end: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_classical_cv.ipynb`

## Conventions
- OpenCV loads images as BGR: convert to RGB before showing them.
- Module 1 operations live in `src/classical.py` and follow `docs/classical_spec.md`.
- Never commit datasets, model weights or `.venv/` (see `.gitignore`).
- Code, comments and file names in English.

## Module 2 — inside a CNN, my dataset and the data page
- `src/cnn.py` follows `docs/cnn_spec.md` (Assignment 2.1). `notebooks/02_my_field.ipynb` uses it on photos in `data/my_field/`.
- My project is described in `project.yaml` (name, classes, format, source). Read it before touching data code.
- Data code lives in `src/data.py` and follows `docs/data_spec.md`. The app lives in `app/app.py` (tab "1 · Data").
- My own improvement is described in `docs/my_feature.md`; it gets its own test in `tests/test_my_feature.py`.
- The dataset is converted to `data/<name>_yolo/` (YOLO format, fixed split, seed 42). Never edit files there by hand.
- Tests: `python -m pytest -q tests` (37 must pass, plus my own). Run the app: `python app/app.py`.
- Skill `review-dataset` (in `.claude/skills/`): use it to gather the evidence about my dataset. It asks questions; I write the conclusions.
- Always propose a plan and wait for my approval before writing code. Do not write my observations for Assignments 2.1 and 2.2.
