## Module 2 — inside a CNN, my dataset and the data page
- `src/cnn.py` follows `docs/cnn_spec.md` (Assignment 2.1). `notebooks/02_my_field.ipynb` uses it on photos in `data/my_field/`.
- My project is described in `project.yaml` (name, classes, format, source). Read it before touching data code.
- Data code lives in `src/data.py` and follows `docs/data_spec.md`. The app lives in `app/app.py` (tab "1 · Data").
- My own improvement is described in `docs/my_feature.md`; it gets its own test in `tests/test_my_feature.py`.
- The dataset is converted to `data/<name>_yolo/` (YOLO format, fixed split, seed 42). Never edit files there by hand.
- Tests: `python -m pytest -q tests` (37 must pass, plus my own). Run the app: `python app/app.py`.
- Skill `review-dataset` (in `.claude/skills/`): use it to gather the evidence about my dataset. It asks questions; I write the conclusions.
- Always propose a plan and wait for my approval before writing code. Do not write my observations for Assignments 2.1 and 2.2.
