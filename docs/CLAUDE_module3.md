## Module 3 — training the detector and reading its results
- Training code lives in `src/train.py` and follows `docs/train_spec.md`. The app has a second tab, "2 · Training".
- Evaluation code lives in `src/evaluate.py` and follows `docs/eval_spec.md`. `notebooks/03_my_results.ipynb` uses it.
- Before training, I fill in `docs/experiment.md` myself. Never write my reason or my prediction for me.
- A real training takes a long time: never start one unless I ask. Check code with the tests first.
- Trained runs go to `runs/`. Never commit `runs/` or any `.pt` file.
- Metrics come from the test split, never from training images. mAP is never "at a threshold": at a threshold, report precision and recall.
- Skill `explain-run`: explains a finished run. Subagents `metrics-reviewer` and `image-auditor`: review my results. They ask and point out; I write the conclusions.
- Tests: `python -m pytest -q tests` (57 must pass, plus my own).
