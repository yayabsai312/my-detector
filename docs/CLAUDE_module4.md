## Module 4 — the full system
- Detection code lives in `src/detect.py` and follows `docs/detect_spec.md`. The app has a third tab, "3 · Detection & results".
- The three pages are one system: the data of page 1 is what page 2 trains on, and the runs of page 2 are what page 3 uses.
- The confidence threshold is in `project.yaml` (`threshold:`). I chose it in Assignment 3.2: never change it for me.
- `README.md` follows `docs/readme_spec.md`. The section "What doesn't work" is mine: you may list problems you find, but I decide what goes there.
- Hook `.claude/hooks/pre_commit_check.py` runs before every `git commit` you make: it blocks data, runs, model weights, big files and failing tests (except the README tests in `tests/test_readme.py`). If it blocks a commit, tell me why and fix the cause; never try to get around it.
- Tests: `python -m pytest -q tests` (75 must pass, plus my own).
