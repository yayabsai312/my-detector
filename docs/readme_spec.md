# What your README must contain (Module 4, Assignment 4.1)

Your `README.md` is for a classmate who has never seen your project. With only your README, a clean computer and your repository, they must be able to run your system from nothing. `python -m pytest -q tests/test_readme.py` checks the parts below (3 tests).

## Sections, in this order (use these exact headings, with `##`)
1. `## What it does` — one or two sentences: what it detects, in which photos, and who would use it.
2. `## Install` — every command, from cloning the repository to `python check_setup.py`. Say which Python version (3.12) and that `requirements.txt` is used. Give the command for Windows and for Mac/Linux when they differ.
3. `## Get the data` — how to download your dataset and convert it (the notebook or the command), and where it ends up.
4. `## Train` — how to start a training on the page "2 · Training", and how long it took on your computer.
5. `## Detect` — how to start the app (`python app/app.py`) and use the page "3 · Detection & results".
6. `## Results` — your test-split mAP50 and mAP50-95, the threshold you chose and why, in two or three lines.
7. `## What doesn't work` — every known limitation, and why. Not an empty section: a known limitation reported is worth more than a silent failure.

## How to check it really works from nothing
Clone your repository into a **new** folder, and follow your README word by word, without using anything you remember.
Every time you have to do something the README does not say, add it to the README. Then a classmate tries it (see the forum activity).

## Rules
- Write it with Claude Code if you want, but you check every command yourself, and the section `What doesn't work` is yours.
- Never put passwords, keys or personal paths (like `C:\Users\yourname`) in it.
