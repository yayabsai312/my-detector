---
name: explain-run
description: Explain a finished or running YOLO training in plain language from its results.csv — how the losses moved, the best epoch, the time, and signs of overfitting. Use it when the user asks how a training went or what its curves mean.
---

# Explain a training run

Goal: help the student read a run. You never start a training and you never write the student's conclusions.

## Steps
1. Find the run. Use the folder the student names; otherwise the newest folder in `runs/` with a `results.csv`.
2. Run, from the project root with `.venv` active:
   `python -c "from src import train; import json,sys; r=train.read_progress(sys.argv[1]); print(json.dumps(train.summarize(r))); print(r[0]); print(r[-1])" <run folder>`
3. Report in plain language, in at most six short lines:
   - how long it took and what the best epoch was;
   - what the training loss and the validation loss did;
   - what mAP50 did, and whether it stopped improving before the end;
   - whether it shows overfitting, using the `overfitting` value, and why.
4. If `<run>_why.txt` exists, read it and ask one question comparing the student's reason with what happened.

## Rules
- Use only the numbers in the files. Never invent a reason you cannot see in them.
- Do not change any file. Do not start or stop a training.
