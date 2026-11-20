# My detector (MAI 630)

## What it does
My system detects reusable and disposable water bottles in photos of gym water-station areas, so that a facility manager can decide whether to install a water vending machine or refill stations. **Current state:** it is trained on a stand-in dataset (Kaggle plastic bottles, one class: `bottle`). It finds bottles, but it does not yet tell reusable from disposable; that needs my own gym photos (see "What doesn't work").

## Install
You need Python 3.12 and git. The packages are listed in `requirements.txt` (pinned for CPU only, no GPU needed).

```bash
git clone https://github.com/yayabsai312/my-detector.git
cd my-detector
python3.12 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name mai630 --display-name "Python (mai630)"
python check_setup.py
```

`check_setup.py` should print the Python version and every package without `MISSING`. On a fresh clone its last line is `ENVIRONMENT OK — now add your photos`, not `READY`: `READY` only appears after you put at least 3 photos in `data/m1_images/`, which is needed for Module 1 only, not for this app.

## Get the data
The dataset is the Kaggle set `siddharthkumarsah/plastic-bottles-image-dataset` (CC BY 4.0, 1.71 GB, described in `project.yaml`). Download and convert it by running the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_my_dataset.ipynb
```

(or open `notebooks/02_my_dataset.ipynb` with the kernel "Python (mai630)" and choose Run All). The raw download goes to the Kaggle cache; the converted dataset (YOLO format, fixed split, seed 42) ends up in `data/bottles_yolo/`. The `data/` folder is not in the repository.

## Train
The trained model is **not** in the repository (weights and `runs/` are never committed), so you train it yourself:

1. Start the app: `python app/app.py` and open http://127.0.0.1:7860.
2. Open the tab "2 · Training", keep training data `all`, epochs `30`, image size `416`, write your reason and press "Start training".

On my computer (CPU only) this took 132.9 minutes (about 2 hours 13 minutes). For a quick try, use 2 epochs: about 9 minutes, but the model is much weaker (mAP50 0.39 on validation). The run is saved in `runs/<name>/weights/best.pt`.

## Detect
Start the app with `python app/app.py` and open http://127.0.0.1:7860. On the tab "3 · Detection & results":

1. Choose a finished run (the one with the best mAP50 is preselected).
2. Upload a photo, or pick one of the 40 test-split photos.
3. Press "Detect", then move the Confidence slider: boxes below the slider value disappear and the count updates. It starts at the threshold in `project.yaml` (0.25).

The tab also shows the threshold table and example figures from `results/` (made by `notebooks/03_my_results.ipynb`).

## Results
On the test split (645 images, one class), my run `all30` reaches **mAP50 0.454** and **mAP50-95 0.343**. I use the confidence threshold **0.25**; at that value precision is 0.58 and recall 0.51 (IoU 0.5).

**Why I chose 0.25.** A facility manager counting bottles is hurt by both kinds of error, so I compared the four thresholds I tested on the test split (776 labeled bottles, IoU 0.5):

- 0.10: precision 0.45, recall 0.54, 520 false detections (942 boxes in total).
- 0.25: precision 0.58, recall 0.51, 286 false detections, 379 bottles missed (683 boxes).
- 0.50: precision 0.70, recall 0.38, 123 false detections, 484 bottles missed (415 boxes).
- 0.75: precision 0.92, recall 0.16, 654 bottles missed (133 boxes).

0.25 has the highest F1 of the four (0.54) and gives the total closest to the 776 labelled bottles (12% low; 0.10 gives 21% high, 0.50 gives 47% low), although part of that closeness comes from false alarms offsetting misses. The facility manager pays for both errors: false alarms can lead to vending capacity that is not needed, and misses can lead to too few refill stations. Going from 0.10 to 0.25 removes 234 false detections and loses only 25 correct ones; going to 0.50 loses 105 more real bottles. I accept some false alarms at 0.25 to avoid missing as many bottles. This is a compromise, not a proven optimum: I read it from the test-split table, so for an unbiased result I should choose the threshold on the validation split and then evaluate once on test, and I should choose it again with real gym photos.

## What doesn't work
- **Only one class.** The model cannot tell reusable bottles from disposable ones, which is the goal of the project, because the stand-in dataset has a single class, `bottle` (see `project.yaml`).
- **Never tested in a gym.** `data/gym/` is empty. I did not use the 27 AI-generated placeholder images, so I have no result on real gym water-station photos.
- **Detection is weak even on the stand-in data.** On the test split (645 images, 776 labeled boxes) mAP50 is 0.454 and mAP50-95 is 0.343. At confidence 0.25 the model found 397 of the 776 bottles (recall 0.51), missed 379, and made 286 false detections (precision 0.58).
- **Label noise in the data.** In the six-image review, the three images counted as wrong (`56_jpg`, `-171_jpg`, `waterbottles_png`) have malformed ground-truth boxes: boxes that start above the image, boxes only 3 to 4% of the image width, and one box covering a whole group of bottles. So some wrong detections come from the labels, not the model. The dataset also has 25 out-of-range boxes (4 in the test split) and 116 tiny boxes (16 in the test split), which I left in and reported. A glass-looking bottle (`z1011`) is in the plastic-bottle data, and one validation/test duplicate pair exists.
- **Threshold chosen on the test table.** I picked 0.25 from the test-split numbers, so they are slightly optimistic; see Results.
- **Three test images were not evaluated.** Their label files mix polygon and box rows, which Ultralytics rejects, so 645 of the 648 test images were used.
- **Very little real-world testing.** One phone photo of a Member's Mark bottle was detected (confidence 0.85). I have not tried other bottle types, several bottles in one photo, bottles held in a hand, or low light.
- **No trained model in the repository.** Weights and `runs/` are never committed, so a new user must train first (about 2 hours 13 minutes on my CPU).

In short, the system shows basic bottle detection, but it is not yet able to tell bottle types apart or to count bottles reliably in a real gym.
