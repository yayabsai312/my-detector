# Working log

What I asked the AI assistant, what I checked, and what I changed. A few lines per session.

## Module 1
- Asked Claude Code to read CLAUDE.md and docs/classical_spec.md, plan src/classical.py (the 15 functions and how to check each), wait for my approval, then write it and run the tests.
- Reviewed the plan before approving it. Checked the code against the spec table: BGR-to-RGB conversion and FileNotFoundError in load_rgb, clipping instead of uint8 wrap-around in brightness_contrast, ValueError for an even blur size and for canny low >= high, and convolve_by_hand using reflected borders without flipping the kernel.
- Ran `python -m pytest -q tests`: all 13 tests passed. Ran `python check_setup.py`: environment OK.
- Changed: nothing by hand; used the generated file as is. Committed it as "Implement Module 1 classical CV operations in src/classical.py".
- Opened notebooks/01_classical_cv.ipynb with the Python (mai630) kernel, ran the setup cells and loaded my three photos of disposable water bottles (Object1-3.jpeg).
- Part A: ran greyscale, brightness/contrast, blur, Otsu threshold (T = 142, 137, 127) and the 3 x 3 kernel cell. Wrote each "what it measured" sentence myself and revised them after feedback from the AI assistant, for example fixing which result looked washed out (the beta result) and adding what I saw in my photos.
- A5: first ran the sharpening kernel without writing a prediction, so I switched to the horizontal-change kernel [[-1,0,1],[-1,0,1],[-1,0,1]], wrote my prediction before running it, then added my result. My hand-written convolution matched OpenCV (difference 0.0).
- Part B: ran Sobel, Laplacian, Canny at three threshold pairs and the side-by-side comparison. Canny's edge pixels fell as the thresholds rose (Object3: 28,345 / 7,196 / 617; Object1: 12,925 / 3,038 / 0).
- The AI assistant reviewed my sentences against the figures, polished the wording of my B5, B6 and comparison text from my own notes, wrote those cells into the notebook file, ran the notebook top to bottom and exported the PDFs. I checked the results against my figures and kept my conclusions about why classical operators fail on clear plastic bottles.
- Changed by hand: the kernel in A5, and the wording of the sentences.

## Module 2
- Added the Module 2 update to my repository (nothing from Module 1 overwritten) and pasted the Module 2 rules into CLAUDE.md. Ran `python check_setup.py`: READY.
- Ran notebooks/02_inside_a_cnn.ipynb with the project .venv kernel (Python 3.12.7): it finished with no errors.
- Asked Claude to read CLAUDE.md and docs/cnn_spec.md and plan src/cnn.py: the nine functions and how to check each. Reviewed the plan table (the nine functions and how each one is checked) and approved it before any code was written.
- Claude wrote src/cnn.py. It used one hook on the layer's output and one on that output's gradient, instead of PyTorch's built-in backward hook, to avoid trouble with ResNet's in-place ReLU.
- Checked: ran `python -m pytest -q tests/test_cnn.py` on my PC -> 10 passed in 2.51s; trace(small_cnn()) weights 216 + 1,152 + 32 = 1,400; ResNet-18 11,689,512 parameters.
- Ran notebooks/02_my_field.ipynb on three photos of disposable water bottles (Object1-3.jpeg copied into data/my_field/). Wrote my reading of the results in the last cell: 1 of 3 correct (bucket, water bottle, broom).
- Looked at the printed top-3 for each photo: Object 1 -> bucket 0.18, bow 0.08, ladle 0.07 (no bottle class); Object 2 -> water bottle 0.90; Object 3 -> broom 0.21, water bottle 0.11 (second).
- The AI assistant reviewed my written reading against the figure and the printed output and flagged what did not match (a copied answer for Object 2, angle not explaining Object 1 vs 2, claims with no evidence, no overall conclusion). I answered its questions from my own observations and rewrote the text; the conclusions are mine. The assistant only fixed the Markdown formatting.
- Changed by hand: nothing in src/cnn.py (used the generated file as is); I wrote and revised my reading in the last cell of 02_my_field.ipynb myself.
- Tested a CLAUDE.md rule: asked Claude Code to delete the tiny boxes from data/bottles_yolo. It did not edit anything, quoted my rule (never edit files there by hand), and proposed a plan instead (a min_box_area option in src/data.py with a test). I did not approve it.
- Ran /review-dataset in Claude Code: it regenerated results/dataset_review.md, summarized the numbers, and asked 11 questions about specific images in the grid. It did not change anything in data/ and did not write my conclusions. Its questions on 00000023_jpg, ScreenShot2019-11-07 and JPEG-image-488 matched the three boxes I had found myself.
- Asked Claude Code to plan src/data.py and app/app.py from docs/data_spec.md; reviewed the plan (including renaming the dataset's "valid" folder to "val") and approved it. 37 passed. Used project.yaml with the Kaggle plastic-bottles dataset (1 class, stand-in).
- Ran notebooks/02_my_dataset.ipynb: 2,177 / 1,174 / 648 images, 5,528 boxes, 25 out-of-range boxes, 116 tiny boxes, 16 duplicate pairs across splits. Opened the 20-image grid and wrote my own observations in the last cell.
- My feature: wrote docs/my_feature.md myself (duplicate image detection across splits). Claude planned it with a test and built find_duplicate_pairs() in src/data.py, a table on the data page and tests/test_my_feature.py. 38 passed.
