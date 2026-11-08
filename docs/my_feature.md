# My own improvement (Assignment 2.2)

**What it does (two or three sentences):**
I want to add a duplicate image detection feature to my dataset module. This feature will identify images that have identical content across the training, validation, and testing datasets. It will display the duplicate image pairs, their filenames, and the dataset splits where they appear so that I can review them.

**Why my dataset needs it (what you saw in the data that made you want it):**
My bottle dataset currently contains 16 duplicate images across different dataset splits. This is important because having the same image in both training and testing data can affect the reliability of model evaluation. The model might perform better on the test dataset because it has already encountered the same image during training. I need this improvement to identify these duplicates and better understand the quality and reliability of my dataset.

**Where it appears:**
I want to add a new function called `find_duplicate_pairs()` in `src/data.py`. The function will identify identical images across the train, validation, and test splits. I also want the Data page in `app/app.py`, under the 1 · Data tab, to display a table showing the duplicate image pairs, their filenames, and their corresponding dataset splits.

**How I will know it works (the test Claude must write in tests/test_my_feature.py):**
Claude must create a test in `tests/test_my_feature.py` using a small fake dataset. The test will contain two identical images placed in different dataset splits and one different image. I will verify that the function correctly identifies the duplicate pair, reports the correct filenames and splits, and does not flag the different image as a duplicate. The test must check the actual expected results, not just whether the function runs successfully.
