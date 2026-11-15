# My training experiment

Fill this in **before** you train. Do not use Claude to write it. Claude may question it afterwards.

## The data I chose
Which option: all / half / one class / large only: all

## Why I chose it
I chose `all` so the first run uses all 2,177 training images and keeps examples of both typical and tiny bottles. About 13% of training boxes are under 2% of an image, and the test set also includes small boxes, so dropping images based on box size could make the model less prepared for part of the final evaluation. The `large` option removes only 8% of images and does not directly clean the 25 out-of-range or 116 tiny boxes, so it does not address those annotation concerns. Using all the data gives me a clear baseline; `half` would be quicker but would leave out about half the examples, and `one_class` is the same as `all` for this one-class dataset.

## What I expect to happen
My prediction, in one or two sentences:
I expect the `all` run to achieve a higher mAP50 than a run trained on `half`, because it can learn from twice as many training images, though the difference may be modest. I’ll compare this prediction with the best mAP50 after training.

---
Fill this in **after** training.

## What actually happened
Epochs, minutes and best mAP50: 30 epochs, 132.9 minutes, best mAP50 0.486 (epoch 24, validation split; last epoch 0.475).

## Was my prediction right? What did I learn?
My prediction that `all` would beat `half` by a modest amount cannot be checked because I trained only `all`; I would need to train `half` with the same settings and compare the results. I learned that the 2-epoch trial’s mAP50 of 0.392 was only an early reference: the 30-epoch run reached a best mAP50 of 0.486, and took 132.9 minutes, close to my estimate of about 2 hours 20 minutes. Training loss kept falling from 1.77 to 1.22 while validation loss flattened near 1.9 after epoch 15; mAP50 fluctuated by up to about 0.02 from epochs 20–30, and the overfitting flag stayed false. The best checkpoint came from epoch 24, so I will use `best.pt` rather than treating epoch 30 as best. This is still only a one-class Kaggle baseline with no gym photos, so I have not tested reusable versus disposable bottles or performance in the gym.
