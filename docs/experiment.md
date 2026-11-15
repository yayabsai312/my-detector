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
Epochs, minutes and best mAP50:

## Was my prediction right? What did I learn?
