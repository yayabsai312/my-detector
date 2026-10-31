# Working log

What I asked the AI assistant, what I checked, and what I changed. A few lines per session.

## Module 1
- Asked Claude Code to read CLAUDE.md and docs/classical_spec.md, plan src/classical.py (the 15 functions and how to check each), wait for my approval, then write it and run the tests.
- Reviewed the plan before approving it. Checked the code against the spec table: BGR-to-RGB conversion and FileNotFoundError in load_rgb, clipping instead of uint8 wrap-around in brightness_contrast, ValueError for an even blur size and for canny low >= high, and convolve_by_hand using reflected borders without flipping the kernel.
- Ran `python -m pytest -q tests`: all 13 tests passed. Ran `python check_setup.py`: environment OK.
- Changed: nothing by hand; used the generated file as is. Committed it as "Implement Module 1 classical CV operations in src/classical.py".