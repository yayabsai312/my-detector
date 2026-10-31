# My detector (MAI 630)

My own computer-vision project, built module by module.

| Module | What this repository gains |
|---|---|
| 1 | Environment, `src/classical.py`, notebook `01_classical_cv.ipynb` (Assignments 1.1 and 1.2) |
| 2 | My project description and dataset; the data page |
| 3 | Training page and evaluation |
| 4 | The full app and this README completed |

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name mai630 --display-name "Python (mai630)"
python check_setup.py
```
