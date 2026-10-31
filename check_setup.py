"""Checks your project environment. Run: python check_setup.py"""
import sys
from pathlib import Path
ok = True
print("Python", sys.version.split()[0])
if sys.version_info[:2] != (3, 12):
    print("  WARNING: the course is tested with Python 3.12"); ok = False
for name in ("numpy", "cv2", "matplotlib", "torch", "ultralytics", "pytest"):
    try:
        m = __import__(name); print(f"{name:12s}", getattr(m, "__version__", "ok"))
    except Exception as e:
        print(f"{name:12s} MISSING ({e.__class__.__name__})"); ok = False
if Path("src/classical.py").exists():
    print("src/classical.py  present  (check it with: python -m pytest -q tests)")
else:
    print("src/classical.py  not yet  (you build it in step 11 of the guide)")
photos = [p for p in Path("data/m1_images").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
print(f"photos        {len(photos)} in data/m1_images" + ("" if len(photos) >= 3 else "  (you need at least 3)"))
print("READY" if ok and len(photos) >= 3 else ("ENVIRONMENT OK — now add your photos" if ok else "NOT READY: read the message above"))
