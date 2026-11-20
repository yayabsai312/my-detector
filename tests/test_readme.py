"""Checks README.md against docs/readme_spec.md (Module 4, Assignment 4.1). These fail until you write your README.
The commit hook does not run this file, so you can commit your code before the README is ready."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ["What it does", "Install", "Get the data", "Train", "Detect", "Results", "What doesn't work"]


def readme_sections():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    parts = re.split(r"^##\s+(.+?)\s*$", text, flags=re.M)
    return {parts[i].strip().replace("’", "'"): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def test_readme_has_every_section_in_order():
    found = [s for s in readme_sections() if s in SECTIONS]
    assert found == SECTIONS, f"README.md sections found: {found}"


def test_readme_says_how_to_run():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "requirements.txt" in text and "python app/app.py" in text
    assert not re.search(r"(/Users/|C:\\\\Users\\\\|/home/)\w+", text), "no personal paths in the README"


def test_readme_declares_what_does_not_work():
    body = readme_sections().get("What doesn't work", "")
    assert len(body.split()) >= 15, "write at least one real limitation, and why"


