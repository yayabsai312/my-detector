#!/usr/bin/env python3
"""Claude Code hook (PreToolUse, Bash): checks every `git commit` that Claude is about to run.

It blocks the commit when the staged files include datasets, training runs, model weights or files over 10 MB,
or when the tests fail (all of them except tests/test_readme.py: you can commit code before your README is ready). Exit code 2 blocks the command, and the message goes back to Claude.
Any other command passes untouched (exit code 0).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
FORBIDDEN_DIRS = ("data/", "runs/", "weights/", ".venv/")
FORBIDDEN_EXT = (".pt", ".pth", ".onnx")
MAX_MB = 10


def block(reason: str) -> None:
    print(f"Commit blocked by .claude/hooks/pre_commit_check.py: {reason}", file=sys.stderr)
    sys.exit(2)


try:
    event = json.load(sys.stdin)
except json.JSONDecodeError:
    sys.exit(0)
command = event.get("tool_input", {}).get("command", "")
if "git commit" not in command:
    sys.exit(0)

# `git commit -a` / `-am` also commits modified tracked files: check those as well.
args = ["git", "diff", "--cached", "--name-only"]
staged = subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.split()
if " -a" in command or " -am" in command or "--all" in command:
    staged += subprocess.run(["git", "diff", "--name-only"], cwd=ROOT, capture_output=True, text=True).stdout.split()

problems = []
for name in sorted(set(staged)):
    path = ROOT / name
    if name.startswith(FORBIDDEN_DIRS) or name.endswith(FORBIDDEN_EXT):
        problems.append(f"{name} (datasets, runs and model weights never go to GitHub)")
    elif path.exists() and path.stat().st_size > MAX_MB * 1024 * 1024:
        problems.append(f"{name} ({path.stat().st_size / 1e6:.0f} MB; the limit is {MAX_MB} MB)")
if problems:
    block("remove these files from the commit with `git restore --staged <file>` and check .gitignore: " + "; ".join(problems))

if (ROOT / "tests").is_dir():
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "tests", "--ignore=tests/test_readme.py"], cwd=ROOT, capture_output=True, text=True)
    if tests.returncode not in (0, 5):                      # 5 = no tests collected
        tail = "\n".join(tests.stdout.strip().splitlines()[-15:])
        block("the tests fail. Fix the code first, then commit again.\n" + tail)
sys.exit(0)
