"""Checks the system as a whole: the three pages, .gitignore and the commit hook (Module 4, Assignment 4.1)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOOK = ROOT / ".claude" / "hooks" / "pre_commit_check.py"


def test_app_has_three_pages():
    import gradio as gr
    from app.app import build_app
    old = Path.cwd()
    os.chdir(ROOT)
    try:
        demo = build_app()
    finally:
        os.chdir(old)
    labels = [b.label for b in demo.blocks.values() if isinstance(b, gr.Tab)]
    assert labels[:3] == ["1 · Data", "2 · Training", "3 · Detection & results"]


def test_gitignore_keeps_data_and_weights_out():
    lines = (ROOT / ".gitignore").read_text().split()
    assert "runs/" in lines and "*.pt" in lines and any(l.startswith("data") for l in lines)


def run_hook(command, repo):
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repo)}
    event = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run([sys.executable, str(HOOK)], input=event, env=env, capture_output=True, text=True)


def fake_repo(tmp_path, filename):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / filename).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / filename).write_bytes(b"x")
    subprocess.run(["git", "add", "-f", filename], cwd=tmp_path, check=True)
    return tmp_path


def test_hook_blocks_weights(tmp_path):
    r = run_hook('git commit -m "my model"', fake_repo(tmp_path, "runs/all_30/weights/best.pt"))
    assert r.returncode == 2 and "best.pt" in r.stderr


def test_hook_lets_clean_commits_and_other_commands_pass(tmp_path):
    repo = fake_repo(tmp_path, "src/detect.py")
    assert run_hook('git commit -m "detect page"', repo).returncode == 0
    assert run_hook("git status", repo).returncode == 0


def test_hook_is_registered():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    entries = settings["hooks"]["PreToolUse"]
    assert any(e.get("matcher") == "Bash" and any("pre_commit_check.py" in h["command"] for h in e["hooks"]) for e in entries)
