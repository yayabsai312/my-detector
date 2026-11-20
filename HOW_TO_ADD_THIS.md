# Module 4 update — how to add it to your repository
Copy **everything in this folder** into your project repository (same folder as `CLAUDE.md`). Use a command, so nothing from earlier modules is deleted (run it inside your project folder):

    Mac/Linux:   cp -R ~/Downloads/MAI630_Module4_Update/. .
    Windows:     Copy-Item -Recurse -Force "$HOME\Downloads\MAI630_Module4_Update\*" .

Do not drag the folders with Finder and choose "Replace". Then:

1. Paste the text of `docs/CLAUDE_module4.md` at the end of your `CLAUDE.md`.
   The command also adds the hidden folder `.claude/hooks/` and the file `.claude/settings.json`, which turns the hook on.
   Close Claude Code and open it again in your project folder, so it reads the hook.
2. Add your threshold from Assignment 3.2 to `project.yaml`, for example `threshold: 0.40`.
3. Follow the Module 4 Complete Practice Guide: you build `src/detect.py`, the page "3 · Detection & results" and your `README.md` with Claude Code.
4. Check that the tests pass: `python -m pytest -q tests` (75). `tests/test_readme.py` fails until your README is written: that is expected, and the hook does not block your commits because of it.
