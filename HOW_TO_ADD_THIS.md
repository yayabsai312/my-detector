# Module 3 update — how to add it to your repository
Copy **everything in this folder** into your project repository (same folder as `CLAUDE.md`). Use a command, so nothing from earlier modules is deleted (run it inside your project folder):

    Mac/Linux:   cp -R ~/Downloads/MAI630_Module3_Update/ .
    Windows:     Copy-Item -Recurse -Force "$HOME\Downloads\MAI630_Module3_Update\*" .

Do not drag the folders with Finder and choose "Replace". Then:

1. Paste the text of `docs/CLAUDE_module3.md` at the end of your `CLAUDE.md`.
   The command also adds the hidden folders `.claude/skills/explain-run/` and `.claude/agents/`.
2. Follow the Module 3 Complete Practice Guide: you build `src/train.py`, the page "2 · Training" and `src/evaluate.py` with Claude Code.
3. Check that the tests pass: `python -m pytest -q tests` (57).
