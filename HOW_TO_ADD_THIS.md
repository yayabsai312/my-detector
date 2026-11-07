# Module 2 update — how to add it to your repository
Copy **everything in this folder** into the repository you started in Module 1 (same folder as `CLAUDE.md`).
Use a command, so files are added and nothing from Module 1 is deleted (run it inside your project folder):

    Mac/Linux:   cp -R ~/Downloads/MAI630_Module2_Update/ .
    Windows:     Copy-Item -Recurse -Force "$HOME\Downloads\MAI630_Module2_Update\*" .

Do not drag the folders with Finder and choose "Replace": that deletes the Module 1 files in docs/ and tests/. Then:

1. Paste the text of `docs/CLAUDE_module2.md` at the end of your `CLAUDE.md`.
   The command also adds the hidden folder `.claude/skills/review-dataset/` (your first Claude Code Skill).
2. Put three photos from your own field in `data/my_field/` (Assignment 2.1).
3. Edit `project.yaml` for your own project (Assignment 2.2).
4. Follow the Module 2 Complete Practice Guide: you build `src/cnn.py`, `src/data.py` and `app/app.py` with Claude Code.
