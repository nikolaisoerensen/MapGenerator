# MapGenerator — Project Notes for Claude Code

## Git worktrees: changes are invisible until merged or tested in-place

This project is frequently worked on via Claude Code sessions that run in an
isolated **git worktree** under `.claude/worktrees/<name>/`, on a branch like
`claude/<name>`. That worktree is a **separate directory and working tree**
from the user's main checkout at the repo root
(`C:\Lokale Dateien\Projects\Python\MapGenerator`, branch `main`).

**Editing files in a worktree does NOT affect the user's main checkout in any
way**, and the running desktop app (`python main.py`) only reflects whatever
directory it was actually launched from. If the user reports "nothing
changed" / "still looks exactly the same" after a fix, **check this before
re-investigating the original bug**:

```
git worktree list
git status --short   # confirm the fix is actually sitting uncommitted here
```

If the user is testing from their normal checkout and the current session's
work is in a worktree, the fix is real but invisible to them until one of:

1. **Test in place (fastest, no merge needed):** run the app directly from
   the worktree directory instead of the main checkout:
   ```
   cd "C:\Lokale Dateien\Projects\Python\MapGenerator\.claude\worktrees\<name>"
   "C:\Lokale Dateien\Projects\Python\MapGenerator\.venv\Scripts\python.exe" main.py
   ```
   Note: the venv lives at the **main checkout root**
   (`...\MapGenerator\.venv`), not per-worktree — worktrees don't get their
   own venv, always invoke the shared one explicitly as shown above.
2. **Commit + PR:** commit the worktree's changes and open a PR against
   `main` for the user to review and merge normally.
3. **Commit + direct merge:** commit on the worktree branch, user pulls/merges
   `claude/<name>` into their main checkout locally without a PR.

Always ask the user which of these they want — don't assume. Default to
suggesting option 1 first when the user just wants to see if a fix works,
since it requires no merge/commit decision at all.

## Verifying backend/generator changes without the live GUI

Most core generator logic (`core/*.py`) can be exercised headlessly via
throwaway smoke-test scripts run through the shared venv — see prior session
memory for established patterns (stubbing `gui.OldManagers.calculator_graph`
if missing on a given branch, building a minimal `FakeScheduler`, driving
`BaseTerrainGenerator` → `WeatherSystemGenerator` → `HydrologySystemGenerator`
→ `BiomeClassificationSystem` end-to-end with real default parameters from
`gui/config/value_default.py`). This validates the actual computation but
**not** the Qt/OpenGL rendering — GUI-facing changes (matplotlib colormaps,
GLSL shaders, Qt widget behavior) still need the user to confirm visually
against the live app, per the worktree note above.
