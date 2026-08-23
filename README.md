# Trading

Canonical version-controlled source code for the Trading ChatGPT project.

## Persistence policy

- User-facing artifacts are persisted to the project's Google Drive workspace.
- Reproducible source code is persisted both to Google Drive and this GitHub repository.
- Routine generated or modified source files may be committed directly to `main`.
- Substantial refactors, risky changes, or coordinated multi-file changes should use a branch and pull request.
- Scratch calculations and disposable helper code should not be committed.

## Repository layout

- `code/` — reproducible scripts and source code
- `project_state/` — lightweight machine-readable or Markdown state needed to reproduce/continue work

Google Drive remains the canonical store for reports, spreadsheets, datasets, and other binary/user-facing artifacts. GitHub remains the canonical version history for source code.
