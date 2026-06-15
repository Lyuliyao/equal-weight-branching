# Repository Guide

This repository combines the numerical code, reference results, and manuscript
for the equal-weight branching project.

## Manuscript

- Main Overleaf-synced manuscript tree: `paper/`
- Current main manuscript file from Overleaf: `paper/cmame/cmame-main.tex`
- Older/local manuscript entry retained in GitHub: `paper/cmame-main.tex`
- Bibliography: `paper/main.bib`
- Figures used by the manuscript: `paper/figure/`

Before editing or reviewing manuscript text, sync from Overleaf first:

```bash
make paper-sync
```

Use `make paper-sync-dry-run` to preview incoming Overleaf changes without
editing files. `make paper-sync` stores the Overleaf Git token in macOS
Keychain on first use, then commits and pushes `paper/` changes to GitHub.

## Code And Results

- Experiment drivers and plotting code: `experiments/`
- Reproducible/reference outputs: `reference_results/`
- Project notes: `notes/`
- Python dependencies: `requirements.txt`

When connecting manuscript claims to code, start from the figure filenames in
`paper/figure/`, then search matching experiment and reference-result names.

## Sync Policy

Overleaf cannot directly sync this existing project to the existing GitHub
subdirectory `paper/`. Treat GitHub as the unified source for ChatGPT/Codex
context, and use the local sync script to mirror Overleaf into `paper/`.

The default sync keeps GitHub-only files under `paper/` instead of deleting
them. Use `scripts/sync-overleaf-paper.sh --delete --commit --push` only when
you intentionally want `paper/` to mirror Overleaf exactly.
