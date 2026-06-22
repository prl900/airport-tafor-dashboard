# research/

Paper-style review of the TAF-from-NWP modelling work.

## Contents
- **`taf_from_nwp_review.md`** — the review in Markdown (renders on GitHub, embeds the PNG
  figures). Start here.
- **`taf_from_nwp_review.tex`** / **`taf_from_nwp_review.pdf`** — the same review as a
  LaTeX paper compiled to PDF (polished, print/share-ready).
- **`figures/make_figures.py`** — regenerates every figure as both `.png` (for the Markdown)
  and `.pdf` (vector, for LaTeX) from the measured experiment results.
- **`figures/*.png`, `figures/*.pdf`** — generated figures.

## Format choice (and why)
Two formats are provided from one shared figure set:

- **Markdown** is the primary format: it renders inline on GitHub/IDEs, diffs cleanly in
  review, and needs no toolchain. Best for an evolving working document.
- **LaTeX → PDF** is provided because `pdflatex` is available and a compiled PDF is the
  natural artifact for a "paper" (proper typography, math, captions). Use this to share or
  archive a fixed version.

(Pandoc could convert one to the other but is not installed; the two are kept in sync
manually — both are short.)

## Rebuild
```bash
# figures (PNG + PDF)
uv run python research/figures/make_figures.py

# PDF (run twice for cross-references); needs a LaTeX install (pdflatex)
cd research && pdflatex -interaction=nonstopmode taf_from_nwp_review.tex \
  && pdflatex -interaction=nonstopmode taf_from_nwp_review.tex \
  && rm -f *.aux *.log *.out
```

The figure values are the measured results from the experiments; provenance is in
`data/research_log.jsonl`, `docs/AUTONOMOUS_RESEARCH.md`, and `docs/PHASE_D_ROADMAP.md`.
