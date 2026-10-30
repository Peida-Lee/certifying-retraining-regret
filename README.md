# Certifying Retraining Regret through Residual Structure

Source and reproducibility materials for the preprint by Peida Li.

The method encloses a fixed ridge predictor's regret against full-feature
retraining over all allowed continuous completions of an interval-valued table.
The paper presents residual-structure certificates, exact row geometry and
validated global refinement with feasible original-table witnesses.

## Contents

- `paper.pdf`: the complete 46-page preprint, with an eight-page main text.
- `main.tex`, `sections/`, `images/`: manuscript source, proofs and figures.
- `anc/`: implementations, protocols, aggregate results and reproduction guide.
- `arxiv_source.zip`: the self-contained source archive prepared for arXiv.

## Reproduce

Start with [anc/README.md](anc/README.md). Run its commands from `anc/`.
The offline checks need no public record files. Public-data reruns download
inputs from their documented providers and reconstruct row-level certificates
locally. The portable release contains aggregate results and synthetic checks.

```text
cd anc
python -m pip install -r requirements.txt
python code/verify_offline.py
python code/verify_refinement_offline.py
python code/verify_row_edge_offline.py
python code/verify_compressed_v29.py
python code/verify_profile_v32.py
python code/extend_validation_baselines.py --aggregate-only --dry-run
python code/plot_paper_figures.py --output-dir ../reproduced_figures
python code/plot_final_scope.py --output ../reproduced_figures/scope_final.pdf
```

Windows setup and SCIP requirements are described in the companion README.
Compile `main.tex` with XeLaTeX or Tectonic. Study-specific precision targets,
runtime accounting, candidate definitions and baseline comparisons appear in
the paper and the supplied protocols.

## Author

Peida Li, School of Basic Medical Sciences, Peking University.
