# Cleanup change record

## Significant moves and renames

- `GAN_baseline/edge-connect/` → `third_party/edge_connect/`, preserving locally modified implementation, original README, license, requirements, and scripts.
- Nested vendor `.git/` → ignored `.local/edge_connect.git/`; no nested Git history will be published.
- Vendor `checkpoints/` → ignored root `checkpoints/edge_connect/`; original checkpoint YAML remains there as a local record.
- Vendor example images and generated outputs → ignored `.local/edgeconnect_examples/` and `.local/edgeconnect_results/`.
- `diffusion_baseline/*.py` → `src/diffusion/`; remaining demo images/results → `.local/diffusion_demo/`.
- `hybrid/*.py` → `src/hybrid/`; remaining cache/scaffolding → `.local/scaffolding/`.
- `eval_metrics.py`, `compute_lpips.py` → `src/evaluation/`; `compute_psnr.py` → `src/evaluation/plot_metrics.py`.
- `data/cocoinpaint_dataset.py` → `src/utils/export_dataset.py`; `data/check.py` → `src/utils/check_dataset.py`.
- `compare_hybrid.py`, `make_compare_grid.py` → `src/utils/`.
- Three root LPIPS CSVs and three `results/*metrics.csv` → `results/metrics/historical/`, byte-identical.
- Original `data/InpaintCOCO_export/test_100.txt` copied byte-identically to public `splits/test_100.txt`; local original retained.
- Original `psnr_bar.png`, `ssim_bar.png`, `lpips_bar.png` → `assets/`; selected comparison `0004_compare.png` copied to `assets/qualitative_0004.png`.
- Full `figures/`, boxplots, original source/configuration copies, and caches → ignored `.local/` recovery storage.
- Datasets remain in ignored `data/`; full GAN/diffusion/hybrid result directories remain in ignored `results/` subdirectories.
- Empty `hybrid_inpainting/third_party/` scaffolding and obsolete root/data package markers were archived locally. No research source, checkpoint, or result was permanently deleted.

## New files and execution changes

Added package markers, CLI entry points, isolated GAN worker, effective test YAML, requirements inventory, Git ignore/attributes, citation metadata, rights notice, lightweight pipeline SVG, and provenance/reproducibility documentation.

Model architecture and loss implementations are untouched. Existing prompts, pipeline calls, preprocessing formulas, and metric computations are retained. Safe execution edits and original methodological differences are distinguished in `reproducibility.md`. Historical files are protected from evaluator overwrites and checked by hashes.

The first commit is proposed only; no commits, remote repository, or pushes are created by cleanup. `first_commit_files.txt` lists the exact proposed publication paths, including itself. `validation.md` records checks and limitations.
