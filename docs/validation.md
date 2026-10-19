# Validation

Cleanup verification date: 2026-10-19. No model inference, training, metric regeneration, package installation, or remote publishing was performed.

## Passed checks

- All public Python files parsed with `ast.parse` (including preserved vendor scripts).
- Seven immutable records (six CSVs and the split) match SHA-256 hashes captured before cleanup.
- The split contains exactly 100 unique three-column rows in original order, samples 0000–0099.
- All 14 first-party modules imported without downloads/model initialization.
- All 11 public CLI help commands, the internal worker help command, and the repository checker were exercised.
- GAN dry run parsed the original 100 samples and found all 300 local split files. It did not load weights.
- Preserved GAN model, network, loss, metric, training, and legacy entry-point files are byte-identical to their pre-cleanup local versions; only the documented dataset mask guard differs among vendor Python files.
- AST comparisons confirmed exact preservation of PSNR/SSIM loader/resize/list functions, LPIPS loading/matching/computation functions, binary-mask conversion, saved-GAN loader, and qualitative masking functions.
- Threshold boundary test retained >10 mask semantics; GAN-result loader returned RGB; split parser matched all original names; historical-output guard rejected historical destination paths.
- Existing comparison-grid CLI executed on local sample 0004, writing only ignored `results/validation/` output.
- Publication checks exclude checkpoints, datasets, caches, recovery material and nested Git metadata; no proposed file exceeds 10 MiB.

These checks establish source/entry-point integrity, not numerical equivalence of new model inference.

## Missing dependencies and unverified commands

The available Python runtime has NumPy and Pillow. Missing research dependencies: matplotlib, torch, torchvision, diffusers, transformers, accelerate, safetensors, huggingface_hub, datasets, scipy, cv2, skimage, imageio, yaml, lpips, future.

CLI help succeeds through deferred model imports; this does **not** establish runtime availability of those packages. Full GAN/diffusion/hybrid inference, dataset export/download, PSNR/SSIM evaluation, LPIPS evaluation, matplotlib plotting, training, and legacy vendor FID/metrics execution remain unverified in a complete research environment. They require the listed dependencies and, where applicable, verified model weights and dataset files.

No full-environment installation or lockfile validation was claimed. Historical dataset/model revisions and exact training provenance remain unrecorded. Legacy vendor `scripts/metrics.py` still references removed SciPy/skimage APIs and is not a supported evaluation command; use the first-party evaluator instead. Legacy vendor example/default checkpoint paths refer to archived local assets; use the documented wrapper rather than invoking vendor `test.py` directly.

## Broken/limited legacy entry points

The original vendor test launcher, training launcher defaults, and auxiliary scripts are preserved for provenance, not presented as portable root commands. The new wrapper is the supported path. Its supplied config is test-only; train/eval require a separate config with training/validation lists. Model-id access returned HTTP 401 during the web check; diffusion model acquisition is unresolved. No replacement was chosen.

## Final publication-copy checks

- A separate copy containing only the proposed public files passed six primary CLI help checks. Its GAN dry run correctly reported all 300 dataset paths absent and no checkpoint directory, confirming it does not accidentally depend on the populated local working tree for launch/help.
- The actual vendor single-channel mask branch was exercised against a local exported L-mode mask with image I/O supplied directly. It retained the same >0 threshold and avoided invalid RGB conversion; this is a narrow loader check, not a full scikit-image/GAN inference test.
- AST comparisons also confirmed the original diffusion pipeline and hybrid refinement call arguments are unchanged.
- Proposed publication text was scanned for local Windows user/project paths, GitHub/Hugging Face token patterns, and private-key headers; no matches were found. This is a targeted scan, not proof that all possible secrets or image metadata are absent.
- Local Git repository initialized on `main`; no commits, staging, remote configuration, GitHub repository creation, or pushes.
