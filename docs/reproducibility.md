# Reproducibility and preserved scientific behavior

## Portability/execution changes

Source modules now have package imports and module entry points. Argument parsing occurs before heavy model imports so help can be inspected without loading weights. Hard-coded drive paths were removed from first-party code. Root defaults are derived from module locations or documented root-relative CLI arguments. Model and dataset paths are configurable.

The GAN wrapper runs the isolated vendor `src` in a subprocess, resolves config paths from the repository root, sets the checkpoint path, and passes exact manifest image/mask lists. It initializes device, OpenCV threads, and the original seed machinery from upstream `main.py`; the old custom `test.py` skipped that setup. CUDNN benchmarking is retained, so a fixed seed is not a guarantee of bitwise determinism.

`configs/gan.yml` preserves the original checkpoint config's effective last-key-wins values with duplicates removed. In particular, `INPUT_SIZE=256`, `MODEL=3`, `SEED=10`, and `DEBUG=1` remain. The launcher does **not** use upstream test mode's `INPUT_SIZE=0` override, which would change preprocessing relative to the local custom test entry point.

One vendor execution fix guards `rgb2gray` for single-channel masks: L-mode masks are already grayscale, while RGB masks still follow the original conversion. The same resize and `>0` threshold remain. No tensor conversion, edge scaling, crop/resize, or model calculations were modernized.

Historical metric paths are protected by an output guard. Bare CSV output filenames now work. Plotting ignores blank CSV rows and the `mean` summary row when reading per-image values; this corrects a plot-reader error and does not alter CSV contents or evaluation formulas. LPIPS plots save and close instead of opening a blocking window.

## Original methodological differences (documented, not corrected)

- Batch diffusion prompt: `realistic filling of missing area, consistent with surroundings`.
- Hybrid prompt: `A high-quality realistic image`; explicit 50 steps and guidance 7.5.
- Batch diffusion retains library defaults for steps, guidance, scheduler, and other unprovided pipeline settings. They depend on installed package/model revisions.
- Batch baseline converts RGB/L images and masks and resizes both to 512×512 using the original PIL defaults. It does not invert the mask.
- The single-image demo does invert its mask and saves a JPEG intermediate. It is not identical to batch inference.
- GAN test preprocessing uses `INPUT_SIZE=256` and the existing center-crop behavior for images; masks use the existing non-center-crop resize. These differences were preserved.
- Hybrid passes saved GAN RGB images and original L-mode masks directly into the existing pipeline; any internal resizing remains Diffusers behavior.
- Neither diffusion path sets a generator or explicit seed. No seed was added during cleanup. Existing hybrid outputs are skipped by filename, without validating their provenance.
- PSNR/SSIM compare full RGB images in [0,1], resize predictions to target size with BICUBIC if needed, and use `data_range=1.0`. SSIM uses `channel_axis=2`.
- LPIPS uses AlexNet, 256×256 RGB PIL resizing, and [-1,1] normalization. Matching is directory-based by numeric prefix and skips debug output suffixes. It does not read `test_100.txt` itself.
- Missing pairs are skipped by the original metric loops. Always verify evaluated counts. No mask-only metric was added.
- Export maps `coco_image` to input and edited `inpaint_image` to target. Calling that target a ground-truth uncorrupted photograph would be inaccurate.
- Original package versions, exact dataset/model revisions, GPU/runtime details, and exact checkpoint training provenance were not recorded. Historical exact numerical or pixel reproduction is not established.
- Existing GAN loader catches item errors and falls back to item 0; this can conceal data problems. That legacy behavior is retained and should be addressed only as a separately reviewed scientific change.
- Existing GAN tensor conversion casts grayscale/edge arrays to uint8. This was already present locally and is retained, even though it deserves separate methodological review.

## Immutable artifacts

The original 100-line split and six historical metric CSVs are preserved byte-for-byte. `docs/artifact_checksums.json` records their SHA-256 digests. `.gitattributes` prevents newline normalization of these records. No cleanup command ran model inference or recomputed historical metrics. Hash checks compare against the pre-cleanup copies.

The restored local data, weights, and previous outputs are deliberately outside the publication manifest. A fresh clone will require their verified acquisition before full inference. Dependencies have been consolidated but are not pinned to an unrecorded historical environment.
