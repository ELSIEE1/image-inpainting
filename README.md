# Hybrid Controllable Inpainting

A Python image-inpainting project that explores how to fill masked image regions by combining **GAN reconstruction with Stable Diffusion refinement**. It compares a GAN baseline, a diffusion baseline, and a sequential hybrid pipeline on a fixed 100-image evaluation split.

Developed for the undergraduate thesis *Hybrid Controllable Inpainting: Integrating Diffusion and GAN Mechanisms for Efficient Image Restoration*.

## My contribution

My project work focuses on the integration and evaluation layer around existing models:

- Connecting saved EdgeConnect reconstructions to a Stable Diffusion refinement stage.
- Building batch inference workflows for the diffusion baseline and hybrid pipeline.
- Exporting image/mask/target pairs from InpaintCOCO and evaluating outputs with PSNR, SSIM, and LPIPS.
- Producing comparison figures and preserving the experiment's 100-sample split and historical CSV results.

The repository also includes portable CLI launchers and verification checks. **EdgeConnect and Stable Diffusion are third-party models**; the hybrid is a sequential integration, not a new model architecture or a jointly trained network.

**Technologies:** Python, PyTorch/torchvision, Hugging Face Diffusers and Datasets, Stable Diffusion inpainting, NumPy, Pillow, OpenCV, scikit-image, LPIPS, and Matplotlib. Dependencies are listed in [requirements.txt](requirements.txt).

## How the two stages work

![GAN to diffusion pipeline](assets/pipeline.svg)

1. **GAN reconstruction:** EdgeConnect predicts edges and completes the masked region, saving a coarse reconstruction.
2. **Diffusion refinement:** the hybrid loader passes that saved RGB reconstruction and its mask to `StableDiffusionInpaintPipeline`, using the original hybrid prompt, 50 inference steps, and guidance scale 7.5.

The standalone diffusion baseline runs on the input image instead of the GAN reconstruction. Prompts and preprocessing differ between the original experiments; those differences are preserved and [documented](docs/reproducibility.md).

## Example and evaluation

![Masked input, GAN, diffusion, hybrid, and target for sample 0004](assets/qualitative_0004.png)

Left to right: masked input, GAN baseline, diffusion baseline, hybrid output, and dataset target. This is an existing selected sample, not a newly generated result or a claim of typical performance.

- **PSNR ↑:** pixel reconstruction quality.
- **SSIM ↑:** structural similarity.
- **LPIPS ↓:** learned perceptual distance using AlexNet.

The original split is preserved in [test_100.txt](splits/test_100.txt). PSNR/SSIM use full-image RGB comparisons, resizing predictions to target dimensions when needed. LPIPS resizes images to 256×256 and matches prediction directories by filename; those directories must contain the intended evaluation samples. The target is InpaintCOCO's edited `inpaint_image`.

[Historical CSVs](results/metrics/historical/) · [PSNR plot](assets/psnr_bar.png) · [SSIM plot](assets/ssim_bar.png) · [LPIPS plot](assets/lpips_bar.png)

The results illustrate a **trade-off between structural fidelity and perceptual quality**. The hybrid does not universally outperform the baselines, and no measured speedup is claimed. Historical results have not been regenerated or overwritten. See [asset provenance](assets/README.md) for the selected figure's outstanding attribution details.

## Setup and basic usage

Run commands from the repository root:

```sh
git clone https://github.com/ELSIEE1/image-inpainting.git
cd image-inpainting
python -m venv .venv
# Activate: .venv\Scripts\Activate.ps1 (PowerShell)
# Or: source .venv/bin/activate (Linux/macOS)
python -m pip install -r requirements.txt
python scripts/check_repository.py
```

Install a matching PyTorch/torchvision pair for your hardware. Exact historical dependency versions were not recorded; requirements are not a validated lockfile. Import/help checks were verified, but full model inference was not validated in the cleanup environment. See [validation details](docs/validation.md).

**Before inference:** export the dataset using the command below and supply verified EdgeConnect generator checkpoints at `checkpoints/edge_connect/`. Both diffusion paths retain `runwayml/stable-diffusion-inpainting`; `--model` accepts a verified local model directory. Weights are not included, exact checkpoint provenance remains unconfirmed, and current access to the original diffusion identifier has not been established. See [model setup](docs/pretrained_models.md) and the [detailed usage guide](docs/usage.md).

```sh
python -m src.utils.export_dataset --output-dir data/InpaintCOCO_export
python -m src.gan.run_edgeconnect --dry-run
python -m src.gan.run_edgeconnect --mode test
python -m src.diffusion.run_sd_inpaintcoco
python -m src.hybrid.run_hybrid
```

Export only to a new dataset location; do not overwrite an existing experiment. The hybrid requires GAN outputs first and skips existing hybrid filenames. Outputs are saved under `results/gan_inpaintcoco/`, `results/sd_inpaintcoco/`, and `results/hybrid_inpaintcoco/`. Paths are configurable through each command's `--help`.

Evaluate predictions without replacing historical CSVs:

```sh
python -m src.evaluation.eval_metrics --gt_dir data/InpaintCOCO_export/gt --pred_dir results/hybrid_inpaintcoco --list_file splits/test_100.txt --out_csv results/metrics/new/hybrid_metrics.csv
python -m src.evaluation.compute_lpips --output-dir results/metrics/new
```

LPIPS evaluates all three prediction directories by default. See the [usage guide](docs/usage.md) for baseline evaluation, plotting, demo mask conventions, and training prerequisites.

## Repository structure

```text
src/gan/                    # Portable EdgeConnect launcher
src/diffusion/              # Stable Diffusion baseline and single-image demo
src/hybrid/                 # GAN output loading and diffusion refinement
src/evaluation/             # PSNR, SSIM, LPIPS, and plotting
src/utils/                  # Dataset export and comparison figures
third_party/edge_connect/   # Existing GAN implementation and original license
configs/                    # GAN test configuration
scripts/                    # Isolated GAN worker and repository checks
splits/                     # Preserved evaluation manifest
assets/                     # Pipeline, qualitative example, metric plots
results/metrics/historical/ # Six original experiment CSVs
docs/                       # Usage, methodology, provenance, and validation
```

Datasets, model weights, full generated outputs, caches, environments, credentials, and local recovery material are excluded through [.gitignore](.gitignore) and remain local.

## Attribution and reproducibility

EdgeConnect is adapted from [knazeri/edge-connect](https://github.com/knazeri/edge-connect), with its original [CC BY-NC 4.0 license](third_party/edge_connect/LICENSE.md), README, and local-change record preserved. Stable Diffusion is used through Hugging Face Diffusers. Dataset export uses `phiyodr/InpaintCOCO`. See [third-party attribution](docs/third_party.md) for component provenance and model/asset terms.

Original prompts, preprocessing, and metric formulas are preserved. Missing diffusion seeds and unrecorded dataset/model revisions limit exact replication. The main [LICENSE](LICENSE) reserves original-contribution rights pending a license selection; it does not relicense third-party materials. [CITATION.cff](CITATION.cff) records the thesis title; final author/institution/year details remain to be confirmed.
