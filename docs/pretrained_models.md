# Pretrained models and provenance

## EdgeConnect

The source checkout identifies upstream as https://github.com/knazeri/edge-connect.git at commit `3142579785c5c8cc92ad655616d40f4f52fc7ec1`. Its original README, license, requirements, and download script are preserved as upstream records.

Four local `.pth` files were found and retained in ignored `checkpoints/edge_connect/`. `checkpoint_inventory.json` records their bytes, SHA-256, and roles. Generator weights are required for inference; discriminator weights are used for training. All are excluded from the public commit, regardless of individual file size.

**Provenance confirmation required:** the particular training dataset, original archive/download, training history, and relationship of these exact local files to an upstream release are not established. The upstream README lists pretrained resources, but they have not been verified as matching these local files. No download URL is promised to reproduce them. Do not run the legacy download script expecting a verified exact match.

The current EdgeConnect constructors instantiate perceptual/style loss components, which request pretrained VGG19 through torchvision even for inference. This behavior is retained; a cold environment may need network access and a substantial additional model cache.

## Stable Diffusion

Original source identifier: `runwayml/stable-diffusion-inpainting`. Retained in both batch paths and the demo, with `--model` accepting a verified local model directory or confirmed identifier. Its original revision and cached weight hashes were not recorded. The Hugging Face page request during cleanup returned HTTP 401, so current access/download availability is **unverified**. No replacement or invented download URL was introduced.

Obtain an authorized copy from its confirmed source, review its model license, and record the model revision/hash before reporting a newly reproducible experiment. There are no Stable Diffusion model files in this repository.

## LPIPS

The existing evaluator uses `lpips.LPIPS(net='alex')`. LPIPS and torchvision may fetch/package pretrained calibration and AlexNet weights depending on the installed versions and caches. They are not committed. Their exact historical versions/hashes are unrecorded. Run evaluation only after installing the listed dependencies and obtaining their required weights.
