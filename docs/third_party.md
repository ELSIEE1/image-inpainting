# Third-party attribution and scope

## EdgeConnect

- Upstream repository: https://github.com/knazeri/edge-connect
- Recorded base commit: `3142579785c5c8cc92ad655616d40f4f52fc7ec1`
- Code location: `third_party/edge_connect/`
- License: original `third_party/edge_connect/LICENSE.md`, CC BY-NC 4.0, retained verbatim.
- Original README and citations remain in `third_party/edge_connect/README.md`.

The pre-cleanup checkout already differed from upstream in `config.yml.example`, `src/config.py`, `src/dataset.py`, `src/edge_connect.py`, and `test.py`. `edgeconnect_original_local_changes.patch` preserves those exact text changes for review, excluding Git metadata. They include YAML-loader compatibility, image I/O/resizing and dtype changes, random augmentation/mask-selection changes, test configuration, output handling, and the custom test launcher. These pre-existing changes were not reverted or redesigned during cleanup.

Cleanup adds only a single-channel mask guard inside vendor `src/dataset.py`. The other vendor source files remain identical to their pre-cleanup local versions. New first-party launchers initialize runtime/device/seed and isolate imports outside the vendor tree. Vendor example images, generated outputs, caches, weights, and nested `.git` metadata are retained locally but excluded from publication.

## Additional provenance references within upstream code

The EdgeConnect README identifies its FID implementation as adapted from https://github.com/mseitzer/pytorch-fid. `src/networks.py` references CycleGAN/pix2pix initialization code; `src/loss.py` references dxyang/StyleTransfer utilities. Their original references are preserved. This cleanup does not assert that EdgeConnect's top-level license replaces those components' terms; component-specific licensing/provenance should be confirmed before release. Legacy FID scripts are preserved for attribution/history and are not advertised as a validated thesis evaluation command.

## Dataset and model assets

The dataset identifier `phiyodr/InpaintCOCO` was confirmed against its Hugging Face dataset page. The current dataset card describes a test split of 1,260 rows and COCO-derived and edited images. The original exporter mapping and local split are preserved; dataset revision and per-image rights remain unconfirmed. See `assets/README.md` for the selected sample's attribution gap.

Stable Diffusion, VGG19, AlexNet, LPIPS, and dependencies retain their respective upstream terms. No third-party weights or datasets are included. The repository's main `LICENSE` does not relicense these materials. Original-contribution licensing and thesis author information await the author's selection/confirmation.
