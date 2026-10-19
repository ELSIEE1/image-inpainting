# Examples

Run the single-image demo using inputs you have permission to use:

```sh
python -m src.diffusion.baseline_sd_inpaint --image /path/to/input.jpg --mask /path/to/mask.jpg --output-dir results/demo
```

This original demo **inverts** its grayscale mask before resizing to 512×512. The batch baseline does not invert the mask. Do not interchange mask conventions silently.

No standalone dataset/demo photographs are redistributed here. The original local demo files remain in the ignored `.local/diffusion_demo/` recovery directory. The selected comparison is under `assets/`, with provenance limitations documented there.
