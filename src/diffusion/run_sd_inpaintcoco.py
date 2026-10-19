import os

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROMPT = "realistic filling of missing area, consistent with surroundings"
INVERT_MASK = False   # 如果你确认方向反了，再改 True
SIZE = (512, 512)

def main():
    parser = argparse.ArgumentParser(description="Existing Stable Diffusion baseline; no seed imposed.")
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/InpaintCOCO_export')
    parser.add_argument('--list-file', type=Path, default=ROOT/'splits/test_100.txt')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/sd_inpaintcoco')
    parser.add_argument('--model', default='runwayml/stable-diffusion-inpainting')
    args = parser.parse_args()
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image, ImageOps
    DATA_ROOT, TEST_TXT, OUT_DIR = args.data_root, args.list_file, args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    with open(TEST_TXT, "r", encoding="utf-8") as f:
        lines = [x.strip().split() for x in f if x.strip()]

    for (img_rel, msk_rel, _) in lines:
        name = os.path.basename(img_rel)

        init_image = Image.open(os.path.join(DATA_ROOT, img_rel)).convert("RGB").resize(SIZE)
        mask = Image.open(os.path.join(DATA_ROOT, msk_rel)).convert("L").resize(SIZE)

        if INVERT_MASK:
            mask = ImageOps.invert(mask)

        if device == "cuda":
            ctx = torch.autocast("cuda")
        else:
            ctx = torch.no_grad()

        with ctx:
            result = pipe(
                prompt=PROMPT,
                image=init_image,
                mask_image=mask,
            ).images[0]

        result.save(os.path.join(OUT_DIR, name))

    print("Done. Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
