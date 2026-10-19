import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description="Refine saved GAN results; preserves original prompt and unseeded behavior.")
    parser.add_argument('--gan-dir', type=Path, default=ROOT/'results/gan_inpaintcoco')
    parser.add_argument('--mask-dir', type=Path, default=ROOT/'data/InpaintCOCO_export/masks')
    parser.add_argument('--list-file', type=Path, default=ROOT/'splits/test_100.txt')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/hybrid_inpaintcoco')
    parser.add_argument('--model', default='runwayml/stable-diffusion-inpainting')
    args = parser.parse_args()
    from .gan_infer import GANResultLoader
    from .sd_refine import SDRefiner
    from PIL import Image
    GAN_DIR, MASK_DIR = args.gan_dir, args.mask_dir
    OUT_DIR, LIST_FILE = args.output_dir, args.list_file
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load GAN results
    gan_loader = GANResultLoader(GAN_DIR)

    # Load Stable Diffusion Inpainting
    sd = SDRefiner(args.model)

    # Read test list
    with open(LIST_FILE, "r") as f:
        lines = f.readlines()

    # Hybrid inference loop
    # Hybrid inference loop
    for line in lines:
        img_path, mask_path, gt_path = line.strip().split()
        name = os.path.basename(img_path)  # e.g. 0000.png

        out_path = os.path.join(OUT_DIR, name)
        if os.path.exists(out_path):
            print("Skip existing:", name)
            continue

        print("Hybrid processing:", name)

        # Load GAN inpainting result
        gan_img = gan_loader.load(name)

        # Load mask
        mask = Image.open(os.path.join(MASK_DIR, os.path.basename(mask_path))).convert("L")

        # Diffusion refinement
        hybrid_img = sd.refine(
            image=gan_img,
            mask=mask,
            prompt="A high-quality realistic image",
            steps=50,
            guidance=7.5
        )

        # Save result
        hybrid_img.save(os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
