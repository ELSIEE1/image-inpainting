import os
from PIL import Image

# 你选的样本
IDS = ["0004", "0022", "0019", "0044", "0023", "0052", "0062", "0073"]

def load_rgb(p):
    return Image.open(p).convert("RGB")

def load_gray(p):
    return Image.open(p).convert("L")

def make_masked(img, mask):
    """
    masked image = 原图中被 mask 区域置灰（或黑）
    """
    img = img.copy()
    mask = mask.resize(img.size)

    pixels = img.load()
    mask_px = mask.load()

    for y in range(img.height):
        for x in range(img.width):
            if mask_px[x, y] > 0:
                pixels[x, y] = (128, 128, 128)  # 灰色遮挡

    return img


def main():
    import argparse
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Original five-column comparison figure.')
    parser.add_argument('--data-root', type=Path, default=root/'data/InpaintCOCO_export')
    parser.add_argument('--results-dir', type=Path, default=root/'results')
    parser.add_argument('--output-dir', type=Path, default=root/'results/figures/compare_final')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    IMG_DIR, MASK_DIR, GT_DIR = (args.data_root / x for x in ('images', 'masks', 'gt'))
    GAN_DIR = args.results_dir / 'gan_inpaintcoco'
    SD_DIR = args.results_dir / 'sd_inpaintcoco'
    HYB_DIR = args.results_dir / 'hybrid_inpaintcoco'
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx in IDS:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))

        img  = load_rgb(f"{IMG_DIR}/{idx}.png")
        mask = load_gray(f"{MASK_DIR}/{idx}.png")

        masked_img = make_masked(img, mask)

        images = [
            masked_img,
            load_rgb(f"{GAN_DIR}/{idx}.png"),
            load_rgb(f"{SD_DIR}/{idx}.png"),
            load_rgb(f"{HYB_DIR}/{idx}.png"),
            load_rgb(f"{GT_DIR}/{idx}.png"),
        ]

        titles = ["Masked Input", "GAN", "Diffusion", "Hybrid (Ours)", "GT"]

        for ax, im, title in zip(axes, images, titles):
            ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{idx}_compare.png", dpi=300)
        plt.close()

    print("Saved final comparison figures.")

if __name__ == '__main__':
    main()
