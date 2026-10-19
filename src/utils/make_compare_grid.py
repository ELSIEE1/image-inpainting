import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def read_list_names(list_file: str):
    names = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            # first column: images/0000.png
            name = os.path.basename(parts[0])
            names.append(name)
    return names

def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_mask(path: str) -> Image.Image:
    # assume L mask 0/255
    return Image.open(path).convert("L")

def resize_to(img: Image.Image, size_wh):
    return img.resize(size_wh, resample=Image.BICUBIC)

def mask_overlay(image_rgb: Image.Image, mask_l: Image.Image, alpha=0.45) -> Image.Image:
    """Overlay mask region with a solid color (red) on top of image."""
    img = image_rgb.copy()
    m = mask_l.resize(img.size, resample=Image.NEAREST)
    m_arr = np.array(m) > 127

    overlay = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)
    overlay[..., 0] = 255  # red
    overlay_img = Image.fromarray(overlay, mode="RGB")

    # blend only on masked region
    img_arr = np.array(img).astype(np.float32)
    ov_arr = np.array(overlay_img).astype(np.float32)
    out = img_arr.copy()
    out[m_arr] = (1 - alpha) * img_arr[m_arr] + alpha * ov_arr[m_arr]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

def add_title(img: Image.Image, title: str, bar_h=34) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    canvas.paste(img, (0, bar_h))

    draw = ImageDraw.Draw(canvas)
    # Use default font (cross-platform safe)
    draw.text((8, 8), title, fill=(0, 0, 0))
    return canvas

def hstack(images, pad=8, bg=(255,255,255)):
    widths = [im.size[0] for im in images]
    heights = [im.size[1] for im in images]
    W = sum(widths) + pad * (len(images) - 1)
    H = max(heights)
    out = Image.new("RGB", (W, H), bg)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.size[0] + pad
    return out

def vstack(rows, pad=8, bg=(255,255,255)):
    widths = [r.size[0] for r in rows]
    heights = [r.size[1] for r in rows]
    W = max(widths)
    H = sum(heights) + pad * (len(rows) - 1)
    out = Image.new("RGB", (W, H), bg)
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.size[1] + pad
    return out

def main(args):
    project = os.path.abspath(args.project_dir)

    data_root = os.path.join(project, "data", "InpaintCOCO_export")
    img_dir  = os.path.join(data_root, "images")
    msk_dir  = os.path.join(data_root, "masks")
    gt_dir   = os.path.join(data_root, "gt")

    gan_dir = os.path.join(project, "results", "gan_inpaintcoco")
    sd_dir  = os.path.join(project, "results", "sd_inpaintcoco")

    list_file = os.path.abspath(args.list_file)
    names_all = read_list_names(list_file)

    # choose which ids
    if args.ids:
        # ids like 0,1,2 or 0001,0017
        names = []
        for s in args.ids:
            s = s.strip()
            if s.isdigit() and len(s) <= 4:
                names.append(f"{int(s):04d}.png")
            else:
                if not s.endswith(".png"):
                    s = s + ".png"
                names.append(s)
    else:
        # random sample
        random.seed(args.seed)
        names = random.sample(names_all, k=min(args.num, len(names_all)))

    os.makedirs(args.out_dir, exist_ok=True)

    for name in names:
        p_img = os.path.join(img_dir, name)
        p_msk = os.path.join(msk_dir, name)
        p_gt  = os.path.join(gt_dir, name)
        p_gan = os.path.join(gan_dir, name)
        p_sd  = os.path.join(sd_dir, name)

        missing = [p for p in [p_img, p_msk, p_gt, p_gan, p_sd] if not os.path.exists(p)]
        if missing:
            print(f"[skip] {name} missing files:\n  " + "\n  ".join(missing))
            continue

        img = load_rgb(p_img)
        msk = load_mask(p_msk)
        gt  = load_rgb(p_gt)
        gan = load_rgb(p_gan)
        sd  = load_rgb(p_sd)

        # unify tile size: use GT size as reference
        W, H = gt.size
        size = (W, H)

        img_r = resize_to(img, size)
        msk_r = msk.resize(size, resample=Image.NEAREST)
        gan_r = resize_to(gan, size)
        sd_r  = resize_to(sd, size)

        overlay = mask_overlay(img_r, msk_r, alpha=0.45)

        # masked image: set masked region to white (for visualization)
        img_arr = np.array(img_r).copy()
        m_arr = (np.array(msk_r) > 127)
        img_arr[m_arr] = 255
        masked = Image.fromarray(img_arr, mode="RGB")

        # build 2 rows x 3 cols
        tiles = [
            add_title(img_r,     f"Input ({name[:-4]})"),
            add_title(overlay,   "Mask overlay"),
            add_title(masked,    "Masked input"),
            add_title(gan_r,     "GAN (EdgeConnect)"),
            add_title(sd_r,      "Diffusion (SD inpaint)"),
            add_title(gt,        "GT"),
        ]
        row1 = hstack(tiles[0:3], pad=10)
        row2 = hstack(tiles[3:6], pad=10)
        grid = vstack([row1, row2], pad=12)

        out_path = os.path.join(args.out_dir, f"compare_{name[:-4]}.png")
        grid.save(out_path)
        print("saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=".", help="capstone_model root dir")
    ap.add_argument("--list_file", default="splits/test_100.txt", help="path to preserved split")
    ap.add_argument("--out_dir", default="results/figures/compare", help="output dir")
    ap.add_argument("--num", type=int, default=8, help="random samples if --ids not set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", nargs="*", help="specific ids: e.g. 0 1 2 or 0001 0017 0099")
    args = ap.parse_args()
    main(args)
