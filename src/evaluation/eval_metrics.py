import os
import argparse
import csv
import numpy as np
from PIL import Image



def load_rgb_01(path: str) -> np.ndarray:
    """Load image as RGB float32 in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def resize_to(img_arr: np.ndarray, size_wh: tuple) -> np.ndarray:
    """Resize numpy RGB [0,1] to (W,H) using PIL."""
    w, h = size_wh
    pil = Image.fromarray((img_arr * 255.0).astype(np.uint8))
    pil = pil.resize((w, h), resample=Image.BICUBIC)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr


def parse_list(list_file: str):
    """
    list_file format: 'images/0000.png masks/0000.png gt/0000.png'
    We only need the filename (0000.png).
    """
    names = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            # Use the image name from the first column
            name = os.path.basename(parts[0])
            names.append(name)
    return names


def main(args):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from .output import protect_historical
    protect_historical(args.out_csv)
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir
    list_file = args.list_file
    out_csv = args.out_csv

    assert os.path.exists(gt_dir), f"GT dir not found: {gt_dir}"
    assert os.path.exists(pred_dir), f"Pred dir not found: {pred_dir}"
    assert os.path.exists(list_file), f"List file not found: {list_file}"

    names = parse_list(list_file)

    rows = []
    psnrs, ssims = [], []

    for name in names:
        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)

        if not os.path.exists(gt_path):
            print(f"[skip] missing gt: {gt_path}")
            continue
        if not os.path.exists(pred_path):
            print(f"[skip] missing pred: {pred_path}")
            continue

        gt = load_rgb_01(gt_path)
        pred = load_rgb_01(pred_path)

        # Align sizes: resize pred to gt if needed
        if gt.shape != pred.shape:
            h, w = gt.shape[0], gt.shape[1]
            pred = resize_to(pred, (w, h))

        psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)

        psnrs.append(psnr)
        ssims.append(ssim)
        rows.append([name, psnr, ssim])

    mean_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    mean_ssim = float(np.mean(ssims)) if ssims else float("nan")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "psnr", "ssim"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["mean", mean_psnr, mean_ssim])

    print(f"Evaluated: {len(rows)} images")
    print(f"Mean PSNR: {mean_psnr:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--list_file", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    main(args)
