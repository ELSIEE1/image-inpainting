import os
import numpy as np
from PIL import Image

def mask_to_binary(mask_pil: Image.Image, thr=10) -> Image.Image:
    # mask 原本是 RGB 图；转灰度再阈值化 -> 单通道 0/255
    m = mask_pil.convert("L")
    arr = np.array(m)
    bin_arr = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(bin_arr, mode="L")


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Export InpaintCOCO test split using original column mapping and mask threshold.')
    parser.add_argument('--output-dir', type=Path, default=Path('data/InpaintCOCO_export'))
    parser.add_argument('--revision', default=None, help='Dataset revision; original revision was not recorded.')
    args = parser.parse_args()
    from datasets import load_dataset
    OUT_ROOT = args.output_dir
    IMG_DIR = OUT_ROOT / 'images'
    MSK_DIR = OUT_ROOT / 'masks'
    GT_DIR = OUT_ROOT / 'gt'
    for directory in (IMG_DIR, MSK_DIR, GT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('phiyodr/InpaintCOCO', split='test', revision=args.revision)
    pairs = []

    for i in range(len(ds)):
        img = ds[i]["coco_image"]       # 输入图
        gt  = ds[i]["inpaint_image"]    # 目标图
        msk = ds[i]["mask"]             # 编辑区域mask

        name = f"{i:04d}.png"           # 统一文件名（非常重要）
        img.save(os.path.join(IMG_DIR, name))
        gt.save(os.path.join(GT_DIR, name))
        mask_to_binary(msk).save(os.path.join(MSK_DIR, name))

        pairs.append((f"images/{name}", f"masks/{name}", f"gt/{name}"))

    # test.txt（三列：image mask gt）
    with open(os.path.join(OUT_ROOT, "test.txt"), "w", encoding="utf-8") as f:
        for a, b, c in pairs:
            f.write(f"{a} {b} {c}\n")

    # EdgeConnect 的 flist 通常用“绝对路径”最省事
    abs_img_dir = os.path.abspath(IMG_DIR)
    abs_msk_dir = os.path.abspath(MSK_DIR)

    with open(os.path.join(OUT_ROOT, "images.flist"), "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            f.write(os.path.join(abs_img_dir, f"{i:04d}.png") + "\n")

    with open(os.path.join(OUT_ROOT, "masks.flist"), "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            f.write(os.path.join(abs_msk_dir, f"{i:04d}.png") + "\n")

    print("Export done:", OUT_ROOT, "num=", len(ds))

if __name__ == '__main__':
    main()
