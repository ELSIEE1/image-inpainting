from pathlib import Path
import re
import csv
from statistics import mean
from PIL import Image

def load_image(path: Path):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))   # 关键：统一尺寸
    return to_tensor(img).unsqueeze(0).to(device)

def extract_id(path: Path):
    stem = path.stem
    match = re.match(r"(\d+)", stem)
    return match.group(1) if match else None

def build_gt_map(gt_dir: Path):
    gt_map = {}
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in gt_dir.glob(ext):
            img_id = extract_id(p)
            if img_id is not None:
                gt_map[img_id] = p
    return gt_map

def is_valid_result_file(path: Path):
    name = path.stem.lower()
    if "_edge" in name or "_masked" in name or "_mask" in name:
        return False
    return True

def compute_lpips(pred_dir: Path, gt_map: dict, model_name: str):
    rows = []

    if not pred_dir.exists():
        print(f"[ERROR] Prediction directory not found: {pred_dir}")
        return rows

    pred_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        pred_files.extend(pred_dir.glob(ext))
    pred_files = sorted(pred_files)

    print(f"[{model_name}] Total files found: {len(pred_files)}")

    valid_files = [p for p in pred_files if is_valid_result_file(p)]
    print(f"[{model_name}] Valid result files: {len(valid_files)}")

    for pred_path in valid_files:
        img_id = extract_id(pred_path)
        if img_id is None:
            print(f"[{model_name}] Skip (cannot parse id): {pred_path.name}")
            continue

        if img_id not in gt_map:
            print(f"[{model_name}] Missing GT for id {img_id} ({pred_path.name})")
            continue

        gt_path = gt_map[img_id]

        try:
            pred = load_image(pred_path)
            gt = load_image(gt_path)

            with torch.no_grad():
                score = loss_fn(pred, gt).item()

            rows.append({
                "image_id": img_id,
                "pred_file": pred_path.name,
                "gt_file": gt_path.name,
                "lpips": score
            })

        except Exception as e:
            print(f"[{model_name}] Error processing {pred_path.name}: {e}")

    print(f"[{model_name}] Matched pairs: {len(rows)}")
    return rows

def save_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "pred_file", "gt_file", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

def print_mean(model_name, rows):
    if len(rows) == 0:
        print(f"{model_name} mean LPIPS: No valid matched files")
    else:
        scores = [r["lpips"] for r in rows]
        print(f"{model_name} mean LPIPS: {mean(scores):.6f}")

def plot_lpips_bar(gan_rows, diff_rows, hybrid_rows):
    import matplotlib.pyplot as plt
    # 计算均值
    gan_mean = mean([r["lpips"] for r in gan_rows]) if gan_rows else 0
    diff_mean = mean([r["lpips"] for r in diff_rows]) if diff_rows else 0
    hybrid_mean = mean([r["lpips"] for r in hybrid_rows]) if hybrid_rows else 0

    models = ["GAN", "Diffusion", "Hybrid"]
    values = [gan_mean, diff_mean, hybrid_mean]

    plt.figure()
    plt.bar(models, values)
    plt.title("LPIPS Comparison (Lower is Better)")
    plt.xlabel("Models")
    plt.ylabel("LPIPS")

    # 在柱子上标数值（论文很加分）
    for i, v in enumerate(values):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center')

    plt.savefig(plot_path)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Original LPIPS evaluation: AlexNet, RGB, 256x256, [-1,1].")
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--gan-dir', type=Path, default=root/'results/gan_inpaintcoco')
    parser.add_argument('--diffusion-dir', type=Path, default=root/'results/sd_inpaintcoco')
    parser.add_argument('--hybrid-dir', type=Path, default=root/'results/hybrid_inpaintcoco')
    parser.add_argument('--gt-dir', type=Path, default=root/'data/InpaintCOCO_export/gt')
    parser.add_argument('--output-dir', type=Path, default=root/'results/metrics/new')
    args = parser.parse_args()
    from .output import protect_historical
    protect_historical(args.output_dir)
    for directory in (args.gan_dir, args.diffusion_dir, args.hybrid_dir, args.gt_dir):
        if not directory.is_dir():
            parser.error(f'Missing directory: {directory}')
    import lpips
    global torch, device, loss_fn, to_tensor, plot_path
    import torch
    import torchvision.transforms as transforms
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = lpips.LPIPS(net='alex').to(device)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gan_result_dir, diff_result_dir = args.gan_dir, args.diffusion_dir
    hybrid_result_dir, gt_dir = args.hybrid_dir, args.gt_dir
    plot_path = args.output_dir / 'lpips_bar.png'
    # ===== main =====
    gt_map = build_gt_map(gt_dir)
    print("GT files mapped:", len(gt_map))

    gan_rows = compute_lpips(gan_result_dir, gt_map, "GAN")
    diff_rows = compute_lpips(diff_result_dir, gt_map, "Diffusion")
    hybrid_rows = compute_lpips(hybrid_result_dir, gt_map, "Hybrid")

    print("\n===== Mean LPIPS (lower is better) =====")
    print_mean("GAN", gan_rows)
    print_mean("Diffusion", diff_rows)
    print_mean("Hybrid", hybrid_rows)

    save_csv(args.output_dir / "gan_lpips.csv", gan_rows)
    save_csv(args.output_dir / "diffusion_lpips.csv", diff_rows)
    save_csv(args.output_dir / "hybrid_lpips.csv", hybrid_rows)

    print("\nSaved:")
    print(" - gan_lpips.csv")
    print(" - diffusion_lpips.csv")
    print(" - hybrid_lpips.csv")

    plot_lpips_bar(gan_rows, diff_rows, hybrid_rows)

if __name__ == '__main__':
    main()
