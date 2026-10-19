import csv
from statistics import mean

def load_metric(csv_path, key):
    values = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("name") != "mean" and row.get(key):
                values.append(float(row[key]))
    return values


def main():
    import argparse
    from pathlib import Path
    from .output import protect_historical
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Plot preserved PSNR/SSIM CSVs; does not recompute metrics.')
    parser.add_argument('--metrics-dir', type=Path, default=root/'results/metrics/historical')
    parser.add_argument('--output-dir', type=Path, default=root/'results/plots')
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    protect_historical(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gan_csv = args.metrics_dir / 'gan_metrics.csv'
    sd_csv = args.metrics_dir / 'sd_metrics.csv'
    hybrid_csv = args.metrics_dir / 'hybrid_metrics.csv'
    # ===== Load data =====
    gan_psnr = load_metric(gan_csv, "psnr")
    sd_psnr = load_metric(sd_csv, "psnr")
    hybrid_psnr = load_metric(hybrid_csv, "psnr")

    gan_ssim = load_metric(gan_csv, "ssim")
    sd_ssim = load_metric(sd_csv, "ssim")
    hybrid_ssim = load_metric(hybrid_csv, "ssim")

    models = ["GAN", "Diffusion", "Hybrid"]

    # ===== Mean values =====
    psnr_means = [mean(gan_psnr), mean(sd_psnr), mean(hybrid_psnr)]
    ssim_means = [mean(gan_ssim), mean(sd_ssim), mean(hybrid_ssim)]

    # =========================
    # 1. PSNR bar chart
    # =========================
    plt.figure(figsize=(6, 4))
    plt.bar(models, psnr_means)
    plt.title("PSNR Comparison")
    plt.ylabel("PSNR")

    for i, v in enumerate(psnr_means):
        plt.text(i, v + 0.2, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(args.output_dir / "psnr_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # 2. SSIM bar chart
    # =========================
    plt.figure(figsize=(6, 4))
    plt.bar(models, ssim_means)
    plt.title("SSIM Comparison")
    plt.ylabel("SSIM")

    for i, v in enumerate(ssim_means):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(args.output_dir / "ssim_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # 3. PSNR boxplot
    # =========================
    plt.figure(figsize=(6, 4))
    plt.boxplot([gan_psnr, sd_psnr, hybrid_psnr], tick_labels=models)
    plt.title("PSNR Distribution")
    plt.ylabel("PSNR")
    plt.tight_layout()
    plt.savefig(args.output_dir / "psnr_box.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # 4. SSIM boxplot
    # =========================
    plt.figure(figsize=(6, 4))
    plt.boxplot([gan_ssim, sd_ssim, hybrid_ssim], tick_labels=models)
    plt.title("SSIM Distribution")
    plt.ylabel("SSIM")
    plt.tight_layout()
    plt.savefig(args.output_dir / "ssim_box.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Finished.")
    print("Saved files:")
    print(" - psnr_bar.png")
    print(" - ssim_bar.png")
    print(" - psnr_box.png")
    print(" - ssim_box.png")
    print()
    print("Mean values:")
    print(f"GAN      PSNR={psnr_means[0]:.4f}, SSIM={ssim_means[0]:.4f}")
    print(f"Diffusion PSNR={psnr_means[1]:.4f}, SSIM={ssim_means[1]:.4f}")
    print(f"Hybrid   PSNR={psnr_means[2]:.4f}, SSIM={ssim_means[2]:.4f}")
if __name__ == '__main__':
    main()
