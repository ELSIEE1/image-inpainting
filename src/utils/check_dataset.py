from PIL import Image

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Preview the first exported image and mask.')
    parser.add_argument('--data-root', type=Path, default=Path('data/InpaintCOCO_export'))
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    img  = Image.open(args.data_root / "images/0000.png")
    mask = Image.open(args.data_root / "masks/0000.png")

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
