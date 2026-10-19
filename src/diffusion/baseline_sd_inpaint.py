import argparse
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(description="Original single-image demo (inverts the mask).")
    parser.add_argument('--image', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/demo'))
    parser.add_argument('--model', default='runwayml/stable-diffusion-inpainting')
    args = parser.parse_args()
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image, ImageOps
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取并反转 mask（白=修补，黑=保留）
    mask_fixed = Image.open(args.mask).convert("L")
    mask_fixed = ImageOps.invert(mask_fixed)
    mask_fixed = mask_fixed.resize((512, 512))
    mask_fixed.save(args.output_dir / "mask_fixed.jpg")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # 2. 加载 Inpainting 模型
    model_id = args.model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # 3. 加载输入图
    init_image = Image.open(args.image).convert("RGB").resize((512, 512))

    prompt = "realistic filling of missing area, consistent with surroundings"

    # 4. 推理
    with torch.autocast(device) if device == "cuda" else torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_fixed,
        ).images[0]

    # 5. 保存结果
    result.save(args.output_dir / "sd_inpaint_output.png")
    print("Saved result to sd_inpaint_output.png")


if __name__ == "__main__":
    main()
