# hybrid/sd_refine.py

class SDRefiner:
    def __init__(self, model_name):
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        # 自动选择 device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device=="cuda" else torch.float32
        ).to(self.device)

    def refine(self, image, mask, prompt, steps=50, guidance=7.5):
        """
        image: PIL Image
        mask: PIL Image
        prompt: str
        steps: inference steps
        guidance: guidance scale
        """
        out = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]  # pipeline 返回对象里 images 是 list
        return out

