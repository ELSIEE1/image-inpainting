# hybrid/gan_infer.py
from PIL import Image
import os

class GANResultLoader:
    def __init__(self, gan_result_dir):
        self.dir = gan_result_dir

    def load(self, name):
        """
        name: '0000.png'
        """
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return Image.open(path).convert("RGB")
