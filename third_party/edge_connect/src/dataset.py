import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import imageio
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from skimage.transform import resize as sk_resize
from .utils import create_mask

imread = imageio.v2.imread


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

        # test mode: mask fixed per image
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        return os.path.basename(self.data[index])

    def load_item(self, index):
        size = self.input_size

        # load image
        img = imread(self.data[index])

        # gray → rgb
        if img.ndim == 2:
            img = gray2rgb(img)

        # resize
        if size != 0:
            img = self.resize(img, size, size)

        img_gray = rgb2gray(img)

        mask = self.load_mask(img, index)
        edge = self.load_edge(img_gray, index, mask)

        # horizontal flip
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1]
            img_gray = img_gray[:, ::-1]
            edge = edge[:, ::-1]
            mask = mask[:, ::-1]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # avoid deprecated np.bool
        mask_bool = None if self.training else (1 - mask / 255).astype(bool)

        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask_bool).astype(float)

        # external edge
        else:
            imgh, imgw = img.shape[:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask_bool)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[:2]
        mask_type = self.mask

        if mask_type == 4:
            mask_type = random.choice([1, 3])

        elif mask_type == 5:
            mask_type = random.randint(1, 3)

        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        if mask_type == 2:
            return create_mask(imgw, imgh, imgw//2, imgh,
                               0 if random.random() < 0.5 else imgw//2, 0)

        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # Portability: exported L-mode masks already have one channel.
            # Preserve the original RGB conversion and >0 threshold otherwise.
            if mask.ndim == 3:
                mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img.astype('uint8'))
        return F.to_tensor(img).float()

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[:2]

        if centerCrop and imgh != imgw:
            side = min(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j+side, i:i+side]

        img = sk_resize(img, (height, width), preserve_range=True, anti_aliasing=True)
        return img.astype(np.float32)

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                files = glob.glob(flist + '/*.jpg') + glob.glob(flist + '/*.png')
                return sorted(files)

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            loader = DataLoader(self, batch_size=batch_size, drop_last=True)
            for item in loader:
                yield item
