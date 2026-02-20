from pathlib import Path
import os
import numpy as np
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset
from utils import image

import random
from typing import List

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import av
from torch import nn


class BicubicDownsample(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.reformatter = av.video.reformatter.VideoReformatter()

    def __call__(self, img: torch.Tensor):
        # Convert the tensor to numpy array and switch from C,H,W to H,W,C
        img_np = img.mul(255).byte().permute(1, 2, 0).numpy()

        # Create a VideoFrame from the numpy array
        frame = av.VideoFrame.from_ndarray(img_np, format="rgb24")

        # Get the new width and height
        new_width = frame.width // self.scale
        new_height = frame.height // self.scale

        # Downsample the frame
        frame = self.reformatter.reformat(frame, width=new_width, height=new_height, format=None,
                                          src_colorspace=None, dst_colorspace=None,
                                          interpolation=av.video.reformatter.Interpolation.BICUBIC)

        # Convert the downsampled frame back to a numpy array
        img_lr_np = frame.to_ndarray(format="rgb24")

        # Convert the numpy array back to tensor and normalize
        img_lr = torch.from_numpy(img_lr_np).permute(2, 0, 1).float().div(255)

        # The HR image needs to be cropped to be divisible by the scale factor
        C, H, W = img.shape
        H_r, W_r = H % self.scale, W % self.scale
        img_hr = img[:, :H - H_r, :W - W_r]

        img_lr_pil = Image.fromarray((img_lr_np).astype('uint8'))  # Convert numpy to PIL Image
        lr_save_path = os.path.join("/localhome/gba50/Finalize/gelkari/downsample/bicubic", 'lr_image.png')  # Define the save path
        img_lr_pil.save(lr_save_path)  # Save the low-resolution image

        # Save the high-resolution image (cropped)
        img_hr_np = img_hr.permute(1, 2, 0).mul(255).byte().numpy()  # Convert tensor to numpy
        img_hr_pil = Image.fromarray(img_hr_np.astype('uint8'))  # Convert numpy to PIL Image
        hr_save_path = os.path.join("/localhome/gba50/Finalize/gelkari/downsample/bicubic", 'hr_image.png')  # Define the save path
        img_hr_pil.save(hr_save_path)  # Save the high-resolution image

        return img_lr, img_hr
        return lr, img


class RandomRotation:
    def __init__(self,
                 percentage: float = 0.5,
                 angle: List = [90, 180, 270]):
        self.percentage = percentage
        self.angles = angle

    def __call__(self,
                 img: Image.Image):
        if isinstance(self.angles, List):
            angle = random.choice(self.angles)
        else:
            angle = self.angles

        if random.random() < self.percentage:
            img = F.rotate(img, angle, expand=True, fill=0)

        return img


class SRDataset(Dataset):
    def __init__(
            self,
            images_dir: str = "./datasets/train",
            crop_size: int = 128,
            scale: int = 2,
            mode: str = "train",
            image_format: str = "png",
            preupsample: bool = False,
            jpeg_level: int = 90,
            rgb_range: float = 1.0,
    ):
        super(SRDataset, self).__init__()
        self.image_path_list = sorted(glob.glob(images_dir + "/*." + image_format))
        self.crop_size = crop_size
        self.scale = scale
        self.mode = mode
        self.preupsample = preupsample
        self.jpeg_level = jpeg_level
        self.rgb_range = rgb_range

        self.degrade = BicubicDownsample(self.scale)
        self.normalize = transforms.Normalize(mean=(0.7888, 0.8542, 0.9605),
                                              std=(1.0, 1.0, 1.0))

        if self.mode == "train":
            self.transforms = transforms.Compose([
                # transforms.RandomCrop(self.crop_size),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # RandomRotation(0.5, [90, 180, 270])
            ])
        elif self.mode == "valid":
            self.transforms = transforms.Compose([
                # transforms.CenterCrop(self.crop_size),
            ])
        else:
            raise ValueError("The mode must be either 'train' or 'valid'.")

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, index: int):
        image_hr = Image.open(self.image_path_list[index]).convert('RGB')
        image_hr = self.transforms(image_hr)

        if self.rgb_range != 1:
            image_hr = F.pil_to_tensor(image_hr).float()
        else:
            image_hr = F.to_tensor(np.array(image_hr) / 255.0)
        # image_hr = self.normalize(image_hr)

        image_lr, image_hr = self.degrade(image_hr)
        image_hr, image_lr = image_hr.float(), image_lr.float()

        return {'lr': image_lr, 'hr': image_hr}
