import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import yaml
import time
import os
import os.path as osp
import shutil
import torch.backends.cudnn as cudnn
import pickle
import argparse
from tqdm import tqdm
import sys
from PIL import Image, ImageFile, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
import glob
import re
import math
from prettytable import PrettyTable
from utils.iotools import *
import random

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    if not isinstance(imgs[0], torch.Tensor):
        return imgs, pids, camids, camids_batch, viewids, img_paths
    else:
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths


def build_transforms(img_size=(384, 128)):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        print(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        print('\n' + str(table))


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, path_only=False, corruption=None, corruption_severity=3):
        self.dataset = dataset
        self.transform = transform
        self.path_only = path_only
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256,128), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize((0.357, 0.323, 0.328),
                                    (0.252, 0.242, 0.239))
            ])
        self.corruptor = None
        if corruption is not None:
            assert corruption in ["gaussian_noise", "poisson_noise", "defocus_blur", "motion_blur", "brightness"], f"Unsupported corruption type: Found {corruption}"
            assert corruption_severity in [1, 2, 3, 4, 5], f"Invalid corruption severity: Found {corruption_severity}"
            self.corruptor = ImageCorruptor(
                corruptions=[corruption], # can be gaussian_noise, poisson_noise, defocus_blur, motion_blur, brightness
                severity=corruption_severity, # 1 (least) to 5 (most)
                p=0.7, # Always apply
                mode="single",
                seed=42
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        if not self.path_only:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((352, 352)),
            ])
            
            # Apply corruption (before transforms)
            if self.corruptor is not None:
                img = self.corruptor(img, index=index)
            if self.transform is not None:
                img = self.transform(img)
            # transform RGB to BGR for shape [3, H, W]
            # img = img[[2, 1, 0], :, :]
            return img, pid, camid, trackid, img_path #img_path.split('/')[-1]
        else:
            return img, pid, camid, trackid, img_path


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset_T2I(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img, img_path


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)



class ImageCorruptor:
    """
    Callable corruption pipeline.
    - severity: int in [1..5] (example mapping below)
    - p: probability to apply a corruption (or you can force always)
    - mode: 'single' chooses one corruption at random, 'all' applies all in a fixed order
    - seed: base seed; per-index determinism comes from (seed + index)
    """
    def __init__(self, corruptions, severity=3, p=1.0, mode="single", seed=0):
        self.corruptions = corruptions  # list of strings
        self.severity = severity
        self.p = p
        self.mode = mode
        self.seed = seed

    def __call__(self, img: Image.Image, index: int = 0) -> Image.Image:
        rng = random.Random(self.seed + int(index))

        if rng.random() > self.p:
            return img

        if self.mode == "single":
            c = rng.choice(self.corruptions)
            return self.apply_one(img, c, rng)
        elif self.mode == "all":
            for c in self.corruptions:
                img = self.apply_one(img, c, rng)
            return img
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def apply_one(self, img: Image.Image, corruption: str, rng: random.Random) -> Image.Image:
        if corruption == "gaussian_noise":
            return self.gaussian_noise(img, rng)
        if corruption == "poisson_noise":
            return self.poisson_noise(img, rng)
        if corruption == "defocus_blur":
            return self.defocus_blur(img, rng)
        if corruption == "motion_blur":
            return self.motion_blur(img, rng)
        if corruption == "brightness":
            return self.brightness(img, rng)

        raise ValueError(f"Unknown corruption: {corruption}")

    # -----------------------
    # (a) Gaussian noise
    # -----------------------
    def gaussian_noise(self, img: Image.Image, rng: random.Random) -> Image.Image:
        # Severity -> sigma in [0.02, 0.10] approximately
        sigma_map = {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.10}
        sigma = sigma_map.get(self.severity, 0.06)

        x = TF.to_tensor(img)  # [0,1], shape [C,H,W]
        noise = torch.randn_like(x) * sigma
        x = (x + noise).clamp(0.0, 1.0)
        return TF.to_pil_image(x)

    # -----------------------
    # (b) Poisson noise
    # -----------------------
    def poisson_noise(self, img: Image.Image, rng: random.Random) -> Image.Image:
        # Lower "lam" => stronger noise. We simulate by scaling intensities to counts.
        lam_map = {1: 60, 2: 30, 3: 20, 4: 12, 5: 8}
        lam = lam_map.get(self.severity, 20)

        x = TF.to_tensor(img).clamp(0.0, 1.0)
        # Convert to "counts" then sample poisson then rescale.
        # (torch.poisson expects rate parameter; rate must be >= 0)
        counts = x * lam
        noisy = torch.poisson(counts) / lam
        noisy = noisy.clamp(0.0, 1.0)
        return TF.to_pil_image(noisy)

    # -----------------------
    # (c) Defocus blur (Gaussian approximation)
    # -----------------------
    def defocus_blur(self, img: Image.Image, rng: random.Random) -> Image.Image:
        # Severity -> radius
        radius_map = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5, 5: 6.0}
        radius = radius_map.get(self.severity, 3.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    # -----------------------
    # (d) Motion blur (kernel-based)
    # -----------------------
    def motion_blur(self, img: Image.Image, rng: random.Random) -> Image.Image:
        # Kernel sizes by severity
        k_map = {1: 5, 2: 9, 3: 13, 4: 17, 5: 21}
        k = k_map.get(self.severity, 13)
        k = int(k) if int(k) % 2 == 1 else int(k) + 1  # ensure odd

        # Choose among a small set of angles (fast and robust)
        angle = rng.choice([0, 45, 90, 135])

        kernel = self._motion_kernel(k, angle=angle, device="cpu")  # [1,1,k,k]
        x = TF.to_tensor(img).unsqueeze(0)  # [1,C,H,W]
        # apply per-channel via groups
        C = x.shape[1]
        kernel_c = kernel.repeat(C, 1, 1, 1)  # [C,1,k,k]
        x = F.pad(x, pad=[k//2, k//2, k//2, k//2], mode="reflect")
        y = F.conv2d(x, kernel_c, groups=C)
        y = y.squeeze(0).clamp(0.0, 1.0)
        return TF.to_pil_image(y)

    def _motion_kernel(self, k: int, angle: int, device="cpu") -> torch.Tensor:
        ker = torch.zeros((k, k), dtype=torch.float32, device=device)
        mid = k // 2

        if angle == 0:
            ker[mid, :] = 1.0
        elif angle == 90:
            ker[:, mid] = 1.0
        elif angle == 45:
            for i in range(k):
                ker[i, k - 1 - i] = 1.0
        elif angle == 135:
            for i in range(k):
                ker[i, i] = 1.0
        else:
            # fallback: horizontal
            ker[mid, :] = 1.0

        ker = ker / ker.sum().clamp_min(1e-8)
        return ker.view(1, 1, k, k)

    # -----------------------
    # (e) Brightness changes
    # -----------------------
    def brightness(self, img: Image.Image, rng: random.Random) -> Image.Image:
        # Severity -> factor range. factor < 1 darkens; > 1 brightens.
        # Example ranges (you can tune to match corruption benchmarks).
        range_map = {
            1: (0.85, 1.15),
            2: (0.70, 1.30),
            3: (0.55, 1.45),
            4: (0.40, 1.60),
            5: (0.25, 1.75),
        }
        lo, hi = range_map.get(self.severity, (0.55, 1.45))
        factor = rng.uniform(lo, hi)
        return ImageEnhance.Brightness(img).enhance(factor)