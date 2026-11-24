"""Flickr8k loading, splitting and batching.

Expects the Kaggle layout:
    data/Flickr8k/Images/*.jpg
    data/Flickr8k/captions.txt   (CSV with header: image,caption)
"""
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_captions(captions_file) -> dict[str, list[str]]:
    caps = defaultdict(list)
    with open(captions_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            caps[row["image"].strip()].append(row["caption"].strip())
    return dict(caps)


def split_images(images, val=1000, test=1000, seed=42):
    """Deterministic 6000/1000/1000 split by image (no caption leakage)."""
    images = sorted(images)
    random.Random(seed).shuffle(images)
    return images[val + test:], images[:val], images[val:val + test]


class CaptionTrainSet(Dataset):
    """One item per (image, caption) pair."""

    def __init__(self, image_dir, captions, images, vocab, transform):
        self.image_dir = Path(image_dir)
        self.items = [(img, c) for img in images for c in captions[img]]
        self.vocab, self.transform = vocab, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, cap = self.items[i]
        x = self.transform(Image.open(self.image_dir / img).convert("RGB"))
        return x, torch.tensor(self.vocab.encode(cap))


class CaptionEvalSet(Dataset):
    """One item per image, with all reference captions."""

    def __init__(self, image_dir, captions, images, transform):
        self.image_dir = Path(image_dir)
        self.images, self.captions, self.transform = images, captions, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        x = self.transform(Image.open(self.image_dir / img).convert("RGB"))
        return x, img


def collate_train(batch, pad_idx=0):
    images, caps = zip(*batch)
    return torch.stack(images), pad_sequence(caps, batch_first=True, padding_value=pad_idx)


def collate_eval(batch):
    images, names = zip(*batch)
    return torch.stack(images), list(names)
