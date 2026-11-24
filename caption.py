"""Caption one or more images with a trained checkpoint.

Example:
    python caption.py --run runs/resnet50 path/to/dog.jpg path/to/beach.jpg
"""
import argparse

import torch
from PIL import Image

from evaluate import load_run
from visionscribe.data import eval_transform
from visionscribe.vocab import END, START


def caption_image(model, vocab, image: Image.Image, device="cpu", beam=3) -> str:
    x = eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    ids = model.generate(x, vocab.stoi[START], vocab.stoi[END], beam_size=beam)[0]
    return vocab.decode(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--run", default="runs/resnet50")
    p.add_argument("--beam", type=int, default=3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_run(args.run, device)
    for path in args.images:
        print(f"{path}: {caption_image(model, vocab, Image.open(path), device, args.beam)}")


if __name__ == "__main__":
    main()
