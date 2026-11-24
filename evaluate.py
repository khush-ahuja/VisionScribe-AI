"""Score trained checkpoints on the held-out Flickr8k test split.

Example (compare DenseNet model against the ResNet-50 baseline):
    python evaluate.py --data data/Flickr8k --runs runs/densenet121 runs/resnet50 --beam 3
Writes results/test_bleu.json and prints a comparison table.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import evaluate_bleu
from visionscribe.data import CaptionEvalSet, collate_eval, eval_transform, load_captions, split_images
from visionscribe.model import CaptionModel
from visionscribe.vocab import Vocabulary


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    vocab = Vocabulary.load(run_dir / "vocab.json")
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    c = ckpt["config"]
    model = CaptionModel(c["vocab_size"], c["embed_dim"], c["hidden_dim"],
                         backbone=c["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Flickr8k")
    p.add_argument("--runs", nargs="+", default=["runs/densenet121"])
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    captions = load_captions(data / "captions.txt")
    _, _, test_imgs = split_images(captions.keys())
    test_dl = DataLoader(CaptionEvalSet(data / "Images", captions, test_imgs, eval_transform()),
                         batch_size=args.batch_size, collate_fn=collate_eval)

    results = {}
    for run in args.runs:
        model, vocab = load_run(run, device)
        results[run] = evaluate_bleu(model, test_dl, captions, vocab, device, beam_size=args.beam)

    print(f"\nTest split: {len(test_imgs)} images, beam={args.beam}")
    print(f"{'run':<24}" + "".join(f"{k:>9}" for k in ["bleu1", "bleu2", "bleu3", "bleu4"]))
    for run, s in results.items():
        print(f"{run:<24}" + "".join(f"{s[k]:>9.4f}" for k in ["bleu1", "bleu2", "bleu3", "bleu4"]))
    if len(results) >= 2:
        (a, sa), (b, sb) = list(results.items())[:2]
        if sb["bleu4"] > 0:
            print(f"\nBLEU-4 change, {a} vs {b}: {100 * (sa['bleu4'] - sb['bleu4']) / sb['bleu4']:+.1f}%")

    Path("results").mkdir(exist_ok=True)
    Path("results/test_bleu.json").write_text(json.dumps({"beam": args.beam, "results": results}, indent=2))


if __name__ == "__main__":
    main()
