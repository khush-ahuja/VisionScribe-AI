"""Train the CNN + LSTM captioner on Flickr8k.

Example:
    python train.py --data data/Flickr8k --backbone densenet121 --epochs 15
    python train.py --data data/Flickr8k --backbone resnet50 --out runs/resnet50   # baseline
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from visionscribe.data import (CaptionEvalSet, CaptionTrainSet, collate_eval, collate_train,
                               eval_transform, load_captions, split_images, train_transform)
from visionscribe.metrics import bleu_scores
from visionscribe.model import CaptionModel
from visionscribe.vocab import END, START, Vocabulary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/Flickr8k")
    p.add_argument("--out", default="runs/densenet121")
    p.add_argument("--backbone", default="densenet121", choices=["densenet121", "resnet50"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--fine-tune", action="store_true", help="also train the CNN backbone")
    p.add_argument("--no-pretrained", action="store_true", help="random-init CNN (for quick tests)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=3, help="early stop on val BLEU-4")
    return p.parse_args()


@torch.no_grad()
def evaluate_bleu(model, loader, captions, vocab, device, beam_size=1):
    model.eval()
    hyps, refs = [], []
    for images, names in loader:
        out = model.generate(images.to(device), vocab.stoi[START], vocab.stoi[END], beam_size=beam_size)
        hyps += [vocab.decode(s) for s in out]
        refs += [captions[n] for n in names]
    return bleu_scores(refs, hyps)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    captions = load_captions(data / "captions.txt")
    train_imgs, val_imgs, _ = split_images(captions.keys())
    vocab = Vocabulary(args.min_freq).build(c for img in train_imgs for c in captions[img])
    vocab.save(out / "vocab.json")
    print(f"device={device} | train={len(train_imgs)} val={len(val_imgs)} images | vocab={len(vocab)}")

    train_dl = DataLoader(CaptionTrainSet(data / "Images", captions, train_imgs, vocab, train_transform()),
                          batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          collate_fn=collate_train, pin_memory=device == "cuda")
    val_dl = DataLoader(CaptionEvalSet(data / "Images", captions, val_imgs, eval_transform()),
                        batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate_eval)

    model = CaptionModel(len(vocab), args.embed_dim, args.hidden_dim,
                         backbone=args.backbone, pretrained=not args.no_pretrained,
                         fine_tune=args.fine_tune).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    config = vars(args) | {"vocab_size": len(vocab)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    best, bad_epochs, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for images, caps in tqdm(train_dl, desc=f"epoch {epoch}", leave=False):
            images, caps = images.to(device), caps.to(device)
            logits = model(images, caps)            # (B, T, V): predicts caps[:, 1:] plus one step
            loss = loss_fn(logits[:, 1:].reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total += loss.item()

        scores = evaluate_bleu(model, val_dl, captions, vocab, device)
        sched.step(scores["bleu4"])
        row = {"epoch": epoch, "train_loss": total / len(train_dl), **scores, "secs": round(time.time() - t0)}
        history.append(row)
        print(json.dumps(row))

        if scores["bleu4"] > best:
            best, bad_epochs = scores["bleu4"], 0
            torch.save({"model": model.state_dict(), "config": config}, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val BLEU-4: {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
