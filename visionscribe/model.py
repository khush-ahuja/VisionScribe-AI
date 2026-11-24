"""CNN encoder + LSTM decoder for image captioning (Show-and-Tell style)."""
import torch
import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    """Pretrained CNN -> fixed-size image embedding.

    backbone: "densenet121" (default) or "resnet50" (baseline for comparison).
    """

    def __init__(self, embed_dim=256, backbone="densenet121", pretrained=True, fine_tune=False):
        super().__init__()
        if backbone == "densenet121":
            net = models.densenet121(weights="DEFAULT" if pretrained else None)
            feat_dim = net.classifier.in_features
            self.cnn = nn.Sequential(net.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        elif backbone == "resnet50":
            net = models.resnet50(weights="DEFAULT" if pretrained else None)
            feat_dim = net.fc.in_features
            self.cnn = nn.Sequential(*list(net.children())[:-1])
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        for p in self.cnn.parameters():
            p.requires_grad = fine_tune
        self.proj = nn.Sequential(nn.Linear(feat_dim, embed_dim), nn.BatchNorm1d(embed_dim))

    def forward(self, images):
        feats = self.cnn(images).flatten(1)
        return self.proj(feats)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=1, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        # Image embedding is fed as the first timestep; teacher forcing on the rest.
        emb = self.dropout(self.embed(captions[:, :-1]))
        inputs = torch.cat([features.unsqueeze(1), emb], dim=1)
        hidden, _ = self.lstm(inputs)
        return self.out(self.dropout(hidden))


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=1,
                 backbone="densenet121", pretrained=True, fine_tune=False, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(embed_dim, backbone, pretrained, fine_tune)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)

    @torch.no_grad()
    def generate(self, images, start_idx, end_idx, max_len=30, beam_size=3):
        """Returns a list of token-id lists, one per image."""
        feats = self.encoder(images)
        if beam_size <= 1:
            return self._greedy(feats, start_idx, end_idx, max_len)
        return [self._beam(f.unsqueeze(0), start_idx, end_idx, max_len, beam_size) for f in feats]

    def _step(self, x, state):
        out, state = self.decoder.lstm(x, state)
        return self.decoder.out(out.squeeze(1)).log_softmax(-1), state

    def _greedy(self, feats, start_idx, end_idx, max_len):
        _, state = self.decoder.lstm(feats.unsqueeze(1))
        tok = torch.full((feats.size(0),), start_idx, device=feats.device)
        seqs = []
        for _ in range(max_len):
            logp, state = self._step(self.decoder.embed(tok).unsqueeze(1), state)
            tok = logp.argmax(-1)
            seqs.append(tok)
        seqs = torch.stack(seqs, 1).tolist()
        return [s[: s.index(end_idx)] if end_idx in s else s for s in seqs]

    def _beam(self, feat, start_idx, end_idx, max_len, k):
        _, state = self.decoder.lstm(feat.unsqueeze(1))
        beams = [([start_idx], 0.0, state)]
        done = []
        for _ in range(max_len):
            cand = []
            for seq, score, st in beams:
                x = self.decoder.embed(torch.tensor([seq[-1]], device=feat.device)).unsqueeze(1)
                logp, new_st = self._step(x, st)
                top = logp[0].topk(k)
                for lp, idx in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((seq + [idx], score + lp, new_st))
            cand.sort(key=lambda c: c[1] / len(c[0]), reverse=True)
            beams = []
            for c in cand:
                (done if c[0][-1] == end_idx else beams).append(c)
                if len(beams) == k:
                    break
            if len(done) >= k or not beams:
                break
        best = max(done or beams, key=lambda c: c[1] / len(c[0]))[0]
        return [t for t in best[1:] if t != end_idx]
