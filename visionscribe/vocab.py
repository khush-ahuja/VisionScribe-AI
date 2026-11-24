"""Word-level vocabulary for captions."""
import json
import re
from collections import Counter

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class Vocabulary:
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.itos = [PAD, START, END, UNK]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def build(self, captions):
        counts = Counter(tok for c in captions for tok in tokenize(c))
        for word, n in sorted(counts.items()):
            if n >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        return self

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi[START]] + [self.stoi.get(t, unk) for t in tokenize(text)] + [self.stoi[END]]

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            w = self.itos[int(i)]
            if w == END:
                break
            if w not in (PAD, START):
                words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"min_freq": self.min_freq, "itos": self.itos}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        v = cls(d["min_freq"])
        v.itos = d["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v
